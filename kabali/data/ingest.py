"""Bulk history ingestion for the NSE cash-equity candidate pool.

Scale is the whole problem here. The candidate pool is ~2,650 EQ-series names and
`research.datastore` chunks daily history at 365 days, so five years is five calls
per symbol -- about 13,000 requests. Done serially at the bridge's 0.25s floor
that is over an hour of wall clock, most of it spent waiting on network latency
rather than on the rate limit.

So ingestion is staged and parallel:

*Staged* -- a short recent window is fetched for every candidate first and used to
drop the illiquid majority. Only survivors get the full five years. Roughly 80% of
the pool fails the turnover gate, so this avoids downloading years of history for
names that can never be traded at our size.

*Parallel* -- each worker owns its own `DhanBridge` (the SDK client is not
documented as thread-safe, and the bridge's own throttle is per-instance), while a
single process-wide `RateLimiter` enforces the aggregate request ceiling. Workers
overlap latency; the exchange still sees one well-behaved request stream.

Resumability is inherited, not implemented: `load_bars` returns from the parquet
cache when the cache already covers the requested range, so a re-run after an
interruption skips everything already stored.
"""
from __future__ import annotations

import datetime as dt
import logging
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import pandas as pd

from kabali.core import DhanBridge, Instrument, ScripMaster, load_bars

log = logging.getLogger(__name__)

NSE_EQ_SEGMENT = "NSE_EQ"
_TEST_SCRIP = re.compile(r"NSETEST")
EQUITY = "EQUITY"

# Aggregate ceiling across all workers. History is a bulk background job with no
# latency requirement -- there is nothing to gain by crowding the limit and a
# throttled account to lose.
#
# MEASURED, not guessed. A full build at 4.0/s ran at ~2.4 min per 100 symbols
# for the first thousand, then a single concurrent CLI call breached the limit
# and Dhan returned DH-904. The throttle that followed was not a brief backoff:
# the next hundred symbols took 346 minutes, and the build finished 6.5 hours
# late with 265 failed symbols. Dhan appears to punish a breach for far longer
# than it lasts, so the safe rate is well under whatever triggers one.
#
# The bridge's own retry (3 attempts, 1.5s * attempt) is nowhere near long
# enough to ride that out, and it is not supposed to be -- the fix is to not
# breach, and to never run two jobs against this account at once.
DEFAULT_REQUESTS_PER_SECOND = 2.5
DEFAULT_WORKERS = 3


class RateLimiter:
    """Process-wide sliding-window limiter, shared across ingest threads."""

    def __init__(self, rate_per_second: float):
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be > 0")
        self.rate = rate_per_second
        self._times: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._times and now - self._times[0] >= 1.0:
                    self._times.popleft()
                if len(self._times) < self.rate:
                    self._times.append(now)
                    return
                sleep_for = 1.0 - (now - self._times[0])
            time.sleep(max(sleep_for, 0.005))


class _BridgePool:
    """One DhanBridge per worker thread, created lazily."""

    def __init__(self, limiter: RateLimiter):
        self._local = threading.local()
        self._limiter = limiter

    def get(self) -> DhanBridge:
        bridge = getattr(self._local, "bridge", None)
        if bridge is None:
            # min_interval=0: pacing is the shared limiter's job, not each
            # bridge's. Leaving the per-instance throttle on would multiply the
            # intended interval by the worker count.
            bridge = DhanBridge(min_interval=0.0)
            _wrap_with_limiter(bridge, self._limiter)
            self._local.bridge = bridge
        return bridge


def _wrap_with_limiter(bridge: DhanBridge, limiter: RateLimiter) -> None:
    """Route every bridge request through the shared limiter."""
    original = bridge._call

    def limited(method: str, *args, **kwargs):
        limiter.acquire()
        return original(method, *args, **kwargs)

    bridge._call = limited  # type: ignore[method-assign]


def candidate_instruments(scrip: ScripMaster | None = None,
                          series: tuple[str, ...] = ("EQ",)) -> list[Instrument]:
    """Every NSE cash-equity instrument in the requested series.

    Series matters more than it looks. NSE lists ~9,900 rows under NSE/E, but
    only the EQ series is the normal rolling-settlement segment that supports
    intraday MIS. SG (sovereign gold), SM/ST (SME), BE (trade-for-trade) and GS
    either forbid intraday or are far too thin, and a bot that ranked them
    alongside real equities would happily pick names it can never exit.
    """
    scrip = scrip or ScripMaster.load(refresh=False)
    df = scrip.df
    want_series = {s.upper() for s in series}

    mask = (
        (df[scrip._c_exch].astype(str).str.upper().str.strip() == "NSE")
        & (df[scrip._c_seg].astype(str).str.upper().str.strip() == "E")
        & (df[scrip._c_instr].astype(str).str.upper().str.strip() == EQUITY)
        & (df[scrip._c_series].astype(str).str.upper().str.strip().isin(want_series))
    )
    pool = df[mask]

    out: list[Instrument] = []
    seen: set[str] = set()
    for _, row in pool.iterrows():
        sec_id = str(row[scrip._c_secid]).split(".")[0]
        if sec_id in seen:
            continue
        seen.add(sec_id)
        symbol = str(row[scrip._c_symbol]).strip().upper()
        # NSE publishes live test scrips (011NSETEST ... 101NSETEST) in the EQ
        # series. They resolve, they have security ids, and they return data --
        # they are simply not a company. ULTIMATE hit the same rows from the
        # currency side, where the only future with a live expiry was 11NSETEST.
        if _TEST_SCRIP.search(symbol):
            continue
        out.append(Instrument(
            symbol=symbol,
            security_id=sec_id,
            exchange_segment=NSE_EQ_SEGMENT,
            instrument_type=EQUITY,
            lot_size=1,
            tick_size=float(row[scrip._c_tick]) if scrip._c_tick else 0.05,
            display_name=symbol,
        ))
    out.sort(key=lambda i: i.symbol)
    return out


@dataclass
class IngestReport:
    """What ingestion actually achieved, including what it could not fetch."""

    timeframe: str
    start: str
    end: str
    requested: int = 0
    fetched: int = 0
    empty: int = 0
    failed: int = 0
    elapsed_s: float = 0.0
    failures: dict[str, str] = field(default_factory=dict)

    def render(self) -> str:
        rate = self.fetched / self.elapsed_s if self.elapsed_s > 0 else 0.0
        lines = [
            f"ingest {self.timeframe} {self.start}..{self.end}",
            f"  requested {self.requested:,} | fetched {self.fetched:,} | "
            f"empty {self.empty:,} | failed {self.failed:,}",
            f"  elapsed {self.elapsed_s / 60:.1f} min ({rate:.1f} symbols/s)",
        ]
        if self.failures:
            shown = list(self.failures.items())[:5]
            lines.append("  first failures:")
            lines += [f"    {sym}: {err[:110]}" for sym, err in shown]
            if len(self.failures) > 5:
                lines.append(f"    ... and {len(self.failures) - 5:,} more")
        return "\n".join(lines)


def date_window(years: int, end: dt.date | None = None) -> tuple[str, str]:
    """Inclusive ISO date window reaching `years` back from `end` (default today)."""
    end = end or dt.date.today()
    start = end - dt.timedelta(days=int(round(years * 365.25)))
    return start.isoformat(), end.isoformat()


def ingest(instruments: list[Instrument], timeframe: str, start: str, end: str,
           workers: int = DEFAULT_WORKERS,
           rate_per_second: float = DEFAULT_REQUESTS_PER_SECOND,
           refresh: bool = False,
           progress_every: int = 100,
           on_bars=None) -> IngestReport:
    """Fetch and cache bars for every instrument.

    `on_bars(instrument, df)` is called for each non-empty result, from a worker
    thread, so a caller can accumulate derived statistics without a second pass
    over the parquet cache. It must be thread-safe.
    """
    report = IngestReport(timeframe=timeframe, start=start, end=end, requested=len(instruments))
    if not instruments:
        return report

    limiter = RateLimiter(rate_per_second)
    pool = _BridgePool(limiter)
    lock = threading.Lock()
    t0 = time.monotonic()
    done = 0

    def work(inst: Instrument):
        bridge = pool.get()
        return inst, load_bars(bridge, inst, timeframe, start, end, refresh=refresh)

    with ThreadPoolExecutor(max_workers=workers) as pool_exec:
        futures = {pool_exec.submit(work, i): i for i in instruments}
        for fut in as_completed(futures):
            inst = futures[fut]
            try:
                _, df = fut.result()
            except Exception as exc:  # noqa: BLE001 - recorded, never raised
                with lock:
                    report.failed += 1
                    report.failures[inst.symbol] = f"{type(exc).__name__}: {exc}"
                    done += 1
            else:
                with lock:
                    if df is None or df.empty:
                        report.empty += 1
                    else:
                        report.fetched += 1
                        if on_bars is not None:
                            on_bars(inst, df)
                    done += 1
            if progress_every and done % progress_every == 0:
                elapsed = time.monotonic() - t0
                log.info("ingest %s: %d/%d in %.1f min (%d failed)",
                         timeframe, done, len(instruments), elapsed / 60, report.failed)

    report.elapsed_s = time.monotonic() - t0
    return report


def liquidity_frame(rows: list[dict]) -> pd.DataFrame:
    """Assemble per-symbol liquidity statistics collected during ingest."""
    if not rows:
        return pd.DataFrame(columns=["symbol", "security_id", "bars", "last_close",
                                     "median_volume", "median_turnover"])
    return pd.DataFrame(rows).sort_values("median_turnover", ascending=False).reset_index(drop=True)


def summarise_liquidity(inst: Instrument, df: pd.DataFrame, tail: int = 60) -> dict:
    """Liquidity statistics for one symbol, from its most recent `tail` bars.

    Turnover uses the MEDIAN of close*volume rather than the mean. A single
    block deal or index-rebalance day can lift a mean turnover by an order of
    magnitude on an otherwise untradeable name, and the mean is exactly the
    statistic that lets such a name into a liquidity-filtered universe.
    """
    recent = df.tail(tail)
    close = pd.to_numeric(recent["close"], errors="coerce")
    volume = pd.to_numeric(recent.get("volume", pd.Series(dtype=float)), errors="coerce").fillna(0)
    turnover = (close * volume).dropna()
    return {
        "symbol": inst.symbol,
        "security_id": inst.security_id,
        "bars": int(len(df)),
        "last_close": float(close.iloc[-1]) if len(close.dropna()) else float("nan"),
        "median_volume": float(volume.median()) if len(volume) else 0.0,
        "median_turnover": float(turnover.median()) if len(turnover) else 0.0,
    }
