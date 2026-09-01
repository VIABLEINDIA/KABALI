"""Live/forward session driver.

`SessionEngine.run` replays a finished day. This drives the SAME stage methods
from a wall clock against a polling data feed, so a forward session and a replay
differ only in what advances the loop -- an iterator over stored bars, or time
passing. No trading logic is duplicated here; if it were, the two would drift and
the paper record would stop being evidence about the live bot.

The one genuinely new problem is the FORMING BAR. A replay is handed bars that
have all closed. A live feed returns the current five-minute candle mid-flight,
and its high, low and close will all change before it finishes. Acting on it is
lookahead in the most literal sense -- the bot would be trading a close that has
not happened. `_confirmed` drops any bar whose interval has not fully elapsed,
so the loop only ever acts on completed candles.

The loop therefore waits for BAR BOUNDARIES rather than on a fixed timer -- see
`_wait`. Polling every 30 seconds would re-download the whole scan list ten times
per five-minute bar, nine of which cannot contain anything new, for a sustained
~3.3 requests/second across the session at 100 symbols. That is the rate whose
breach cost this project a six-hour throttle during the history build, and it
buys nothing: an unfinished bar is not actionable at any polling frequency.

SQUARE-OFF IS UNCONDITIONAL. It runs on a wall-clock deadline regardless of feed
health, and it runs before the broker's own auto-square-off so the bot chooses
its exit prices rather than inheriting whatever the broker's process produces.
If the feed has failed the positions are still closed, at last known marks.
"""
from __future__ import annotations

import datetime as dt
import logging
import time

import pandas as pd

from kabali.engine.session import (
    EXIT_SQUAREOFF, SessionEngine, SessionResult, fill_source_of,
)
from kabali.risk.book import SessionBook

log = logging.getLogger(__name__)

BAR_SECONDS = {"1m": 60, "5m": 300, "15m": 900}


class LiveSession:
    """Drives one forward session against a polling feed."""

    #: seconds after a bar boundary before we trust the feed to have published it
    BOUNDARY_GRACE_SECONDS = 5

    def __init__(self, cfg, registry, broker, regime, instruments: dict,
                 daily_context: dict, feed, timeframe: str = "5m"):
        self.cfg = cfg
        self.engine = SessionEngine(cfg, registry, broker, regime,
                                    instruments, daily_context)
        self.feed = feed                      # callable: () -> {symbol: DataFrame}
        self.timeframe = timeframe
        self.bar_seconds = BAR_SECONDS.get(timeframe, 300)
        self.pending: list[tuple] = []

    # ------------------------------------------------------------------- run
    def run(self, trade_date: dt.date | None = None,
            poll_seconds: int = 0, now_fn=None,
            max_polls: int | None = None) -> SessionResult:
        """Poll until square-off, then flatten. Returns the session result."""
        now_fn = now_fn or (lambda: pd.Timestamp.now())
        trade_date = trade_date or now_fn().date()

        book = SessionBook(trade_date=trade_date, capital=self.cfg.capital,
                           exit_cost_rate=self.engine.exit_cost_rate)
        self.engine.circuit.reset()
        result = SessionResult(
            trade_date=trade_date, book=book,
            regime_label=getattr(self.engine.regime, "label", "unknown"),
            universe_size=len(self.engine.instruments),
            loss_limit=self.cfg.risk.daily_loss_limit(self.cfg.capital),
            fill_source=fill_source_of(self.engine.broker),
        )

        last_seen: pd.Timestamp | None = None
        polls = 0
        log.info("live session %s starting | %d symbols | regime %s",
                 trade_date, len(self.engine.instruments), result.regime_label)

        while True:
            polls += 1
            if max_polls is not None and polls > max_polls:
                break

            now = pd.Timestamp(now_fn())
            if now.time() >= self.cfg.session.square_off:
                log.warning("square-off deadline %s reached", self.cfg.session.square_off)
                break

            try:
                raw = self.feed()
            except Exception as exc:                       # noqa: BLE001
                # A feed failure must not end the session -- open positions still
                # need managing and squaring off. Retry on the next poll.
                log.error("feed failed: %s", exc)
                self._wait(now_fn, poll_seconds)
                continue

            bars = {s: self._confirmed(d, now) for s, d in raw.items()}
            bars = {s: d for s, d in bars.items() if d is not None and len(d)}
            if not bars:
                self._wait(now_fn, poll_seconds)
                continue

            latest = max(pd.Timestamp(d["timestamp"].iloc[-1]) for d in bars.values())
            if last_seen is not None and latest <= last_seen:
                self._wait(now_fn, poll_seconds)   # no new confirmed bar yet
                continue
            last_seen = latest
            result.timesteps += 1

            current = {s: d.iloc[-1] for s, d in bars.items()}
            marks = {s: float(r["close"]) for s, r in current.items()}

            self.engine._manage_exits(book, current, now, result)

            if self.pending:
                self.engine._fill_pending(self.pending, book, current, now, result)
                self.pending = []

            decision = self.engine.circuit.check_session(book, marks, now)
            if decision.halted:
                if decision.kind != "square_off":
                    result.halted = True
                    result.halt_reason = decision.reason
                    result.halt_kind = decision.kind
                log.warning("circuit: %s", decision.describe())
                if decision.flatten:
                    reason = (EXIT_SQUAREOFF if decision.kind == "square_off" else "circuit")
                    self.engine._flatten(book, current, now, reason, result)
                    break
                self._wait(now_fn, poll_seconds)
                continue

            self.pending = self.engine._scan(book, bars, current, marks, now, result)
            if self.pending:
                log.info("%s: %d entr%s queued for the next bar", now.time(),
                         len(self.pending), "y" if len(self.pending) == 1 else "ies")

            self._wait(now_fn, poll_seconds)

        # Unconditional flatten. Runs even if the feed died.
        if book.positions:
            now = pd.Timestamp(now_fn())
            try:
                raw = self.feed()
                current = {s: d.iloc[-1] for s, d in raw.items() if d is not None and len(d)}
            except Exception as exc:                       # noqa: BLE001
                log.error("feed unavailable at square-off (%s); closing at last marks", exc)
                current = {}
            self.engine._flatten(book, current, now, EXIT_SQUAREOFF, result)
            if book.positions:
                log.critical("POSITIONS STILL OPEN AFTER SQUARE-OFF: %s -- "
                             "close these manually", list(book.positions))

        log.info("live session finished | %s", book.summary())
        return result

    def _wait(self, now_fn, poll_seconds: int) -> None:
        """Sleep until the next bar is due, not for a fixed interval.

        The obvious loop -- poll every 30 seconds -- re-downloads the whole scan
        list ten times per five-minute bar, nine of which cannot contain new
        information. At 100 symbols that is a sustained ~3.3 requests/second for
        the entire session, right at the ceiling whose breach cost this project
        a six-hour throttle during the history build.

        Waiting for the bar boundary instead cuts the load by an order of
        magnitude and loses nothing: a bar that has not closed cannot be acted
        on. The grace period covers the feed publishing a moment late.

        `poll_seconds` is a CEILING, not the interval: 0 (the default) means
        "sleep the whole way to the boundary". Passing a small positive value
        forces tighter polling, which only a test driving a frozen clock has a
        reason to want -- capping it at 30 in production would restore exactly
        the behaviour this method exists to remove.
        """
        secs = int(self.bar_seconds)
        now = pd.Timestamp(now_fn())
        elapsed = (now.hour * 3600 + now.minute * 60 + now.second) % secs
        until_next = secs - elapsed + self.BOUNDARY_GRACE_SECONDS
        delay = until_next if poll_seconds <= 0 else min(until_next, poll_seconds)
        if delay > 0:
            time.sleep(delay)

    # ---------------------------------------------------------------- detail
    def _confirmed(self, df: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame | None:
        """Drop the bar still forming.

        A candle stamped 10:05 on a 5-minute feed covers 10:05..10:10 and is only
        final once 10:10 has passed. Anything else is a partial bar whose close
        has not been decided.
        """
        if df is None or df.empty:
            return None
        d = df.copy()
        d["timestamp"] = pd.to_datetime(d["timestamp"])
        cutoff = pd.Timestamp(now) - pd.Timedelta(int(self.bar_seconds), unit="s")
        return d[d["timestamp"] <= cutoff].sort_values("timestamp").reset_index(drop=True)


def make_dhan_feed(bridge, instruments: dict, timeframe: str = "5m",
                   limiter=None):
    """Build a callable that returns today's intraday bars per symbol.

    Deliberately simple polling rather than a websocket. At a five-minute
    decision cadence the extra complexity of a streaming feed buys nothing, and
    a REST poll that fails is trivially retried on the next cycle where a dropped
    socket is not.
    """
    from kabali.core import load_bars

    def feed() -> dict:
        today = dt.date.today().isoformat()
        out = {}
        for symbol, inst in instruments.items():
            try:
                if limiter is not None:
                    limiter.acquire()
                out[symbol] = load_bars(bridge, inst, timeframe, today, today, refresh=True)
            except Exception as exc:                       # noqa: BLE001
                log.debug("feed miss %s: %s", symbol, exc)
        return out

    return feed
