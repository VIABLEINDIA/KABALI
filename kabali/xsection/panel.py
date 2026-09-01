"""Aligned daily price panel for cross-sectional work.

Everything KABALI and ULTIMATE tested before this was a *time-series* rule: each
symbol judged against its own history, one `bars` frame at a time. A
cross-sectional rule asks a different question -- how does this name rank against
the other six hundred today -- and that question cannot be asked until every name
sits on one shared calendar. Building that panel is the whole job here.

Three things make it more than a concat:

*Corporate actions.* A 1:2 split reads as a -50% day. In a time-series backtest
that is one bad trade. In a cross-sectional ranking it pins the name to the
bottom of the momentum sort for the next twelve months, and the portfolio then
buys whatever the split displaced. `research.corporate_actions` already carries a
detector careful enough to separate splits from genuine crashes, so every series
passes through it and the events are counted in the report rather than silently
swallowed.

*Alignment without invention.* Names list, halt, and go untraded. Missing days
stay NaN and are never forward-filled into tradeable prices: a forward-filled
close is a price at which nobody could transact, and a momentum rule fed flat
synthetic prices reads them as low volatility and ranks them up. NaN propagates
into the signal, which excludes the name. That is correct and must not be papered
over.

*Survivorship, stated rather than solved.* These are the names listed and liquid
TODAY. Every company that delisted, was acquired, or fell out of liquidity across
the window is absent, and for a momentum rule that absence flatters -- the losers
that kept losing until they were removed are exactly what a momentum sort would
have shed or shorted. `Panel.caveats()` carries this so no report can be written
without it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from kabali.core import ensure_core_importable

ensure_core_importable()
from research import corporate_actions as ca  # noqa: E402

log = logging.getLogger(__name__)

SEGMENT = "NSE_EQ"
#: Five years of daily bars. The universe builder's staged ingest granted this
#: depth only to names that passed its liquidity screen, so the threshold doubles
#: as a coarse liquidity filter.
MIN_BARS = 1200
#: A date counts as a session only if this fraction of the panel actually traded.
#: Guards against a few stale files inventing a day the market never held.
MIN_CROSS_SECTION = 0.30

#: A bar is a bad print if it moves more than this and the move is largely undone
#: on the very next session. Set below the 35% corporate-action floor on purpose:
#: the screen has to run FIRST so that a bad print is never mistaken for a split.
SPIKE_MOVE = 0.15
#: How much of the move must reverse to call it a print rather than a real event.
SPIKE_REVERSION = 0.60
#: More bad prints than this and the series is not repairable bar by bar -- the
#: whole name is dropped. See the module docstring on scale interleaving.
MAX_BAD_PRINTS = 3

_OHLCV = ["timestamp", "open", "high", "low", "close", "volume"]


@dataclass
class Panel:
    """Wide daily frames on one calendar: index = date, columns = symbol."""

    close: pd.DataFrame
    open: pd.DataFrame
    volume: pd.DataFrame
    turnover: pd.DataFrame
    actions: pd.DataFrame = field(default_factory=pd.DataFrame)
    dropped: dict = field(default_factory=dict)

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    def slice(self, start: str | None = None, end: str | None = None) -> "Panel":
        sel = slice(start, end)
        return Panel(
            close=self.close.loc[sel], open=self.open.loc[sel],
            volume=self.volume.loc[sel], turnover=self.turnover.loc[sel],
            actions=self.actions, dropped=self.dropped,
        )

    def caveats(self) -> list[str]:
        """Limitations that must travel with any result computed from this panel."""
        return [
            "SURVIVORSHIP: the panel holds names listed and liquid today; "
            "delisted and acquired companies are absent. For a momentum rule "
            "this biases results UPWARD, so a negative result here is "
            "trustworthy and a positive one is provisional.",
            "Liquidity is screened on trailing turnover at each rebalance, but "
            "panel membership itself is not point-in-time.",
            "Corporate actions are detected from the price series alone; a split "
            "shallower than the detector's 35% floor stays unadjusted.",
            "No dividends. Total return is understated by roughly the dividend "
            "yield -- close to neutral across a cross-sectional sort, but not "
            "zero.",
        ]

    def describe(self) -> str:
        traded = self.close.notna().sum(axis=1)
        return (
            f"{len(self.symbols)} symbols x {len(self.dates)} sessions  "
            f"{self.dates[0].date()} .. {self.dates[-1].date()}\n"
            f"cross-section per day: min {traded.min()}, "
            f"median {int(traded.median())}, max {traded.max()}\n"
            f"corporate actions adjusted: {len(self.actions)}"
        )


def bad_prints(close: pd.Series) -> list[int]:
    """Positions of bars whose move is undone by the next session.

    A split is permanent; a bad print reverts. That single distinction is what
    `research.corporate_actions` cannot see, because its corroborating signal --
    the session opening already in the new scale -- is satisfied by a corrupt bar
    whose OPEN is corrupt too. HATSUN on 2022-12-21 closes at 468 between two
    sessions at 912 and 900, opens at 467, and passes the open check cleanly.
    Adjusting on it would have halved a year of that name's history and shown the
    ranking a stock that doubled overnight.

    Returns integer positions into `close`, so the caller can mask the bar rather
    than rescale everything before it.
    """
    c = close.to_numpy(dtype=float)
    if len(c) < 3:
        return []
    r = c[1:] / c[:-1] - 1.0
    out = []
    for i in range(len(r) - 1):
        if abs(r[i]) > SPIKE_MOVE and r[i] * r[i + 1] < 0 \
                and abs(r[i + 1]) > SPIKE_REVERSION * abs(r[i]):
            out.append(i + 1)
    return out


def _symbol_map(screen_csv: Path) -> dict[str, str]:
    """security_id -> symbol, from the liquidity screen the ingest already wrote.

    Read off disk rather than resolved through `ScripMaster` so that panel
    building stays offline. Talking to Dhan is the ingest's job, not research's.
    """
    df = pd.read_csv(screen_csv, dtype={"security_id": str})
    df = df.drop_duplicates(subset="security_id", keep="first")
    return dict(zip(df.security_id, df.symbol))


def load_panel(bars_root: str | Path = "D:/ULTIMATE/data/bars",
               screen_csv: str | Path = "state/liquidity_screen.csv",
               min_bars: int = MIN_BARS,
               adjust_actions: bool = True) -> Panel:
    """Build the aligned panel from the local parquet cache. No network."""
    bars_root, screen_csv = Path(bars_root), Path(screen_csv)
    names = _symbol_map(screen_csv)

    closes: dict[str, pd.Series] = {}
    opens: dict[str, pd.Series] = {}
    vols: dict[str, pd.Series] = {}
    action_rows: list[dict] = []
    dropped = {"unnamed": 0, "too_short": 0, "unreadable": 0, "corrupt_series": []}
    masked_bars = 0

    for path in sorted(bars_root.glob(f"{SEGMENT}_*_D.parquet")):
        sid = path.stem[len(SEGMENT) + 1:-2]
        symbol = names.get(sid)
        if symbol is None:
            dropped["unnamed"] += 1
            continue
        try:
            df = pd.read_parquet(path, columns=_OHLCV)
        except Exception:
            dropped["unreadable"] += 1
            continue
        if len(df) < min_bars:
            dropped["too_short"] += 1
            continue

        df = (df.drop_duplicates(subset="timestamp", keep="last")
                .sort_values("timestamp")
                .reset_index(drop=True))

        spikes = bad_prints(df.close)
        if len(spikes) > MAX_BAD_PRINTS:
            # Not a series with a few bad ticks: MOTHERSON interleaves whole
            # sessions at two price scales 62 times over five years. There is no
            # honest per-bar repair for that, and a name that periodically jumps
            # 50% would sit at the top of a momentum sort on the strength of a
            # data defect.
            dropped["corrupt_series"].append(symbol)
            continue
        if spikes:
            df.loc[df.index[spikes], ["open", "high", "low", "close", "volume"]] = np.nan
            masked_bars += len(spikes)

        if adjust_actions:
            report = ca.detect(df)
            if report.count:
                df, report = ca.adjust(df, report)
                for date, ratio, label in zip(report.dates, report.ratios, report.nearest):
                    action_rows.append({"symbol": symbol, "date": pd.Timestamp(date),
                                        "ratio": float(ratio), "nearest": label})

        df = df.set_index("timestamp")
        closes[symbol], opens[symbol], vols[symbol] = df.close, df.open, df.volume

    if not closes:
        raise RuntimeError(f"no usable series under {bars_root} with >= {min_bars} bars")

    close = pd.DataFrame(closes).sort_index()
    open_ = pd.DataFrame(opens).reindex(close.index)
    volume = pd.DataFrame(vols).reindex(close.index)

    live = close.notna().sum(axis=1) >= MIN_CROSS_SECTION * close.shape[1]
    close, open_, volume = close[live], open_[live], volume[live]

    dropped["masked_bars"] = masked_bars
    return Panel(close=close, open=open_, volume=volume,
                 turnover=close * volume,
                 actions=pd.DataFrame(action_rows), dropped=dropped)
