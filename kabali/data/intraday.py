"""Intraday bar access: session slicing and the per-symbol daily context.

Two responsibilities, both about giving the engine exactly what it may see.

SESSION SLICING splits a long intraday series into per-day frames and trims each
to the tradeable window. Bars outside `market_open .. market_close` do exist in
the feed and would quietly corrupt the opening range and session VWAP if left in.

DAILY CONTEXT supplies the three per-symbol numbers the strategies need from
BEFORE the session -- previous close, yesterday's ATR, and median volume for the
capacity cap. Each is computed strictly from bars dated earlier than the session,
which is the whole point: `prev_close` taken from a frame that includes today is
today's close, and a gap rule fed that number is reading the answer.
"""
from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd

from kabali.core import atr, load_bars

log = logging.getLogger(__name__)

DEFAULT_TIMEFRAME = "5m"
ATR_WINDOW = 14
VOLUME_WINDOW = 100


def fetch_intraday(bridge, instrument, start: str, end: str,
                   timeframe: str = DEFAULT_TIMEFRAME,
                   refresh: bool = False) -> pd.DataFrame:
    """Cached intraday bars. Chunking at Dhan's 90-day cap is the datastore's job."""
    return load_bars(bridge, instrument, timeframe, start, end, refresh=refresh)


def split_sessions(bars: pd.DataFrame, session_cfg,
                   min_bars: int = 10) -> dict[dt.date, pd.DataFrame]:
    """Split an intraday series into one frame per trading day, window-trimmed.

    Days with fewer than `min_bars` are dropped. A half-session (a muhurat
    session, or a symbol that listed intraday) has an opening range and a VWAP
    that mean something different from a full day's, and mixing them into a
    paper record silently changes what the record is evidence of.
    """
    if bars is None or bars.empty:
        return {}
    df = bars.copy()
    ts = pd.to_datetime(df["timestamp"])
    df["_date"] = ts.dt.date
    df["_time"] = ts.dt.time

    in_window = (df["_time"] >= session_cfg.market_open) & (df["_time"] <= session_cfg.market_close)
    df = df[in_window]

    out: dict[dt.date, pd.DataFrame] = {}
    for day, chunk in df.groupby("_date", sort=True):
        if len(chunk) < min_bars:
            continue
        out[day] = (chunk.drop(columns=["_date", "_time"])
                    .sort_values("timestamp").reset_index(drop=True))
    return out


def daily_context(daily_bars: pd.DataFrame, as_of: dt.date) -> dict:
    """Previous close, previous ATR and median volume, strictly before `as_of`.

    Returns an empty dict when there is not enough prior history, which callers
    must treat as "cannot trade this symbol today" rather than substituting
    defaults. A gap rule with a guessed previous close is worse than no gap rule.
    """
    if daily_bars is None or daily_bars.empty:
        return {}
    d = daily_bars.copy()
    ts = pd.to_datetime(d["timestamp"])
    prior = d[ts.dt.date < as_of]
    if len(prior) < ATR_WINDOW + 2:
        return {}

    close = pd.to_numeric(prior["close"], errors="coerce")
    high = pd.to_numeric(prior["high"], errors="coerce")
    low = pd.to_numeric(prior["low"], errors="coerce")
    volume = pd.to_numeric(prior.get("volume", pd.Series(dtype=float)),
                           errors="coerce").fillna(0.0)

    a = atr(high, low, close, ATR_WINDOW)
    prev_close = float(close.iloc[-1])
    atr_daily = float(a.iloc[-1]) if len(a.dropna()) else float("nan")
    med_vol = float(volume.tail(VOLUME_WINDOW).median()) if len(volume) else 0.0

    if not np.isfinite(prev_close) or prev_close <= 0:
        return {}
    return {
        "prev_close": prev_close,
        "atr_daily": atr_daily,
        "median_volume": med_vol,
        "daily": prior.tail(250).reset_index(drop=True),
    }


def align_sessions(per_symbol: dict[str, dict[dt.date, pd.DataFrame]]) -> list[dt.date]:
    """Trading dates present for at least one symbol, in order."""
    days: set[dt.date] = set()
    for sessions in per_symbol.values():
        days.update(sessions)
    return sorted(days)
