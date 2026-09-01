"""Shared measurements used by more than one intraday rule.

Kept out of `base.py` so the interface stays an interface. Everything here reads
only confirmed bars from a `SymbolContext`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from kabali.core import atr

# A stop closer than this to the entry is inside the noise: normal bid-ask
# bounce would trigger it, and the resulting loss rate has nothing to do with
# whether the idea was right. Expressed as a fraction of price.
MIN_STOP_FRACTION = 0.0025      # 25 bps
# A stop wider than this makes the position too small to clear fixed costs at
# our capital, so the setup is skipped rather than sized down to nothing.
MAX_STOP_FRACTION = 0.030       # 3%


def intraday_atr(ctx, window: int = 14) -> float:
    """ATR over confirmed intraday bars, falling back to a range estimate.

    Early in the session there are fewer bars than the ATR window, and Wilder
    ATR is NaN until it fills. Rather than disabling every rule for the first
    hour, fall back to the mean true range of whatever has closed so far.
    """
    d = ctx.intraday
    if len(d) < 2:
        return float("nan")
    high = pd.to_numeric(d["high"], errors="coerce")
    low = pd.to_numeric(d["low"], errors="coerce")
    close = pd.to_numeric(d["close"], errors="coerce")
    if len(d) >= window:
        v = atr(high, low, close, window).iloc[-1]
        if np.isfinite(v) and v > 0:
            return float(v)
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    v = float(tr.dropna().mean()) if len(tr.dropna()) else float("nan")
    return v if np.isfinite(v) and v > 0 else float("nan")


def relative_volume(ctx, lookback: int = 20) -> float:
    """Latest bar volume over the median of the preceding `lookback` bars.

    Confirmation, not a signal. A breakout on below-average volume is the
    profile of a level being brushed rather than taken, and it is the most
    common way an opening-range rule ends up buying the high of the day.
    """
    d = ctx.intraday
    if len(d) < 3 or "volume" not in d:
        return float("nan")
    vol = pd.to_numeric(d["volume"], errors="coerce").fillna(0.0)
    latest = float(vol.iloc[-1])
    base = vol.iloc[max(0, len(vol) - 1 - lookback):-1]
    med = float(base.median()) if len(base) else 0.0
    if med <= 0:
        return float("nan")
    return latest / med


def stop_is_sane(entry: float, stop: float) -> bool:
    """Reject stops that are inside the noise or too wide to size."""
    if not (np.isfinite(entry) and np.isfinite(stop)) or entry <= 0:
        return False
    frac = abs(entry - stop) / entry
    return MIN_STOP_FRACTION <= frac <= MAX_STOP_FRACTION


def clamp_stop(entry: float, stop: float, side: str) -> float:
    """Push a too-tight stop out to the minimum sane distance."""
    frac = abs(entry - stop) / entry if entry > 0 else 0.0
    if frac >= MIN_STOP_FRACTION:
        return stop
    dist = entry * MIN_STOP_FRACTION
    return entry - dist if side == "long" else entry + dist


def session_extremes(ctx) -> tuple[float, float]:
    d = ctx.intraday
    return (float(pd.to_numeric(d["high"], errors="coerce").max()),
            float(pd.to_numeric(d["low"], errors="coerce").min()))


def before(ts, cutoff) -> bool:
    """True when the timestamp's clock time is at or before `cutoff`."""
    return pd.Timestamp(ts).time() <= cutoff
