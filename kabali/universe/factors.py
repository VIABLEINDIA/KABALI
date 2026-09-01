"""Per-symbol factors for intraday universe selection.

What "high momentum" means for an intraday bot is not what it means for a
monthly-rebalanced momentum portfolio, and conflating the two is the first way
this selection goes wrong.

A classic momentum screen ranks on trailing return, because it intends to hold
for months and harvest continuation. This bot holds for hours and is flat by
15:10, so trailing return alone is close to useless: a stock that rose 80% in a
year in ten gap-ups and 240 flat sessions has superb momentum and nothing an
intraday rule can capture. What an intraday strategy actually needs is

  1. RANGE      -- enough daily travel to clear ~15 bps of round-trip cost
  2. CLEANNESS  -- travel that trends rather than chops, or stop-outs eat it
  3. LIQUIDITY  -- depth to enter and exit our size without moving the price
  4. DIRECTION  -- a relative-strength tilt the regime can point long or short

Every factor below serves one of those four. All are computed on confirmed bars
only, through `research.indicators`, whose causality is pinned by ULTIMATE's
tests. Nothing here may read a bar later than `as_of`.

Factors are RAW here and unitless -- cross-sectional standardisation happens in
`selector.py`, because a z-score is only meaningful against the peer set being
ranked on that same day.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from kabali.core import atr

# Trading-day lookbacks. 252 ~ one year, 21 ~ one month.
LOOKBACK_SHORT = 21
LOOKBACK_MED = 63
LOOKBACK_LONG = 126
LOOKBACK_YEAR = 252
SKIP_RECENT = 21          # the "-1" in 12-1 momentum
ATR_WINDOW = 14
SLOPE_WINDOW = 90
EFFICIENCY_WINDOW = 63
TURNOVER_FAST = 20
TURNOVER_SLOW = 100

# Longest history any factor needs. Below this the row is not scored at all
# rather than scored on partial data -- a symbol with 40 bars would otherwise
# get a NaN-heavy composite that ranks unpredictably against full-history peers.
MIN_BARS = LOOKBACK_YEAR + SKIP_RECENT + 5


@dataclass(frozen=True)
class Factors:
    """Raw factor values for one symbol as of one date."""

    symbol: str
    security_id: str
    as_of: pd.Timestamp
    bars: int
    close: float

    # direction
    ret_21: float
    ret_63: float
    ret_126: float
    ret_252_21: float          # 12-1 momentum, the standard skip-a-month form

    # cleanness
    efficiency_ratio: float    # Kaufman: |net move| / summed absolute moves
    slope_r2: float            # annualised log-regression slope * R^2 (Clenow)

    # range
    atr_pct: float             # ATR14 / close -- the daily move available
    range_pct: float           # mean (high-low)/close -- the intraday move available
    gap_pct: float             # mean |overnight gap| -- range NOT capturable intraday

    # liquidity
    median_turnover: float
    turnover_expansion: float  # 20d median turnover / 100d median turnover
    median_volume: float
    zero_volume_frac: float

    def as_dict(self) -> dict:
        d = asdict(self)
        d["as_of"] = pd.Timestamp(self.as_of)
        return d


def _safe_return(close: pd.Series, lookback: int, skip: int = 0) -> float:
    """Simple return over `lookback` bars ending `skip` bars ago."""
    need = lookback + skip + 1
    if len(close) < need:
        return float("nan")
    end = close.iloc[-1 - skip]
    start = close.iloc[-1 - skip - lookback]
    if not np.isfinite(start) or not np.isfinite(end) or start <= 0:
        return float("nan")
    return float(end / start - 1.0)


def efficiency_ratio(close: pd.Series, window: int = EFFICIENCY_WINDOW) -> float:
    """Kaufman efficiency ratio: how directly the price got where it went.

    1.0 is a straight line; near 0 is chop that ends where it started. This is
    the factor that separates a stock which travelled 30% in a trend from one
    which travelled 30% in aggregate and finished flat. The second is the one
    that stops out an intraday trend rule repeatedly.
    """
    if len(close) < window + 1:
        return float("nan")
    seg = close.iloc[-(window + 1):]
    net = abs(float(seg.iloc[-1]) - float(seg.iloc[0]))
    path = float(seg.diff().abs().sum())
    if not np.isfinite(path) or path <= 0:
        return float("nan")
    return net / path


def slope_r2(close: pd.Series, window: int = SLOPE_WINDOW) -> float:
    """Annualised exponential regression slope weighted by fit quality.

    Fitting log price against time gives a compounding growth rate that is
    comparable across price levels. Multiplying by R^2 discounts a steep slope
    that the data does not actually support -- a jump-then-flat series can show
    a large slope with a poor fit, and undiscounted it would outrank a genuine
    steady trend.
    """
    if len(close) < window:
        return float("nan")
    seg = pd.to_numeric(close.iloc[-window:], errors="coerce")
    if seg.isna().any() or (seg <= 0).any():
        return float("nan")
    y = np.log(seg.to_numpy(dtype=float))
    x = np.arange(window, dtype=float)
    x_c = x - x.mean()
    y_c = y - y.mean()
    denom = float((x_c ** 2).sum())
    if denom <= 0:
        return float("nan")
    beta = float((x_c * y_c).sum() / denom)
    ss_tot = float((y_c ** 2).sum())
    r2 = 1.0 if ss_tot <= 0 else max(0.0, 1.0 - float(((y_c - beta * x_c) ** 2).sum()) / ss_tot)
    # exp(beta * 252) - 1 overflows on a degenerate fit; clamp the exponent.
    annual = math.exp(max(min(beta * 252.0, 10.0), -10.0)) - 1.0
    return annual * r2


def compute_factors(symbol: str, security_id: str, bars: pd.DataFrame,
                    as_of: pd.Timestamp | str | None = None) -> Factors | None:
    """Factor row for one symbol, using only bars at or before `as_of`.

    Returns None when the symbol has too little history to score, which the
    caller must treat as "not rankable", not as "ranked last".
    """
    if bars is None or bars.empty:
        return None
    df = bars
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
        if as_of is not None:
            df = df[df["timestamp"] <= pd.Timestamp(as_of)]
    if len(df) < MIN_BARS:
        return None

    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    volume = pd.to_numeric(df.get("volume", pd.Series(index=df.index, dtype=float)),
                           errors="coerce").fillna(0.0)
    if close.tail(LOOKBACK_YEAR).isna().all():
        return None

    last_close = float(close.iloc[-1])
    if not np.isfinite(last_close) or last_close <= 0:
        return None

    atr_series = atr(high, low, close, ATR_WINDOW)
    atr_last = float(atr_series.iloc[-1]) if len(atr_series.dropna()) else float("nan")

    rng = ((high - low) / close.replace(0.0, np.nan)).tail(LOOKBACK_SHORT)
    prev_close = close.shift(1)
    gap = ((df["open"].astype(float) - prev_close).abs() / prev_close.replace(0.0, np.nan))
    gap_recent = gap.tail(LOOKBACK_SHORT)

    turnover = (close * volume)
    t_fast = float(turnover.tail(TURNOVER_FAST).median())
    t_slow = float(turnover.tail(TURNOVER_SLOW).median())
    expansion = t_fast / t_slow if np.isfinite(t_slow) and t_slow > 0 else float("nan")

    vol_recent = volume.tail(TURNOVER_SLOW)
    zero_frac = float((vol_recent <= 0).mean()) if len(vol_recent) else 1.0

    return Factors(
        symbol=symbol,
        security_id=security_id,
        as_of=pd.Timestamp(df["timestamp"].iloc[-1]) if "timestamp" in df.columns
        else pd.Timestamp(as_of) if as_of is not None else pd.Timestamp("NaT"),
        bars=int(len(df)),
        close=last_close,
        ret_21=_safe_return(close, LOOKBACK_SHORT),
        ret_63=_safe_return(close, LOOKBACK_MED),
        ret_126=_safe_return(close, LOOKBACK_LONG),
        ret_252_21=_safe_return(close, LOOKBACK_YEAR - SKIP_RECENT, skip=SKIP_RECENT),
        efficiency_ratio=efficiency_ratio(close),
        slope_r2=slope_r2(close),
        atr_pct=atr_last / last_close if np.isfinite(atr_last) else float("nan"),
        range_pct=float(rng.mean()) if len(rng.dropna()) else float("nan"),
        gap_pct=float(gap_recent.mean()) if len(gap_recent.dropna()) else float("nan"),
        median_turnover=t_slow,
        turnover_expansion=expansion,
        median_volume=float(vol_recent.median()) if len(vol_recent) else 0.0,
        zero_volume_frac=zero_frac,
    )


def factors_frame(rows: list[Factors]) -> pd.DataFrame:
    """Collect factor rows into a frame indexed by symbol."""
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([r.as_dict() for r in rows]).set_index("symbol")
