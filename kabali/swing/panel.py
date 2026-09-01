"""Aligned daily-bar panel and the vectorised indicators the ranking needs.

A cross-sectional strategy asks a question the per-symbol loop is the wrong
shape for: on this date, how does every eligible name rank against every other?
Answering that one symbol at a time means recomputing indicators inside the date
loop, which for 850 symbols over 1,200 dates is a million redundant regressions
and turns a two-minute study into an afternoon.

So the bars are loaded once into wide frames -- dates down, symbols across, one
frame per OHLCV field -- and every indicator is computed for the whole panel in
one pass. The date loop then only slices rows that already exist.

Two properties matter more than the speed:

CAUSALITY IS STRUCTURAL. Every indicator here is a trailing window ending at
row i, so row i holds only what was knowable at that session's close. The tests
pin it by recomputing on truncated history and demanding the same value.

THE PANEL NEVER FETCHES. It reads `kabali.data.cache`, which is filesystem-only.
A symbol missing from the cache is a counted omission, never an unbounded
network wait discovered mid-study. That failure mode already cost this project
hours once.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from kabali.data.cache import cached_bars

log = logging.getLogger(__name__)

FIELDS = ("open", "high", "low", "close", "volume")


@dataclass
class Panel:
    """Aligned OHLCV for many symbols on one trading calendar."""

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    security_ids: dict[str, str]
    missing: list[str]

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)

    def __len__(self) -> int:
        return len(self.close)

    def slice(self, start=None, end=None) -> "Panel":
        def cut(df: pd.DataFrame) -> pd.DataFrame:
            out = df
            if start is not None:
                out = out[out.index >= pd.Timestamp(start)]
            if end is not None:
                out = out[out.index <= pd.Timestamp(end)]
            return out
        return Panel(*(cut(getattr(self, f)) for f in FIELDS),
                     security_ids=self.security_ids, missing=self.missing)

    def describe(self) -> str:
        if not len(self):
            return "empty panel"
        return (f"{len(self.symbols)} symbols x {len(self)} sessions "
                f"({self.dates.min().date()} -> {self.dates.max().date()}), "
                f"{len(self.missing)} not cached")


def build_panel(instruments, calendar: pd.DatetimeIndex | None = None,
                min_bars: int = 250, timeframe: str = "D") -> Panel:
    """Read cached daily bars for `instruments` onto one shared calendar.

    Symbols with fewer than `min_bars` rows on disk are dropped outright: a name
    with 30 bars cannot be ranked on a 90-day regression and would only add a
    NaN column. Symbols listed part-way through the window keep NaNs before
    their first bar, which is correct -- they were not investable then, and the
    min_periods guards exclude them until they are.
    """
    frames: dict[str, dict[str, pd.Series]] = {f: {} for f in FIELDS}
    sec_ids: dict[str, str] = {}
    missing: list[str] = []

    for inst in instruments:
        df = cached_bars(inst, timeframe, min_bars=min_bars)
        if df is None:
            missing.append(inst.symbol)
            continue
        idx = pd.DatetimeIndex(pd.to_datetime(df["timestamp"])).normalize()
        if idx.has_duplicates:
            keep = ~idx.duplicated(keep="last")
            df, idx = df[keep], idx[keep]
        for f in FIELDS:
            frames[f][inst.symbol] = pd.Series(
                pd.to_numeric(df[f], errors="coerce").to_numpy(), index=idx)
        sec_ids[inst.symbol] = inst.security_id

    if not sec_ids:
        empty = pd.DataFrame(index=pd.DatetimeIndex([], name="date"))
        return Panel(empty, empty.copy(), empty.copy(), empty.copy(), empty.copy(),
                     security_ids={}, missing=missing)

    wide = {f: pd.DataFrame(frames[f]).sort_index() for f in FIELDS}
    index = wide["close"].index if calendar is None else pd.DatetimeIndex(calendar).normalize()
    wide = {f: d.reindex(index) for f, d in wide.items()}
    for d in wide.values():
        d.index.name = "date"

    return Panel(wide["open"], wide["high"], wide["low"], wide["close"], wide["volume"],
                 security_ids=sec_ids, missing=missing)


# ------------------------------------------------------------------ indicators
# Wide-frame twins of `research.indicators`. The tests assert these agree with
# the per-series originals column by column -- a divergent second definition of
# ATR would silently change every stop distance in the study.

def sma(frame: pd.DataFrame, length: int) -> pd.DataFrame:
    return frame.rolling(length, min_periods=length).mean()


def true_range(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """Wide-frame true range.

    `fmax`, not `maximum`: the first row has no previous close, so two of the
    three candidates are NaN, and `maximum` would propagate that into a NaN
    true range. The reference implementation takes a row-wise pandas max, which
    skips NaN, and the difference is not cosmetic -- one NaN at the head shifts
    the whole Wilder average and every ATR-derived stop with it.
    """
    prev = close.shift(1)
    return pd.DataFrame(
        np.fmax.reduce([
            (high - low).to_numpy(dtype=float),
            (high - prev).abs().to_numpy(dtype=float),
            (low - prev).abs().to_numpy(dtype=float),
        ]),
        index=close.index, columns=close.columns)


def atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame,
        length: int) -> pd.DataFrame:
    """Wilder ATR, matching `research.indicators.atr` exactly."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean()


def rolling_slope_r2(close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Annualised exponential-regression slope x R^2, for every column at once.

    The same statistic as `kabali.universe.factors.slope_r2`, derived in closed
    form so it can be rolled rather than looped. Fitting log(price) against
    t = 0..w-1 with fixed spacing makes the t-side of the regression constant:

        beta = Sxy / Stt      with Stt = sum (t - tbar)^2
        R^2  = beta^2 * Stt / Syy

    so both fall out of rolling sums of y, y^2 and t*y. The equivalence to the
    looped original is asserted in the tests, not assumed here.
    """
    if window < 3:
        raise ValueError("slope window must be >= 3")

    y = np.log(close.where(close > 0))
    t = np.arange(window, dtype=float)
    t_mean = float(t.mean())
    s_tt = float(((t - t_mean) ** 2).sum())

    sum_y = y.rolling(window, min_periods=window).sum()
    sum_y2 = (y ** 2).rolling(window, min_periods=window).sum()
    s_xy = _rolling_dot(y, t) - t_mean * sum_y

    beta = s_xy / s_tt
    s_yy = sum_y2 - sum_y ** 2 / window

    # A perfectly flat window has zero variance to explain; the fit is exact,
    # so R^2 is 1 rather than 0/0. Guarding here keeps the NaNs meaning
    # "not enough history" and nothing else.
    r2 = (beta ** 2 * s_tt / s_yy.where(s_yy > 0)).clip(upper=1.0)
    r2 = r2.mask((s_yy <= 0) & s_yy.notna(), 1.0)

    annual = np.exp((beta * 252.0).clip(lower=-10.0, upper=10.0)) - 1.0
    return annual * r2


def _rolling_dot(y: pd.DataFrame, t: np.ndarray) -> pd.DataFrame:
    """Rolling dot product of each trailing window with the fixed vector `t`.

    A rolling `.apply` over 850 columns is the slow path this module exists to
    avoid, so the windows are formed once as a strided view and contracted in a
    single matmul. NaNs propagate, which is what min_periods would have done.
    """
    w = len(t)
    arr = y.to_numpy(dtype=float)
    out = np.full(arr.shape, np.nan)
    if arr.shape[0] >= w:
        windows = np.lib.stride_tricks.sliding_window_view(arr, w, axis=0)
        out[w - 1:] = windows @ t
    return pd.DataFrame(out, index=y.index, columns=y.columns)


def rolling_median_turnover(close: pd.DataFrame, volume: pd.DataFrame,
                            window: int) -> pd.DataFrame:
    """Median of close*volume over the trailing window.

    Median rather than mean for the reason `summarise_liquidity` gives: one
    block deal lifts a mean turnover by an order of magnitude and lets an
    untradeable name through a liquidity filter.
    """
    return (close * volume).rolling(window, min_periods=max(2, window // 2)).median()


def max_abs_gap(open_: pd.DataFrame, close: pd.DataFrame, window: int) -> pd.DataFrame:
    """Largest absolute overnight gap, as a fraction, over the trailing window."""
    prev = close.shift(1)
    gap = ((open_ - prev) / prev.where(prev > 0)).abs()
    return gap.rolling(window, min_periods=max(2, window // 2)).max()
