"""Stage 1 of `docs/xsection_hypothesis.md`: does the momentum signal exist?

This is a RESEARCH instrument, not a strategy. It is long-short, and CNC has no
shorting -- stock lending on Indian mid-caps is too thin and expensive to treat
as available. Going long-short here is what removes market beta from the
measurement, which is the entire point: the first cross-sectional run's
outperformance was explained by beta 1.18 (t = 5.99) with alpha t = 0.46, and no
long-only construction can separate those two. Whether a surviving signal can be
*traded* long-only is Stage 2's question and is deliberately not asked here.

TWO ESTIMATORS, NEITHER CROWNED AFTER THE FACT. The decile spread compares the
tails and assumes nothing about the shape of the relationship. Fama-MacBeth
regresses across every rankable name and is the more efficient estimator if the
relationship is monotonic, which is an assumption the spread does not need.
They answer slightly different questions, so both are reported and disagreement
between them is itself informative. The spread is primary for the decision rule,
fixed in advance.

WHY NEWEY-WEST. Formation windows overlap: a name ranked in January is ranked
on twelve months of returns that mostly also determine its February rank. That
induces serial correlation in the spread, and plain OLS standard errors would
read it as independent evidence and overstate significance. Bartlett kernel at
21 lags, one formation month.

THE SIGNAL IS NOT RE-CHOSEN HERE. Formation and skip come from
`kabali.xsection.momentum` unchanged -- the canonical 12-1 construction, fixed
before the first run. Re-picking them against this panel is the selection
pressure the pre-registration exists to refuse.

MEMORY. The twenty-year panel is ~4,900 x ~12,000. Materialising a signal panel
costs three copies of that and does not fit, so scores are computed per date
from two rows of close, and the daily spread from two more. Only the
eligibility mask is a full array, and it is boolean and built in column chunks.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from kabali.xsection.momentum import (LOOKBACK, MIN_PRICE, MIN_TURNOVER, SKIP,
                                      TURNOVER_WINDOW)

log = logging.getLogger(__name__)

#: Formation cadence in sessions. One month, matching the pre-registered design.
STEP = 21

#: Names per side of the spread. The pre-registered primary specification.
N_SIDE = 100

#: Newey-West lag length: one formation month of overlap.
NW_LAGS = 21

#: Below this the panel cannot resolve the hypothesis at any effect size the
#: literature considers plausible, and a run is a pilot rather than a verdict.
#: Measured in `scripts/power_xsection.py`: a 100x100 spread needs ~22 years for
#: a 5%/yr effect.
MIN_YEARS_FOR_VERDICT = 15.0


# ------------------------------------------------------------------ inference
def newey_west(x: np.ndarray, lags: int = NW_LAGS) -> tuple[float, float, float]:
    """Mean, autocorrelation-robust standard error, and t-statistic.

    Bartlett kernel. Returns the SE *of the mean*, so the t-statistic tests the
    mean against zero -- which is the hypothesis, not whether the series has any
    structure.
    """
    x = np.asarray(x, dtype="float64")
    x = x[np.isfinite(x)]
    n = len(x)
    if n < lags + 5:
        return float("nan"), float("nan"), float("nan")
    mu = x.mean()
    e = x - mu
    gamma0 = float(e @ e) / n
    var = gamma0
    for lag in range(1, lags + 1):
        w = 1.0 - lag / (lags + 1.0)             # Bartlett
        cov = float(e[lag:] @ e[:-lag]) / n
        var += 2.0 * w * cov
    var = max(var, 0.0)
    se = float(np.sqrt(var / n))
    return float(mu), se, (mu / se if se > 0 else float("nan"))


# ---------------------------------------------------------------- eligibility
def eligibility_mask(close: pd.DataFrame, turnover: pd.DataFrame,
                     lookback: int = LOOKBACK, skip: int = SKIP,
                     window: int = TURNOVER_WINDOW,
                     min_turnover: float = MIN_TURNOVER,
                     min_price: float = MIN_PRICE,
                     chunk: int = 2000) -> np.ndarray:
    """The point-in-time screen, as a boolean array, built in column chunks.

    Identical in meaning to `momentum.eligible`, but the rolling median is the
    one genuinely expensive step here -- on the full panel it is another float
    array the size of the panel. Chunking bounds peak memory to `chunk` columns
    regardless of how many names the twenty years contain.
    """
    n, m = close.shape
    mask = np.zeros((n, m), dtype=bool)
    cl = close.to_numpy()
    for a in range(0, m, chunk):
        b = min(a + chunk, m)
        t = turnover.iloc[:, a:b]
        liquid = (t.rolling(window, min_periods=window // 2).median()
                  >= min_turnover).to_numpy()
        block = cl[:, a:b]
        priced = block >= min_price
        tradeable = np.isfinite(block)
        formed = np.zeros_like(priced)
        # Both ends of the formation window must exist: a name that was not
        # trading a year ago has no momentum to measure.
        formed[lookback:] = (np.isfinite(block[skip:n - lookback + skip])
                             & np.isfinite(block[:n - lookback]))
        mask[:, a:b] = liquid & priced & tradeable & formed
    return mask


def _scores(cl: np.ndarray, i: int, lookback: int, skip: int) -> np.ndarray:
    """12-1 momentum for every name on row i, from two rows of close."""
    past, recent = cl[i - lookback], cl[i - skip]
    with np.errstate(divide="ignore", invalid="ignore"):
        return recent / past - 1.0


# ------------------------------------------------------- estimator A: spread
@dataclass
class SpreadResult:
    """Daily returns of the long-short spread, and the test on their mean."""
    daily: pd.Series
    mean_daily: float
    se_daily: float
    t_stat: float
    n_side: int
    rebalances: int

    @property
    def annual_pct(self) -> float:
        return 252.0 * 100.0 * self.mean_daily

    @property
    def ci95_annual(self) -> tuple[float, float]:
        h = 1.96 * 252.0 * 100.0 * self.se_daily
        return self.annual_pct - h, self.annual_pct + h

    @property
    def years(self) -> float:
        """Span of the SPREAD, which is not the span of the panel.

        A rebalance needs 4x n_side eligible names, and NSE does not supply 400
        liquid ones until 2010 -- so a 20.7-year panel yields 16.6 years of
        testable spread. Reporting the panel's length here overstated the sample
        the statistics were actually computed on, which is exactly the kind of
        inflation the power analysis exists to prevent.
        """
        if self.daily.empty:
            return 0.0
        return (self.daily.index[-1] - self.daily.index[0]).days / 365.25

    def render(self) -> str:
        lo, hi = self.ci95_annual
        span = (f"{self.daily.index[0].date()} -> {self.daily.index[-1].date()}"
                if not self.daily.empty else "empty")
        return (f"decile spread {self.n_side}x{self.n_side}: "
                f"{self.annual_pct:+.2f}%/yr  t={self.t_stat:+.2f} (Newey-West) "
                f"95% CI [{lo:+.1f}, {hi:+.1f}]  "
                f"{len(self.daily):,} sessions, {self.rebalances:,} rebalances  "
                f"[{span}, {self.years:.1f}y]")


def decile_spread(close: pd.DataFrame, mask: np.ndarray, n_side: int = N_SIDE,
                  step: int = STEP, lookback: int = LOOKBACK,
                  skip: int = SKIP) -> SpreadResult:
    """Top n_side minus bottom n_side by momentum, reformed every `step`.

    Baskets are fixed between rebalances and equal-weighted each session, which
    is the standard academic construction: it measures the signal rather than a
    particular weight-drift policy, and Stage 2 is where an implementable scheme
    gets tested.
    """
    cl = close.to_numpy()
    n = cl.shape[0]
    longs: np.ndarray = np.array([], dtype=int)
    shorts: np.ndarray = np.array([], dtype=int)
    out_dates, out_vals, rebals = [], [], 0

    for i in range(lookback + 1, n):
        if (i - lookback - 1) % step == 0:
            s = _scores(cl, i, lookback, skip)
            ok = mask[i] & np.isfinite(s)
            idx = np.flatnonzero(ok)
            if len(idx) >= 4 * n_side:
                order = idx[np.argsort(s[idx], kind="mergesort")]
                shorts, longs = order[:n_side], order[-n_side:]
                rebals += 1
            else:
                longs = shorts = np.array([], dtype=int)
        if len(longs) == 0:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            r = cl[i] / cl[i - 1] - 1.0
        rl, rs = r[longs], r[shorts]
        rl, rs = rl[np.isfinite(rl)], rs[np.isfinite(rs)]
        if len(rl) < n_side // 2 or len(rs) < n_side // 2:
            continue
        out_dates.append(close.index[i])
        out_vals.append(float(rl.mean() - rs.mean()))

    daily = pd.Series(out_vals, index=pd.DatetimeIndex(out_dates), name="spread")
    mu, se, t = newey_west(daily.to_numpy())
    return SpreadResult(daily, mu, se, t, n_side, rebals)


# -------------------------------------------------- estimator B: Fama-MacBeth
@dataclass
class FamaMacBethResult:
    slopes: pd.Series
    mean: float
    se: float
    t_stat: float
    periods: int
    avg_names: float

    def render(self) -> str:
        return (f"Fama-MacBeth: {100 * self.mean:+.2f}%/period  "
                f"t={self.t_stat:+.2f}  {self.periods:,} periods, "
                f"{self.avg_names:,.0f} names per cross-section")


def fama_macbeth(close: pd.DataFrame, mask: np.ndarray, step: int = STEP,
                 lookback: int = LOOKBACK, skip: int = SKIP) -> FamaMacBethResult:
    """Regress forward return on cross-sectional momentum rank, period by period.

    Ranks are scaled to [0, 1] within each cross-section, so the slope reads
    directly as the expected return difference between the best- and
    worst-ranked name -- the same quantity the spread estimates, by a route that
    uses every name instead of the tails.
    """
    cl = close.to_numpy()
    n = cl.shape[0]
    slopes, widths, dates = [], [], []

    for i in range(lookback + 1, n - step, step):
        s = _scores(cl, i, lookback, skip)
        ok = mask[i] & np.isfinite(s)
        with np.errstate(divide="ignore", invalid="ignore"):
            fwd = cl[i + step] / cl[i] - 1.0
        ok &= np.isfinite(fwd)
        idx = np.flatnonzero(ok)
        if len(idx) < 50:
            continue
        order = np.argsort(s[idx], kind="mergesort")
        rank = np.empty(len(idx), dtype="float64")
        rank[order] = np.arange(len(idx))
        rank /= max(len(idx) - 1, 1)              # scale to [0, 1]
        y = fwd[idx]
        x = rank - rank.mean()
        denom = float(x @ x)
        if denom <= 0:
            continue
        slopes.append(float(x @ (y - y.mean()) / denom))
        widths.append(len(idx))
        dates.append(close.index[i])

    ser = pd.Series(slopes, index=pd.DatetimeIndex(dates), name="slope")
    # Non-overlapping forward windows, so plain SE is the standard FM statistic;
    # a short lag still guards against residual month-to-month persistence.
    mu, se, t = newey_west(ser.to_numpy(), lags=3)
    return FamaMacBethResult(ser, mu, se, t, len(ser),
                             float(np.mean(widths)) if widths else float("nan"))


# ------------------------------------------------------------------ reporting
@dataclass
class Stage1Result:
    spread: SpreadResult
    fm: FamaMacBethResult
    first_half: SpreadResult | None
    second_half: SpreadResult | None
    years: float

    @property
    def confirmatory(self) -> bool:
        return self.years >= MIN_YEARS_FOR_VERDICT

    def verdict(self) -> str:
        """PASS / FAIL / INCONCLUSIVE, exactly as pre-registered."""
        t = self.spread.t_stat
        if not np.isfinite(t):
            return "INCONCLUSIVE"
        if t <= -2.0 or self.spread.mean_daily < 0:
            return "FAIL"
        if t < 2.0:
            return "INCONCLUSIVE"
        if self.first_half and self.second_half:
            if not (self.first_half.mean_daily > 0 and self.second_half.mean_daily > 0):
                return "INCONCLUSIVE"
        return "PASS (stage 1 only -- stage 2 still required)"

    def render(self) -> str:
        lines = [self.spread.render(), self.fm.render()]
        if self.first_half and self.second_half:
            lines.append(
                f"halves: first {self.first_half.annual_pct:+.2f}%/yr "
                f"(t={self.first_half.t_stat:+.2f}), "
                f"second {self.second_half.annual_pct:+.2f}%/yr "
                f"(t={self.second_half.t_stat:+.2f})")
        lines.append(f"sample {self.years:.1f} years of testable spread "
                     f"(panel is longer; rebalances need 4x n_side names)")
        if not self.confirmatory:
            lines.append(
                f"PILOT ONLY -- {self.years:.1f} years is below the "
                f"{MIN_YEARS_FOR_VERDICT:.0f} the power analysis requires. "
                "This result cannot settle the hypothesis in either direction.")
        lines.append(f"VERDICT: {self.verdict()}")
        return "\n".join(lines)


def run_stage1(close: pd.DataFrame, turnover: pd.DataFrame,
               n_side: int = N_SIDE, step: int = STEP) -> Stage1Result:
    """Both estimators plus the pre-registered sample split."""
    mask = eligibility_mask(close, turnover)
    spread = decile_spread(close, mask, n_side=n_side, step=step)
    fm = fama_macbeth(close, mask, step=step)

    half = len(close) // 2
    first = second = None
    if half > LOOKBACK + 2 * step:
        first = decile_spread(close.iloc[:half], mask[:half], n_side, step)
        second = decile_spread(close.iloc[half:], mask[half:], n_side, step)

    # The tested span is the spread's, not the panel's: rebalances cannot start
    # until 4x n_side names pass the screen.
    return Stage1Result(spread, fm, first, second, spread.years)
