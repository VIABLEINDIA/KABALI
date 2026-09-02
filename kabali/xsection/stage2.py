"""Stage 2 of `docs/xsection_hypothesis.md`: is the signal tradable long-only?

Runs only because Stage 1 passed. Stage 1 established that the ranking separates
future winners from future losers in this sample; that is a different claim from
"a Rs 40,000 CNC account can harvest it", and this file asks only the second.

WHAT CHANGES FROM STAGE 1, AND WHY EACH CHANGE COSTS SOMETHING.

*Long-only.* CNC has no shorting. Stage 1's spread was +28.12%/yr, of which the
short leg contributed 16 points -- the bottom decile lagged the eligible
universe by 16.07pp while the top beat it by 12.05pp. Most of the measured
effect is therefore unreachable before this file runs a single session.

*Concentrated.* Stage 1 held 100 names a side. Rs 40,000 against a Rs 5,000
minimum position notional holds eight. A signal measured on a 100-name basket
is not the same estimator as the same signal in an 8-name book: the
idiosyncratic variance the basket averages away is exactly what a small account
must carry. Both are reported, because the difference between them is a
statement about the account rather than about the signal.

*Costs are real.* Stage 1 modelled none. Delivery STT is 0.1% per side, so a
round trip is ~28bps before slippage, and the engine charges it on the fill.

*Fills are next-open.* `portfolio.run` decides on the close of t and fills at the
open of t+1, with integer shares and no borrowing. That engine predates this
panel and is reused unchanged rather than rewritten to flatter the result.

A NOTE ON THE PRE-REGISTERED BAR. The design fixed two things: a benchmark
levered to the strategy's realised beta, and Sharpe as the criterion. Those
overlap more than intended -- Sharpe is invariant to leverage, so scaling the
benchmark by beta leaves its Sharpe unchanged and the beta match does no work in
the Sharpe test. It does real work in the *return* comparison, where an
unlevered benchmark would flatter a high-beta strategy. Both are reported here
and the Sharpe bar stands as the pre-registered decision rule; the redundancy is
recorded rather than quietly resolved, because noticing it after seeing results
is precisely when a criterion is most tempting to restate.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from kabali.core import ensure_core_importable
from kabali.xsection import momentum as mom
from kabali.xsection import portfolio as pf

ensure_core_importable()
from research.metrics import max_drawdown_pct, sharpe_ratio  # noqa: E402

log = logging.getLogger(__name__)

#: Fixed before this panel existed -- the defaults already in XSecConfig, used
#: by the first cross-sectional run. Chosen here for that reason: a position
#: count picked against these results would be the selection pressure the
#: pre-registration exists to refuse.
PRIMARY_POSITIONS = 10
PRIMARY_REBALANCE = 21


def equal_weight_eligible(close: pd.DataFrame, mask: pd.DataFrame,
                          start) -> pd.Series:
    """Equal-weight hold of the names the strategy was allowed to choose from.

    Benchmarking against every listed name would compare the strategy to a
    universe it could not have bought, crediting the liquidity screen as if it
    were skill. `fill_method=None` keeps a return from being computed across a
    gap the portfolio could not have traded through.
    """
    c = close.loc[start:]
    m = mask.loc[start:].reindex_like(c).fillna(False)
    rets = c.pct_change(fill_method=None).where(m)
    return (1.0 + rets.mean(axis=1, skipna=True).fillna(0.0)).cumprod()


@dataclass
class Leg:
    label: str
    equity: pd.Series
    sharpe: float
    cagr_pct: float
    max_dd_pct: float
    total_pct: float

    def render(self) -> str:
        return (f"{self.label:<26} {self.total_pct:+9.1f}%  "
                f"CAGR {self.cagr_pct:+6.2f}%  Sharpe {self.sharpe:5.2f}  "
                f"maxDD {self.max_dd_pct:5.1f}%")


def _leg(label: str, equity: pd.Series) -> Leg:
    e = equity.dropna()
    years = max((e.index[-1] - e.index[0]).days / 365.25, 1e-9)
    total = e.iloc[-1] / e.iloc[0] - 1.0
    return Leg(label, e, sharpe_ratio(e),
               100.0 * ((1.0 + total) ** (1.0 / years) - 1.0),
               max_drawdown_pct(e), 100.0 * total)


@dataclass
class Stage2Result:
    strategy: Leg
    benchmark: Leg
    beta_matched: Leg
    beta: float
    t_beta_vs_one: float
    alpha_annual_pct: float
    t_alpha: float
    n_positions: int
    trades: int
    cost_rupees: float
    cost_pct_capital: float

    def passed(self) -> bool:
        """The pre-registered bar: beat the benchmark's Sharpe after costs."""
        return bool(np.isfinite(self.strategy.sharpe)
                    and self.strategy.sharpe > self.benchmark.sharpe)

    def render(self) -> str:
        lines = [
            self.strategy.render(),
            self.benchmark.render(),
            self.beta_matched.render(),
            (f"beta {self.beta:.2f} (t vs 1 = {self.t_beta_vs_one:+.2f}) | "
             f"alpha {self.alpha_annual_pct:+.2f}%/yr (t={self.t_alpha:+.2f})"),
            (f"{self.trades:,} fills, costs Rs {self.cost_rupees:,.0f} "
             f"({self.cost_pct_capital:.1f}% of capital)"),
            (f"SHARPE BAR: strategy {self.strategy.sharpe:.2f} vs benchmark "
             f"{self.benchmark.sharpe:.2f} -> "
             f"{'PASS' if self.passed() else 'FAIL'}"),
        ]
        return "\n".join(lines)


def run_stage2(close: pd.DataFrame, open_: pd.DataFrame, turnover: pd.DataFrame,
               n_positions: int = PRIMARY_POSITIONS,
               rebalance_days: int = PRIMARY_REBALANCE,
               capital: float = 40_000.0) -> Stage2Result:
    """Long-only run, against the universe it could actually buy."""
    from kabali.xsection.evaluate import alpha_beta

    scores = mom.momentum(close)
    mask = mom.eligible(close, turnover)
    cfg = pf.XSecConfig(capital=capital, n_positions=n_positions,
                        rebalance_days=rebalance_days)
    res = pf.run(close, open_, scores, mask, cfg)

    start = res.equity.index[0]
    bench = equal_weight_eligible(close, mask, start) * capital
    bench = bench.reindex(res.equity.index).ffill()

    ab = alpha_beta(res.equity, bench)

    # Lever the benchmark to the strategy's realised beta. Sharpe is unchanged
    # by this (see the module docstring); the return comparison is not.
    br = bench.pct_change().fillna(0.0)
    matched = capital * (1.0 + ab.beta * br).cumprod()

    cost = float(res.trades["cost"].sum()) if len(res.trades) else 0.0
    return Stage2Result(
        strategy=_leg(f"strategy (N={n_positions})", res.equity),
        benchmark=_leg("equal-weight eligible", bench),
        beta_matched=_leg(f"benchmark x beta {ab.beta:.2f}", matched),
        beta=ab.beta, t_beta_vs_one=ab.t_beta_vs_one,
        alpha_annual_pct=ab.alpha_annual_pct, t_alpha=ab.t_alpha,
        n_positions=n_positions, trades=len(res.trades),
        cost_rupees=cost, cost_pct_capital=100.0 * cost / capital)
