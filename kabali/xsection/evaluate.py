"""Separating selection skill from market exposure, and from luck.

A concentrated long-only portfolio in a bull market will beat a diversified one
almost regardless of how it picks names, because picking ten names out of six
hundred raises volatility and volatility pays when the market rises. Total return
therefore cannot answer the only question worth asking -- does the RANKING add
anything? -- and two separate tests are needed to get at it.

*The random control.* Run the identical engine, over the identical dates, with
the identical eligibility mask and cost model, but rank names at random. The
distribution over many seeds is what "no selection skill, same concentration,
same costs" actually looks like on this panel. If the real strategy sits inside
that distribution, the ranking earned nothing.

*The regression.* Random portfolios control for concentration but not for the
KIND of name a rule prefers. Momentum systematically buys high-beta names, and a
high-beta portfolio beats a random one in a rising market without any skill being
involved. Regressing daily strategy returns on the equal-weight benchmark splits
the outperformance into the part explained by market exposure (beta) and the part
that is not (alpha). Only the second is evidence of an edge.

The two together are what make a null result mean something. Validated on
`controls.synthetic_panel`: a planted edge of 0.02 shows up as alpha t = 3.2 with
beta unchanged at 0.98, so the method detects an edge of the size that would
matter, and it does not mistake beta for one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class AlphaBeta:
    beta: float
    alpha_annual_pct: float
    t_alpha: float
    t_beta_vs_one: float
    correlation: float
    observations: int

    def verdict(self, threshold: float = 2.0) -> str:
        if abs(self.t_alpha) < threshold:
            return (f"no significant alpha (t={self.t_alpha:.2f}); "
                    f"outperformance is consistent with beta {self.beta:.2f}")
        sign = "positive" if self.alpha_annual_pct > 0 else "negative"
        return f"significant {sign} alpha {self.alpha_annual_pct:+.1f}%/yr (t={self.t_alpha:.2f})"


def _aligned_returns(equity: pd.Series, benchmark: pd.Series):
    idx = equity.index.intersection(benchmark.index)
    rs = equity.reindex(idx).pct_change().dropna()
    rb = benchmark.reindex(idx).pct_change().dropna()
    j = rs.index.intersection(rb.index)
    return rs.loc[j], rb.loc[j]


def alpha_beta(equity: pd.Series, benchmark: pd.Series,
               periods_per_year: int = 252) -> AlphaBeta:
    """OLS of strategy returns on benchmark returns, with t-statistics.

    Ordinary least squares with plain standard errors. Daily equity returns are
    mildly autocorrelated and heteroskedastic, so these t-statistics are, if
    anything, slightly generous -- which is the right direction for a test whose
    job is to avoid crediting an edge that is not there.
    """
    rs, rb = _aligned_returns(equity.dropna(), benchmark.dropna())
    n = len(rs)
    if n < 30:
        raise ValueError(f"need at least 30 aligned observations, have {n}")

    beta, alpha = np.polyfit(rb.to_numpy(), rs.to_numpy(), 1)
    resid = rs.to_numpy() - (alpha + beta * rb.to_numpy())
    s = resid.std(ddof=2)
    se_alpha = s / np.sqrt(n)
    se_beta = s / (rb.std(ddof=1) * np.sqrt(n))
    return AlphaBeta(
        beta=float(beta),
        alpha_annual_pct=float(periods_per_year * 100.0 * alpha),
        t_alpha=float(alpha / se_alpha) if se_alpha else float("nan"),
        t_beta_vs_one=float((beta - 1.0) / se_beta) if se_beta else float("nan"),
        correlation=float(np.corrcoef(rs, rb)[0, 1]),
        observations=n,
    )


def random_portfolios(close, open_, mask, config, run_fn, trials: int = 200,
                      seed0: int = 0) -> pd.DataFrame:
    """Same engine, same eligibility, random ranking. One row per trial.

    This is the empirical null for "a concentrated portfolio of this size, paying
    these costs, over these dates". Comparing the strategy against it asks
    strictly about the ranking, because every other moving part is held fixed.
    """
    rows = []
    for k in range(trials):
        rng = np.random.default_rng(seed0 + k)
        scores = pd.DataFrame(rng.random(close.shape),
                              index=close.index, columns=close.columns)
        res = run_fn(close, open_, scores, mask, config)
        e = res.equity.dropna()
        rows.append({"seed": seed0 + k,
                     "total_return_pct": 100.0 * (e.iloc[-1] / e.iloc[0] - 1.0),
                     "final_equity": float(e.iloc[-1]),
                     "trades": int(len(res.trades))})
    return pd.DataFrame(rows)


def percentile_of(value: float, sample: pd.Series) -> float:
    """Where the strategy falls in the random-portfolio distribution, 0-100."""
    return float(100.0 * (sample < value).mean())
