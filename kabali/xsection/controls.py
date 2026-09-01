"""Synthetic panels that prove what the engine can and cannot see.

ULTIMATE's negative findings were credible for one reason: the same machinery
found a planted edge and rejected pure noise. Without those two runs, "we tested
it and found nothing" is indistinguishable from "our backtester is broken", and
the second explanation is by far the more common one.

The cross-sectional engine is new code, so it earns the same two checks before
any conclusion is drawn from it:

*Negative control.* 606 independent random walks. There is no cross-sectional
structure to find -- last year's winners are not next month's winners, because
nothing connects them. A momentum portfolio here must land on top of the
equal-weight benchmark before costs and below it after, because all it does is
pay spread and STT to shuffle between statistically identical names. Any other
outcome means the engine is manufacturing return.

*Positive control.* The same walks, except each name's drift is tilted by its own
trailing 12-1 return. The edge is planted, it is exactly the effect the strategy
claims to harvest, and the engine must find it clearly. If it cannot, a null
result on real data says nothing about the market and everything about the code.

Both panels are generated from a seeded RNG so a control run is reproducible;
an unreproducible control is not a control.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Idiosyncratic daily volatility, roughly an Indian mid cap's.
DAILY_VOL = 0.020
#: Common-factor daily volatility. Without this every name is independent, the
#: equal-weight benchmark averages 606 uncorrelated walks, and its variance very
#: nearly vanishes -- the first version of this file produced a benchmark Sharpe
#: of 8.2 and a 0.29% maximum drawdown, which is not a market and makes any
#: comparison against it meaningless. Real names share a market factor, so the
#: benchmark has to carry one too.
MARKET_VOL = 0.009
#: Sessions per formation window, matching the live signal's definition.
LOOKBACK = 252
SKIP = 21


def synthetic_panel(n_symbols: int = 606, n_days: int = 1237, seed: int = 7,
                    momentum_strength: float = 0.0,
                    drift: float = 0.0004,
                    start: str = "2021-08-30") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random-walk closes and opens, optionally with a planted momentum effect.

    `momentum_strength` is the fraction of a daily standard deviation added as
    drift to a name sitting one cross-sectional standard deviation above the mean
    on trailing 12-1 return. At 0.0 the panel is pure noise; at 0.10 the tilt is
    small enough to be realistic and large enough that a working engine must see
    it.

    The market-wide `drift` is deliberately non-zero: the real panel ran through
    a strong bull market, and a control with no drift would compare a strategy
    against a flat benchmark and flatter it for no reason.
    """
    rng = np.random.default_rng(seed)
    symbols = [f"SYN{i:04d}" for i in range(n_symbols)]
    dates = pd.bdate_range(start, periods=n_days)

    # One market factor plus idiosyncratic noise. Beta is dispersed so the
    # cross-section is not a single scaled series.
    market = rng.normal(drift, MARKET_VOL, size=(n_days, 1))
    beta = rng.normal(1.0, 0.25, size=(1, n_symbols))
    rets = market * beta + rng.normal(0.0, DAILY_VOL, size=(n_days, n_symbols))

    if momentum_strength:
        # A plain causal walk: day t's tilt is read from cumulative log prices
        # that were finalised strictly before t. Written this way rather than
        # vectorised-with-corrections because a control whose own construction
        # needs careful reading cannot settle an argument about lookahead.
        logp = np.zeros_like(rets)
        acc = np.zeros(n_symbols)
        for t in range(n_days):
            if t >= LOOKBACK:
                past = logp[t - SKIP] - logp[t - LOOKBACK]
                z = (past - past.mean()) / (past.std() + 1e-12)
                rets[t] += momentum_strength * DAILY_VOL * z
            acc = acc + np.log1p(rets[t])
            logp[t] = acc

    close = pd.DataFrame(100.0 * np.cumprod(1.0 + rets, axis=0),
                         index=dates, columns=symbols)
    # Opens gap from the prior close by a fraction of a day's move -- enough that
    # decide-on-close/fill-on-open is not a free lunch in the control either.
    gaps = rng.normal(0.0, DAILY_VOL * 0.4, size=(n_days, n_symbols))
    open_ = close.shift(1) * (1.0 + gaps)
    open_.iloc[0] = close.iloc[0]
    return close, open_


def synthetic_turnover(close: pd.DataFrame, seed: int = 11) -> pd.DataFrame:
    """Turnover comfortably above the liquidity floor, so it never gates."""
    rng = np.random.default_rng(seed)
    noise = rng.lognormal(0.0, 0.3, size=close.shape)
    return pd.DataFrame(5e7 * noise, index=close.index, columns=close.columns)
