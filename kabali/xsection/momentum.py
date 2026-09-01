"""The cross-sectional momentum signal, and the rules for who may be ranked.

PARAMETERS ARE PRE-COMMITTED. Every number here is the textbook value for the
Jegadeesh-Titman construction, fixed before the first run and not swept. That is
a deliberate response to what ULTIMATE found: 6,036 configurations produced zero
robust parameter regions, and the mechanism was that sweeping lets the best-
looking setting be noise. One honest test of the canonical form is worth more
than a surface of tuned ones, and if 12-1 momentum has nothing here, a 9-2
variant that happens to look better on the same 606 names is not evidence that
it does.

WHY 12-1 AND NOT 12-0. The most recent month is skipped because short-horizon
returns reverse: last month's biggest winners are disproportionately likely to
give it back. Including that month mixes a reversal effect into a momentum
signal and muddies whatever is being measured. The skip is not a tuned parameter,
it is part of the definition.
"""
from __future__ import annotations

import pandas as pd

#: Formation window: twelve months of trading sessions.
LOOKBACK = 252
#: Skipped most-recent month, to keep short-term reversal out of the signal.
SKIP = 21
#: Trailing window for the point-in-time liquidity screen.
TURNOVER_WINDOW = 60
#: Rupees of median daily turnover required to be rankable. Far above anything
#: a 40k book needs; it is here so the result stays honest at larger size.
MIN_TURNOVER = 1e7
#: Below this a price is mostly tick noise, and a 5 paise tick on a 8 rupee
#: share is 60bps of spread the cost model does not know about.
MIN_PRICE = 10.0


def momentum(close: pd.DataFrame, lookback: int = LOOKBACK,
             skip: int = SKIP) -> pd.DataFrame:
    """Total return from t-lookback to t-skip, per name, aligned to date t.

    Vectorised across the whole panel: shifting is the same operation for every
    column, so the entire five-year signal surface costs two shifts and a divide.
    """
    return close.shift(skip) / close.shift(lookback) - 1.0


def eligible(close: pd.DataFrame, turnover: pd.DataFrame,
             lookback: int = LOOKBACK, skip: int = SKIP,
             window: int = TURNOVER_WINDOW,
             min_turnover: float = MIN_TURNOVER,
             min_price: float = MIN_PRICE) -> pd.DataFrame:
    """Boolean mask: may this name be ranked on this date?

    Every condition reads only backwards. The liquidity screen uses a trailing
    median rather than a mean because one block deal should not qualify a name
    that is otherwise untradeable, and rather than a point value because a single
    quiet session should not disqualify a liquid one.
    """
    liquid = turnover.rolling(window, min_periods=window // 2).median() >= min_turnover
    priced = close >= min_price
    # The signal needs a valid close at BOTH ends of the formation window; a name
    # that was not trading a year ago has no momentum to measure.
    formed = close.shift(skip).notna() & close.shift(lookback).notna()
    tradeable = close.notna()
    return liquid & priced & formed & tradeable


def rank_top(scores: pd.Series, mask: pd.Series, n: int) -> list[str]:
    """The n highest-scoring eligible names on one date, best first.

    Ties are broken by the symbol name so a rerun picks the same portfolio. That
    matters more than it sounds: an unstable tiebreak makes the whole backtest
    irreproducible and quietly turns re-runs into a sweep.
    """
    s = scores[mask & scores.notna()]
    if s.empty:
        return []
    ordered = s.sort_values(ascending=False, kind="mergesort")
    ordered = ordered.to_frame("s").assign(sym=ordered.index).sort_values(
        ["s", "sym"], ascending=[False, True])
    return list(ordered.index[:n])
