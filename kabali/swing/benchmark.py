"""What the strategy has to beat.

A long-only momentum system run through a rising market will show a profit that
is not an edge. Net P&L cannot answer the question on its own, so every swing
result is reported against two passive alternatives that were available to the
same account over the same window, charged through the same cost model:

NIFTY BUY AND HOLD -- the market return. Buying the index is not literally
possible in the cash segment; the nearest real instrument is an index ETF, which
adds a small expense ratio and tracking error this does not model. It is
therefore a slightly generous benchmark, which is the safe direction for a
benchmark to be wrong in.

EQUAL-WEIGHT UNIVERSE -- the average investable stock. This is the sharper of
the two, because it holds the same survivorship bias as the strategy: both are
computed over names that exist in the cache today, so the comparison between
them is far less contaminated than either number alone. It is a RETURN series,
not an implementable portfolio -- ₹40,000 cannot hold 500 names -- and it is
labelled that way wherever it is reported.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from kabali.core import get_cost_model


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    equity: pd.Series
    initial_capital: float
    note: str = ""

    @property
    def net_pnl(self) -> float:
        return float(self.equity.iloc[-1] - self.initial_capital)

    @property
    def return_pct(self) -> float:
        return float(self.net_pnl / self.initial_capital * 100.0)

    def cagr_pct(self) -> float | None:
        if len(self.equity) < 2:
            return None
        years = (self.equity.index[-1] - self.equity.index[0]).days / 365.25
        mult = float(self.equity.iloc[-1] / self.initial_capital)
        if years <= 0 or mult <= 0:
            return None
        return float((mult ** (1.0 / years) - 1.0) * 100.0)

    def max_drawdown_pct(self) -> float:
        eq = self.equity.dropna()
        if len(eq) < 2:
            return 0.0
        return float(((eq / eq.cummax()) - 1.0).min() * 100.0)


def _edges(costs) -> tuple[float, float]:
    """(entry drag, exit drag) as fractions of notional, for one round trip."""
    model = get_cost_model(costs.profile)
    slip = costs.slippage_bps / 10_000.0
    # `charge` is linear in turnover for everything except the brokerage cap,
    # which delivery does not have, so a rate is exact here rather than an
    # approximation of one.
    per_side = model.charge(100.0, 1.0, "buy") / 100.0, model.charge(100.0, 1.0, "sell") / 100.0
    return per_side[0] + slip, per_side[1] + slip


def buy_and_hold(prices: pd.Series, capital: float, costs,
                 name: str = "buy_and_hold", note: str = "") -> BenchmarkResult:
    """Buy at the first close, hold, sell at the last. Costs on both legs."""
    px = pd.Series(prices).astype(float).dropna()
    if len(px) < 2:
        raise ValueError(f"{name}: need at least 2 prices, got {len(px)}")
    entry_drag, exit_drag = _edges(costs)
    invested = capital * (1.0 - entry_drag)
    curve = invested * (px / float(px.iloc[0]))
    curve.iloc[-1] = float(curve.iloc[-1]) * (1.0 - exit_drag)
    return BenchmarkResult(name=name, equity=curve, initial_capital=float(capital), note=note)


def equal_weight(panel, symbols, capital: float, costs,
                 name: str = "equal_weight_universe") -> BenchmarkResult:
    """Equal-weighted total return of `symbols`, bought day one and held.

    Names without a price on the first date are dropped rather than joined
    later: a benchmark that quietly adds constituents as they list is picking
    entry dates the strategy never had.
    """
    close = panel.close[[s for s in symbols if s in panel.close.columns]]
    if close.empty:
        raise ValueError("equal_weight: no symbols with cached prices")
    first = close.iloc[0]
    usable = [c for c in close.columns if np.isfinite(first[c]) and first[c] > 0]
    if not usable:
        raise ValueError("equal_weight: no symbol has a price on the first date")

    norm = close[usable] / first[usable]
    # Forward-fill so a single missing print does not drop a name out of the
    # average for a day and manufacture a jump in the benchmark.
    basket = norm.ffill().mean(axis=1)

    entry_drag, exit_drag = _edges(costs)
    curve = capital * (1.0 - entry_drag) * basket
    curve.iloc[-1] = float(curve.iloc[-1]) * (1.0 - exit_drag)
    return BenchmarkResult(
        name=name, equity=curve, initial_capital=float(capital),
        note=(f"{len(usable)} names, equal weighted, bought at the first close and held. "
              f"A return series, not an implementable portfolio at this capital."),
    )
