"""Long-only cross-sectional rebalance engine with exact Indian delivery costs.

This is the piece where a backtest lies if it is written carelessly, so the
modelling choices are stated rather than buried:

*Decide on the close, fill on the next open.* The ranking is computed from data
through the close of session t and executed at the open of t+1. Nothing in the
signal may touch t+1. This is the single most common way a portfolio backtest
manufactures an edge, and it is also the one KABALI's intraday replay already got
caught by -- sized at the signal price, filled at the next open, and a gap made
real risk exceed sized risk.

*Sized on the close, filled at the open, and the difference is real.* Quantity is
computed from the close of t because that is the last price known when the order
is placed. The fill happens at the open of t+1, which is a different number. When
the gap runs against the book the position costs more than budgeted and the
engine reduces quantity or skips the name outright rather than quietly borrowing
cash it does not have.

*Sell before buy.* Buys are funded from realised cash. Running them in the other
order would let the book spend proceeds it has not received, which is leverage
by accident -- and in a CNC delivery account, leverage that does not exist.

*Integer shares.* At forty thousand rupees across ten names, a four-thousand
rupee slice of a three-thousand rupee share is one share, not 1.33. Rounding is
not a detail at this size; it is a material part of the tracking error, and a
backtest using fractional shares would report a portfolio nobody can hold.

*Marks are not trades.* Daily equity marks forward-fill a missing close so the
curve stays continuous, but a forward-filled price is never used to size or fill
anything. That distinction is what the intraday engine learned the hard way when
unrealised profit was counted as risk budget.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from kabali.core import ensure_core_importable

ensure_core_importable()
from research.costs import get_cost_model  # noqa: E402


@dataclass
class XSecConfig:
    capital: float = 40_000.0
    n_positions: int = 10
    #: Trading sessions between rebalances. Monthly is the canonical momentum
    #: cadence and keeps friction near 1.5% of capital a year at this size.
    rebalance_days: int = 21
    cost_profile: str = "equity_delivery"
    #: Fraction of equity held back so integer rounding and an adverse open do
    #: not overdraw the account.
    cash_buffer: float = 0.02


@dataclass
class XSecResult:
    equity: pd.Series
    trades: pd.DataFrame
    holdings: pd.DataFrame
    config: XSecConfig
    #: Daily uninvested cash. Kept rather than discarded so a test can assert the
    #: account never borrows -- an invariant that is cheap to state and, in a CNC
    #: delivery account, impossible to violate in real life.
    cash: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    #: Daily open position count, for the same reason.
    positions: pd.Series = field(default_factory=lambda: pd.Series(dtype=int))
    stats: dict = field(default_factory=dict)


def _mark_prices(close: pd.DataFrame) -> pd.DataFrame:
    """Continuous prices for MARKING ONLY. Never used to size or fill."""
    return close.ffill()


def run(close: pd.DataFrame, open_: pd.DataFrame, scores: pd.DataFrame,
        mask: pd.DataFrame, config: XSecConfig | None = None,
        start: str | None = None) -> XSecResult:
    """Walk the panel forward, rebalancing into the top-ranked eligible names."""
    from kabali.xsection.momentum import rank_top

    cfg = config or XSecConfig()
    costs = get_cost_model(cfg.cost_profile)
    marks = _mark_prices(close)

    dates = close.index
    first = 0 if start is None else int(dates.searchsorted(pd.Timestamp(start)))
    # Warm-up: the first date on which any name has a score at all.
    valid = scores.notna().any(axis=1)
    first = max(first, int(np.argmax(valid.to_numpy())) if valid.any() else len(dates))
    if first >= len(dates) - 1:
        raise ValueError("no dates left after warm-up; panel is too short")

    cash = cfg.capital
    qty: dict[str, int] = {}
    trades: list[dict] = []
    equity_rows: list[tuple] = []
    holding_rows: list[dict] = []

    rebalance_on = set(range(first, len(dates) - 1, cfg.rebalance_days))

    for i in range(first, len(dates)):
        date = dates[i]

        # --- execute yesterday's decision at today's open -------------------
        if i - 1 in rebalance_on:
            decide, exec_i = dates[i - 1], i
            target = rank_top(scores.loc[decide], mask.loc[decide], cfg.n_positions)
            cash, qty, filled = _rebalance(
                cash, qty, target, close.iloc[i - 1], open_.iloc[exec_i],
                marks.iloc[i - 1], costs, cfg, date, trades)
            holding_rows.append({"date": date, "target": len(target),
                                 "filled": filled, "held": len(qty)})

        # --- mark ------------------------------------------------------------
        px = marks.iloc[i]
        # Summed per position, not in aggregate: one unmarkable name must not
        # blank the value of every other holding on that day.
        value = 0.0
        for s_, q in qty.items():
            m = float(px.get(s_, np.nan))
            if np.isfinite(m):
                value += q * m
        equity_rows.append((date, cash + value, cash, len(qty)))

    equity = pd.DataFrame(equity_rows,
                          columns=["date", "equity", "cash", "positions"]
                          ).set_index("date")
    return XSecResult(equity=equity["equity"], trades=pd.DataFrame(trades),
                      holdings=pd.DataFrame(holding_rows), config=cfg,
                      cash=equity["cash"], positions=equity["positions"])


def _rebalance(cash, qty, target, decide_close, exec_open, decide_marks,
               costs, cfg, date, trades):
    """One rebalance: exit what left the target, then buy into what entered."""
    target_set = set(target)

    # --- sells first: buys are funded from realised cash ---------------------
    for sym in [s for s in list(qty) if s not in target_set]:
        price = exec_open.get(sym, np.nan)
        if not np.isfinite(price) or price <= 0:
            continue          # untradeable today; the position is carried, not vanished
        n = qty.pop(sym)
        fill = costs.slip_price(float(price), "sell")
        charge = costs.charge(fill, n, "sell")
        cash += fill * n - charge
        trades.append({"date": date, "symbol": sym, "side": "sell", "qty": n,
                       "price": round(fill, 2), "cost": round(charge, 2)})

    # --- buys ----------------------------------------------------------------
    entering = [s for s in target if s not in qty]
    if not entering:
        return cash, qty, 0

    held_value = 0.0
    for sym, q in qty.items():
        m = float(decide_marks.get(sym, np.nan))
        if np.isfinite(m):
            held_value += q * m
    budget = (cash + held_value) * (1.0 - cfg.cash_buffer) / cfg.n_positions

    filled = 0
    for sym in entering:
        # Sized on the last known price, filled at a price we do not yet know.
        size_px = decide_close.get(sym, np.nan)
        fill_px = exec_open.get(sym, np.nan)
        if not np.isfinite(size_px) or not np.isfinite(fill_px) or fill_px <= 0:
            continue
        n = int(budget // float(size_px))
        if n <= 0:
            continue          # a single share costs more than the slice allows
        fill = costs.slip_price(float(fill_px), "buy")
        # The open moved against the sizing price: shrink to what cash covers.
        while n > 0 and fill * n + costs.charge(fill, n, "buy") > cash:
            n -= 1
        if n <= 0:
            continue
        charge = costs.charge(fill, n, "buy")
        cash -= fill * n + charge
        qty[sym] = n
        filled += 1
        trades.append({"date": date, "symbol": sym, "side": "buy", "qty": n,
                       "price": round(fill, 2), "cost": round(charge, 2)})

    return cash, qty, filled
