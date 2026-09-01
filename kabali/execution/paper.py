"""Paper broker -- simulated fills charged at the real statutory cost model.

A paper broker is only useful if it is pessimistic in the same places a real one
is expensive. This one uses `research.costs`, the identical fingerprinted model
ULTIMATE's backtests ran on, so a paper session and a backtest are comparable
numbers rather than two different accounting conventions.

Slippage is charged AGAINST the order on both legs -- buys fill above the
reference, sells below. Modelling it as symmetric noise would let it average out
over many trades, which is precisely the error that makes a marginal strategy
look viable: real slippage is a cost with a sign, not a random variable with mean
zero, because we are always the one crossing the spread.

What this deliberately does NOT simulate, and what forward paper trading against
live fills is therefore still required to establish:

  * market impact -- our size is small, but the model assumes zero
  * partial fills -- every order fills completely or not at all
  * queue position and rejection -- a live MARKET order can be rejected for
    margin, circuit limits, or a scrip-level ban, and none of that appears here
  * circuit limits -- a stock frozen at its upper band cannot be sold into

Those omissions are the reason the live gate requires observed slippage to be
compared against modelled slippage before real money is permitted.
"""
from __future__ import annotations

import itertools

import pandas as pd

from kabali.core import get_cost_model
from kabali.execution.broker import BUY, Broker, BrokerState, Fill, OrderIntent


class PaperBroker(Broker):
    """Fills every order at reference price plus adverse slippage, minus costs."""

    mode = "paper"

    def __init__(self, cfg, starting_margin: float | None = None):
        self.cfg = cfg
        self.costs = get_cost_model(cfg.costs.profile)
        self.slippage_bps = cfg.costs.slippage_bps
        # MIS buying power. Assumed leverage, not the broker's advertised
        # maximum -- see config.account.assumed_mis_leverage.
        self._margin = (starting_margin if starting_margin is not None
                        else cfg.capital * cfg.account.assumed_mis_leverage)
        self._start_margin = self._margin
        self._ids = itertools.count(1)
        self.fills: list[Fill] = []

    # ------------------------------------------------------------------ core
    def submit(self, intent: OrderIntent) -> Fill:
        ref = float(intent.reference_price)
        if ref <= 0:
            return self._reject(intent, "reference price is not positive")

        slip = ref * self.slippage_bps / 10_000.0
        price = ref + slip if intent.side == BUY else ref - slip
        price = round(price, 2)
        if price <= 0:
            return self._reject(intent, "slippage drove the fill price to zero")

        notional = price * intent.quantity
        if intent.side == BUY and notional > self._margin:
            return self._reject(
                intent,
                f"insufficient simulated margin: need Rs {notional:,.0f}, "
                f"have Rs {self._margin:,.0f}")

        cost = self.costs.charge(price, intent.quantity, "buy" if intent.side == BUY else "sell")
        self._margin += -notional if intent.side == BUY else notional
        self._margin -= cost

        fill = Fill(
            symbol=intent.symbol, side=intent.side, quantity=intent.quantity,
            price=price, cost=round(cost, 2), at=intent.at,
            order_id=f"PAPER-{next(self._ids):06d}",
            status="filled",
            slippage=round(price - ref if intent.side == BUY else ref - price, 4),
            note=intent.reason,
        )
        self.fills.append(fill)
        return fill

    def state(self) -> BrokerState:
        return BrokerState(
            available_margin=round(self._margin, 2),
            utilised_margin=round(self._start_margin - self._margin, 2),
            source="simulated",
            fetched_at=pd.Timestamp.now(),
            extra={"cost_profile": self.cfg.costs.profile,
                   "cost_fingerprint": self.costs.fingerprint(),
                   "slippage_bps": self.slippage_bps},
        )

    # ----------------------------------------------------------------- detail
    def _reject(self, intent: OrderIntent, note: str) -> Fill:
        fill = Fill(symbol=intent.symbol, side=intent.side, quantity=0,
                    price=0.0, cost=0.0, at=intent.at, status="rejected", note=note)
        self.fills.append(fill)
        return fill

    def fills_frame(self) -> pd.DataFrame:
        if not self.fills:
            return pd.DataFrame(columns=["symbol", "side", "quantity", "price",
                                         "cost", "slippage", "at", "status", "note"])
        return pd.DataFrame([{
            "symbol": f.symbol, "side": f.side, "quantity": f.quantity,
            "price": f.price, "cost": f.cost, "slippage": f.slippage,
            "at": f.at, "status": f.status, "note": f.note, "order_id": f.order_id,
        } for f in self.fills])
