"""Position book and session accounting.

The book is the single source of truth for "what do we own and what has it cost
us today". Risk limits are enforced against it, so it has to be right about two
things that are easy to get wrong.

REALISED AND UNREALISED ARE BOTH REAL. A daily loss limit checked against
realised P&L only is not a loss limit: four open positions sitting at -Rs 190
each have lost Rs 760 of the Rs 800 budget, and a bot that has not booked them
yet would happily open a fifth. `equity_pnl` counts both.

COSTS ARE CHARGED AT THE POINT THEY OCCUR. Entry cost is deducted when the
position opens, not netted at exit. At Rs 40,000 with roughly Rs 21 of round-trip
cost on a Rs 20,000 position, twelve trades is about Rs 250 -- 31% of the daily
loss budget spent on friction alone. Accounting for it late makes the bot look
profitable right up until it settles.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class Position:
    """One open intraday position."""

    symbol: str
    security_id: str
    side: str                     # long | short
    quantity: int
    entry_price: float
    stop: float
    target: float
    strategy: str
    opened_at: pd.Timestamp
    entry_cost: float = 0.0
    confidence: float = 0.5
    # Cost-inclusive risk from the sizer. `risk_amount` prefers it because the
    # circuit breaker budgets against what a stop-out actually removes, and the
    # price-only figure understates that by the round trip.
    planned_risk: float = float("nan")
    # Highest (long) / lowest (short) price seen since entry, for trailing logic.
    excursion: float = float("nan")
    order_id: str | None = None

    @property
    def direction(self) -> int:
        return 1 if self.side == "long" else -1

    @property
    def notional(self) -> float:
        return self.quantity * self.entry_price

    @property
    def price_risk(self) -> float:
        """Rupees of price movement between entry and stop."""
        return abs(self.entry_price - self.stop) * self.quantity

    @property
    def risk_amount(self) -> float:
        """Rupees the account loses if the stop fills exactly, costs included."""
        import math
        if isinstance(self.planned_risk, float) and math.isfinite(self.planned_risk):
            return self.planned_risk
        return self.price_risk

    def unrealised(self, price: float) -> float:
        return (price - self.entry_price) * self.quantity * self.direction

    def stop_hit(self, low: float, high: float) -> bool:
        return low <= self.stop if self.side == "long" else high >= self.stop

    def target_hit(self, low: float, high: float) -> bool:
        return high >= self.target if self.side == "long" else low <= self.target

    def update_excursion(self, high: float, low: float) -> None:
        best = high if self.side == "long" else low
        if pd.isna(self.excursion):
            self.excursion = best
        elif self.side == "long":
            self.excursion = max(self.excursion, best)
        else:
            self.excursion = min(self.excursion, best)


@dataclass
class ClosedTrade:
    """A completed round trip, with every cost attributed."""

    symbol: str
    side: str
    quantity: int
    entry_price: float
    exit_price: float
    strategy: str
    opened_at: pd.Timestamp
    closed_at: pd.Timestamp
    entry_cost: float
    exit_cost: float
    reason: str                   # stop | target | squareoff | signal | circuit

    @property
    def direction(self) -> int:
        return 1 if self.side == "long" else -1

    @property
    def gross_pnl(self) -> float:
        return (self.exit_price - self.entry_price) * self.quantity * self.direction

    @property
    def costs(self) -> float:
        return self.entry_cost + self.exit_cost

    @property
    def net_pnl(self) -> float:
        return self.gross_pnl - self.costs

    @property
    def is_win(self) -> bool:
        return self.net_pnl > 0

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol, "side": self.side, "quantity": self.quantity,
            "entry_price": self.entry_price, "exit_price": self.exit_price,
            "strategy": self.strategy, "opened_at": self.opened_at,
            "closed_at": self.closed_at, "gross_pnl": round(self.gross_pnl, 2),
            "costs": round(self.costs, 2), "net_pnl": round(self.net_pnl, 2),
            "reason": self.reason,
        }


@dataclass
class SessionBook:
    """Open positions and realised results for one trading day."""

    trade_date: dt.date
    capital: float
    # Cost of closing, as a fraction of notional. Set by the engine from the
    # cost model so `equity_pnl` can be measured on liquidation value.
    exit_cost_rate: float = 0.0
    positions: dict[str, Position] = field(default_factory=dict)
    closed: list[ClosedTrade] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    consecutive_losses: int = 0

    # ------------------------------------------------------------ accounting
    @property
    def realised_pnl(self) -> float:
        return sum(t.net_pnl for t in self.closed)

    @property
    def open_entry_costs(self) -> float:
        return sum(p.entry_cost for p in self.positions.values())

    def unrealised_pnl(self, marks: dict[str, float]) -> float:
        total = 0.0
        for sym, pos in self.positions.items():
            px = marks.get(sym)
            if px is not None:
                total += pos.unrealised(px)
        return total

    @property
    def open_exit_costs(self) -> float:
        """What it would cost to close everything currently open."""
        if self.exit_cost_rate <= 0:
            return 0.0
        return sum(abs(p.notional) * self.exit_cost_rate for p in self.positions.values())

    def equity_pnl(self, marks: dict[str, float]) -> float:
        """Session P&L on LIQUIDATION value: what we would have if we flattened now.

        Three terms, and the third is the one that is easy to omit. Marks alone
        say what the book is worth; they do not say what it is worth *realised*,
        and an intraday book is always realised before the close. Closing is not
        free, so a limit measured on marks trips at exactly -Rs 800 and then pays
        another ~Rs 24 per position to get out -- landing past the limit it was
        supposed to enforce. A paper replay showed precisely that: a session
        halted on the limit and settled at -Rs 872.

        Counting the exit cost up front makes the breaker fire a little earlier
        so the settled result lands inside the budget, which is what the limit
        was always meant to promise.
        """
        return (self.realised_pnl + self.unrealised_pnl(marks)
                - self.open_entry_costs - self.open_exit_costs)

    @property
    def gross_exposure(self) -> float:
        return sum(abs(p.notional) for p in self.positions.values())

    @property
    def open_risk(self) -> float:
        """Total rupees at risk across open positions if every stop fills."""
        return sum(p.risk_amount for p in self.positions.values())

    @property
    def trade_count(self) -> int:
        return len(self.closed) + len(self.positions)

    @property
    def wins(self) -> int:
        return sum(1 for t in self.closed if t.is_win)

    # --------------------------------------------------------------- mutation
    def add(self, pos: Position) -> None:
        if pos.symbol in self.positions:
            raise ValueError(f"already holding {pos.symbol}; the engine must not stack positions")
        self.positions[pos.symbol] = pos

    def close(self, symbol: str, price: float, at: pd.Timestamp, exit_cost: float,
              reason: str) -> ClosedTrade:
        pos = self.positions.pop(symbol)
        trade = ClosedTrade(
            symbol=pos.symbol, side=pos.side, quantity=pos.quantity,
            entry_price=pos.entry_price, exit_price=price, strategy=pos.strategy,
            opened_at=pos.opened_at, closed_at=at,
            entry_cost=pos.entry_cost, exit_cost=exit_cost, reason=reason,
        )
        self.closed.append(trade)
        self.consecutive_losses = 0 if trade.is_win else self.consecutive_losses + 1
        return trade

    def reject(self, symbol: str, strategy: str, reason: str, at: pd.Timestamp) -> None:
        """Record a signal the risk engine refused, so the log explains the day."""
        self.rejected.append({"symbol": symbol, "strategy": strategy,
                              "reason": reason, "at": at})

    # ----------------------------------------------------------------- output
    def trades_frame(self) -> pd.DataFrame:
        if not self.closed:
            return pd.DataFrame(columns=["symbol", "side", "quantity", "entry_price",
                                         "exit_price", "strategy", "opened_at", "closed_at",
                                         "gross_pnl", "costs", "net_pnl", "reason"])
        return pd.DataFrame([t.as_dict() for t in self.closed])

    def summary(self, marks: dict[str, float] | None = None) -> str:
        marks = marks or {}
        n = len(self.closed)
        gross = sum(t.gross_pnl for t in self.closed)
        costs = sum(t.costs for t in self.closed)
        wr = (self.wins / n * 100) if n else 0.0
        return (
            f"{self.trade_date} | closed {n} (win {wr:.0f}%) | open {len(self.positions)} | "
            f"gross Rs {gross:,.0f} | costs Rs {costs:,.0f} | "
            f"realised Rs {self.realised_pnl:,.0f} | "
            f"equity Rs {self.equity_pnl(marks):,.0f} "
            f"({self.equity_pnl(marks) / self.capital * 100:+.2f}%)"
        )
