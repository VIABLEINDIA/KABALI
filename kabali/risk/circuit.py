"""Circuit breakers -- the limits that stop the session regardless of signal quality.

These are checked BEFORE sizing and before every entry, and they are ordered from
most to least severe so the reported reason is the real one.

The distinction that matters: a HALT ends the session (no further entries, and
`flatten` says whether open positions must be closed too), while a BLOCK refuses
one entry and lets the session continue. Losing Rs 800 is a halt. Already holding
four positions is a block.

Three of these encode findings rather than preferences:

DAILY LOSS is measured on equity P&L, open positions included. Checking realised
only would let four losing positions sit at -Rs 190 apiece, Rs 760 down, and
still admit a fifth.

MAX TRADES exists because of cost arithmetic, not discipline. Twelve round trips
at roughly Rs 21 each is about Rs 250 -- 31% of the Rs 800 daily budget gone to
friction. An unbounded bot in a choppy session can trade that budget away without
ever being wrong about direction.

CONSECUTIVE LOSSES is a regime detector of last resort. Four stops in a row
usually means the classifier's family label is wrong for what the market is
actually doing, and the cheapest response to a model that has stopped describing
reality is to stop acting on it.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

HALT = "halt"
BLOCK = "block"
ALLOW = "allow"


@dataclass(frozen=True)
class CircuitDecision:
    """Whether trading may continue, and why not."""

    state: str                # allow | block | halt
    reason: str = ""
    flatten: bool = False     # halt only: close open positions immediately
    # Why the halt fired. `square_off` is the normal end of every session and is
    # NOT a risk event; reporting it as one would drown the halts that matter.
    kind: str = ""

    @property
    def allowed(self) -> bool:
        return self.state == ALLOW

    @property
    def halted(self) -> bool:
        return self.state == HALT

    def describe(self) -> str:
        if self.allowed:
            return "allow"
        tag = "HALT" if self.halted else "BLOCK"
        return f"{tag}: {self.reason}" + (" [flatten now]" if self.flatten else "")


class CircuitBreaker:
    """Session-level trading limits, evaluated against the book."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.risk = cfg.risk
        self.session = cfg.session
        self._halt: CircuitDecision | None = None

    @property
    def is_halted(self) -> bool:
        return self._halt is not None

    @property
    def halt_reason(self) -> str:
        return self._halt.reason if self._halt else ""

    def _trip(self, reason: str, flatten: bool, kind: str) -> CircuitDecision:
        self._halt = CircuitDecision(HALT, reason, flatten, kind)
        return self._halt

    def check_session(self, book, marks: dict[str, float],
                      now: pd.Timestamp) -> CircuitDecision:
        """Session-wide checks, independent of any particular signal.

        Once tripped a halt is sticky. A daily loss limit that un-tripped because
        an open position ticked back up would let the bot re-enter on the same
        capital it just lost, which is the opposite of what the limit is for.
        """
        if self._halt is not None:
            return self._halt

        capital = self.cfg.capital
        equity = book.equity_pnl(marks)

        loss_limit = self.risk.daily_loss_limit(capital)
        if equity <= -loss_limit:
            return self._trip(
                f"daily loss limit hit: Rs {equity:,.0f} against a Rs {loss_limit:,.0f} budget",
                flatten=True, kind="loss_limit")

        profit_lock = self.risk.daily_profit_lock(capital)
        if equity >= profit_lock:
            # Stop opening, but let winners run to their own targets.
            return self._trip(
                f"daily profit lock hit: Rs {equity:,.0f} at or above Rs {profit_lock:,.0f}",
                flatten=False, kind="profit_lock")

        if book.consecutive_losses >= self.risk.max_consecutive_losses:
            return self._trip(
                f"{book.consecutive_losses} consecutive losses -- regime read is likely wrong",
                flatten=False, kind="consecutive_losses")

        t = pd.Timestamp(now).time()
        if t >= self.session.square_off:
            return self._trip(
                f"square-off time {self.session.square_off} reached",
                flatten=True, kind="square_off")

        return CircuitDecision(ALLOW)

    def check_entry(self, book, marks: dict[str, float], now: pd.Timestamp,
                    symbol: str) -> CircuitDecision:
        """Per-entry checks, run after `check_session` allows the session."""
        session = self.check_session(book, marks, now)
        if not session.allowed:
            return session

        t = pd.Timestamp(now).time()
        if t < self.session.opening_range_end:
            return CircuitDecision(
                BLOCK, f"before {self.session.opening_range_end}; opening range still forming")
        if t >= self.session.entry_cutoff:
            return CircuitDecision(
                BLOCK, f"past entry cutoff {self.session.entry_cutoff}")

        if symbol in book.positions:
            return CircuitDecision(BLOCK, f"already holding {symbol}")

        if len(book.positions) >= self.risk.max_concurrent_positions:
            return CircuitDecision(
                BLOCK, f"already at {self.risk.max_concurrent_positions} concurrent positions")

        if book.trade_count >= self.risk.max_trades_per_day:
            return CircuitDecision(
                BLOCK, f"daily trade cap of {self.risk.max_trades_per_day} reached")

        # The budget test is on the WORST CASE, and the worst case counts only
        # REALISED P&L plus the full entry-to-stop risk of everything open.
        #
        # Unrealised profit is deliberately excluded. An earlier version used
        # `min(equity, 0)`, which let an open position's paper gain count as spare
        # budget -- and a paper replay found the exact failure: on 2026-06-24 three
        # open shorts were up ~Rs 230 at 11:35, that showed as room for a fourth,
        # and all four then reversed to their stops. The session settled at
        # -Rs 934 against an Rs 800 limit. Paper gains are not budget; they are
        # the thing that disappears on the way to the stop.
        #
        # Realised P&L does count, in both directions: money already booked is
        # genuinely spent or genuinely banked.
        #
        # The invariant this establishes: admit a position only while
        #     limit + realised - open_risk >= risk_per_trade
        # so after admitting it, limit + realised - open_risk >= 0 -- i.e. every
        # open position stopping out at once still lands inside the daily limit.
        capital = self.cfg.capital
        worst_case_room = (self.risk.daily_loss_limit(capital)
                           + book.realised_pnl - book.open_risk)
        if worst_case_room < self.risk.risk_per_trade(capital):
            return CircuitDecision(
                BLOCK,
                f"worst-case room Rs {worst_case_room:,.0f} is below the Rs "
                f"{self.risk.risk_per_trade(capital):,.0f} another position would risk "
                f"(realised Rs {book.realised_pnl:,.0f}, open risk Rs {book.open_risk:,.0f})")

        if book.gross_exposure >= self.risk.gross_exposure_cap(capital):
            return CircuitDecision(
                BLOCK, f"gross exposure Rs {book.gross_exposure:,.0f} at the cap")

        return CircuitDecision(ALLOW)

    def reset(self) -> None:
        """Clear the halt. Only legitimate at the start of a new session."""
        self._halt = None
