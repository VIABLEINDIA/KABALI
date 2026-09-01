"""Broker interface shared by the paper and live implementations.

The engine is written against this interface and never against a concrete
broker, which is what makes "the same code that was paper-tested is the code
that goes live" a structural fact rather than a promise. The only difference
between a paper session and a live one is which object is constructed.

`OrderIntent` exists so the engine hands the broker a fully specified,
already-risk-approved instruction. A broker implementation is not allowed to
size, re-price, or decide anything -- if a broker could adjust quantity, the
risk engine's guarantees would stop being guarantees.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field

import pandas as pd

BUY = "buy"
SELL = "sell"


@dataclass(frozen=True)
class OrderIntent:
    """A risk-approved instruction to trade. Immutable by design."""

    symbol: str
    security_id: str
    side: str                    # buy | sell
    quantity: int
    reference_price: float       # price the fill is quoted against
    reason: str                  # entry:<strategy> | exit:<stop|target|squareoff|circuit>
    at: pd.Timestamp
    stop: float | None = None
    target: float | None = None
    #: Price at the moment the decision was actually made, when that differs from
    #: the reference the fill is quoted against. An entry decided on a bar's close
    #: cannot fill until the next bar opens, and the move between the two is a real
    #: cost that no simulator invents -- it is in the bars. Carrying it here is what
    #: lets the gate compare assumed slippage against a measured one. None means
    #: decision and reference coincide, so there is nothing extra to measure.
    decision_price: float | None = None

    def __post_init__(self):
        if self.side not in (BUY, SELL):
            raise ValueError(f"side must be {BUY} or {SELL}, got {self.side!r}")
        if self.quantity < 1:
            raise ValueError(f"quantity must be >= 1, got {self.quantity}")


@dataclass
class Fill:
    """What actually happened, including what it cost."""

    symbol: str
    side: str
    quantity: int
    price: float
    cost: float
    at: pd.Timestamp
    order_id: str = ""
    status: str = "filled"        # filled | rejected | partial
    note: str = ""
    slippage: float = 0.0         # fill price minus reference, signed against us

    @property
    def ok(self) -> bool:
        return self.status == "filled" and self.quantity > 0

    @property
    def notional(self) -> float:
        return self.price * self.quantity

    def describe(self) -> str:
        if not self.ok:
            return f"{self.symbol} {self.side.upper()} {self.status.upper()}: {self.note}"
        return (f"{self.symbol} {self.side.upper()} {self.quantity} @ {self.price:.2f} "
                f"cost Rs {self.cost:.2f} slip {self.slippage:+.2f}")


@dataclass
class BrokerState:
    """Whatever the broker knows about the account right now."""

    available_margin: float = float("nan")
    utilised_margin: float = float("nan")
    source: str = "unknown"
    fetched_at: pd.Timestamp | None = None
    extra: dict = field(default_factory=dict)


class Broker(abc.ABC):
    """Minimal surface the engine needs from a venue."""

    #: human-readable, appears in every log line and run artifact
    mode: str = "abstract"

    @abc.abstractmethod
    def submit(self, intent: OrderIntent) -> Fill:
        """Execute an intent and return what happened. Must never raise on a
        rejected order -- a rejection is a Fill with status='rejected'."""

    @abc.abstractmethod
    def state(self) -> BrokerState:
        """Current account state, best effort."""

    def describe(self) -> str:
        return f"{type(self).__name__}(mode={self.mode})"
