"""Strategy interface for intraday signals.

Three rules are enforced here rather than left to each strategy, because every
one of them is a mistake that is easy to make once per strategy and invisible
until it costs money.

1. EVERY SIGNAL CARRIES A STOP. Not optional, and validated on construction. The
   stop is what sizes the position -- `risk_per_trade / (entry - stop)` -- so a
   signal without one cannot be sized, and a bot that sized it some other way
   (fixed quantity, fixed notional) would take wildly different risk on a calm
   stock and a violent one. This is why `Signal.__post_init__` raises instead of
   defaulting.

2. ONLY CONFIRMED BARS. `SymbolContext.intraday` holds bars that have CLOSED. The
   forming bar is excluded by the engine before the context is built. Reading a
   partial bar's high or close is lookahead that backtests beautifully and cannot
   be reproduced live, because at decision time that bar had not finished.

3. REGIME PERMISSION IS THE STRATEGY'S OWN DECLARATION. Each strategy names the
   regime families it is valid in, and the router will not call it elsewhere. A
   breakout rule that runs in chop is not a bad signal, it is a rule being asked
   a question it was never designed to answer -- ULTIMATE's intraday venue lost
   money in every regime bucket precisely because nothing gated it.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np
import pandas as pd

LONG = "long"
SHORT = "short"
SIDES = (LONG, SHORT)


@dataclass(frozen=True)
class Signal:
    """One proposed intraday trade, fully specified before sizing."""

    symbol: str
    side: str
    entry: float
    stop: float
    target: float
    strategy: str
    confidence: float = 0.5
    at: pd.Timestamp | None = None
    reasons: tuple[str, ...] = ()

    def __post_init__(self):
        if self.side not in SIDES:
            raise ValueError(f"side must be one of {SIDES}, got {self.side!r}")
        for name in ("entry", "stop", "target"):
            v = getattr(self, name)
            if not np.isfinite(v) or v <= 0:
                raise ValueError(f"{self.strategy}/{self.symbol}: {name} must be finite and > 0, got {v}")
        if self.side == LONG and not (self.stop < self.entry < self.target):
            raise ValueError(
                f"{self.strategy}/{self.symbol}: long needs stop < entry < target, "
                f"got {self.stop} / {self.entry} / {self.target}"
            )
        if self.side == SHORT and not (self.target < self.entry < self.stop):
            raise ValueError(
                f"{self.strategy}/{self.symbol}: short needs target < entry < stop, "
                f"got {self.target} / {self.entry} / {self.stop}"
            )

    @property
    def risk_per_unit(self) -> float:
        return abs(self.entry - self.stop)

    @property
    def reward_per_unit(self) -> float:
        return abs(self.target - self.entry)

    @property
    def reward_risk(self) -> float:
        r = self.risk_per_unit
        return self.reward_per_unit / r if r > 0 else float("inf")

    @property
    def direction(self) -> int:
        return 1 if self.side == LONG else -1

    def describe(self) -> str:
        return (f"{self.symbol:<14} {self.side.upper():<5} @{self.entry:>9.2f} "
                f"stop {self.stop:>9.2f} target {self.target:>9.2f} "
                f"R:R {self.reward_risk:>4.2f} conf {self.confidence:.2f} "
                f"[{self.strategy}]")


@dataclass
class SymbolContext:
    """Everything a strategy may look at for one symbol at one instant."""

    symbol: str
    security_id: str
    now: pd.Timestamp
    intraday: pd.DataFrame          # CONFIRMED bars for today, oldest first
    daily: pd.DataFrame             # daily history, ending yesterday
    regime: object                  # RegimeState
    session: object                 # SessionConfig
    prev_close: float = float("nan")
    atr_daily: float = float("nan")  # yesterday's ATR14, for stop distances
    tick_size: float = 0.05

    @property
    def bars(self) -> int:
        return len(self.intraday)

    @property
    def last(self) -> pd.Series | None:
        return self.intraday.iloc[-1] if len(self.intraday) else None

    @property
    def price(self) -> float:
        last = self.last
        return float(last["close"]) if last is not None else float("nan")

    def minutes_since_open(self) -> float:
        if not len(self.intraday):
            return 0.0
        open_ts = pd.Timestamp(self.intraday["timestamp"].iloc[0])
        return (pd.Timestamp(self.now) - open_ts).total_seconds() / 60.0

    def session_vwap(self) -> float:
        """Volume-weighted average price for the session so far.

        Falls back to the mean typical price if the feed reports no volume --
        a zero-volume VWAP would be NaN and silently disable every VWAP rule.
        """
        d = self.intraday
        if not len(d):
            return float("nan")
        typical = (d["high"].astype(float) + d["low"].astype(float)
                   + d["close"].astype(float)) / 3.0
        vol = pd.to_numeric(d.get("volume", pd.Series(index=d.index, dtype=float)),
                            errors="coerce").fillna(0.0)
        total = float(vol.sum())
        if total <= 0:
            return float(typical.mean())
        return float((typical * vol).sum() / total)

    def opening_range(self, end_time) -> tuple[float, float] | None:
        """(high, low) of bars that closed at or before `end_time`."""
        d = self.intraday
        if not len(d):
            return None
        ts = pd.to_datetime(d["timestamp"])
        mask = ts.dt.time <= end_time
        seg = d[mask]
        if len(seg) < 1:
            return None
        return float(seg["high"].astype(float).max()), float(seg["low"].astype(float).min())

    def round_tick(self, price: float) -> float:
        t = self.tick_size if self.tick_size and self.tick_size > 0 else 0.05
        return round(round(price / t) * t, 2)


class IntradayStrategy(abc.ABC):
    """Base class for intraday entry rules."""

    #: regime families this rule is valid in; the router will not call it elsewhere
    families: tuple[str, ...] = ("trend", "range", "volatile")
    #: minimum confirmed bars needed before the rule may fire
    warmup_bars: int = 3

    @property
    def name(self) -> str:
        return type(self).__name__

    def allowed_in(self, family: str) -> bool:
        return family in self.families

    def ready(self, ctx: SymbolContext) -> bool:
        return ctx.bars >= self.warmup_bars

    @abc.abstractmethod
    def generate(self, ctx: SymbolContext) -> list[Signal]:
        """Return zero or more signals. Never raises on ordinary no-setup days."""

    def __repr__(self) -> str:
        return f"<{self.name} families={'/'.join(self.families)}>"


class StrategyRegistry:
    """Named collection of strategies, filtered by regime at call time."""

    def __init__(self, strategies: list[IntradayStrategy] | None = None):
        self._items: dict[str, IntradayStrategy] = {}
        for s in strategies or []:
            self.register(s)

    def register(self, strategy: IntradayStrategy) -> IntradayStrategy:
        if strategy.name in self._items:
            raise ValueError(f"strategy {strategy.name} already registered")
        self._items[strategy.name] = strategy
        return strategy

    def for_family(self, family: str) -> list[IntradayStrategy]:
        return [s for s in self._items.values() if s.allowed_in(family)]

    def all(self) -> list[IntradayStrategy]:
        return list(self._items.values())

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items
