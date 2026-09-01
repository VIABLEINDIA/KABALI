"""The session engine -- one trading day, start to square-off.

This is the only place that decides to trade. It is written against a stream of
CONFIRMED bars and against the `Broker` interface, which is what makes a paper
session and a live session the same code path with a different broker object
plugged in. There is no separate backtest loop to drift out of sync with the
live loop, because there is only one loop.

Order of operations inside each timestep is deliberate, and each step is placed
where it is for a reason:

  1. MARK      update prices first. Every limit downstream is measured against
               current marks, so a limit checked on stale prices is not a limit.
  2. MANAGE    exits before entries, always. A stop that should have fired must
               fire before the capital it frees is considered for a new position,
               otherwise the bot can exceed its position count for one timestep --
               and that timestep is exactly when the market is moving.
  3. CIRCUIT   session-level halt check, after exits so a stop-out that breaches
               the daily limit halts the session in the same step it happens.
  4. ENTER     only what survives the circuit, the sizer, and the router.

STOP-VERSUS-TARGET IS RESOLVED PESSIMISTICALLY. When one bar's range contains
both levels, the stop is taken. Without tick data the true sequence is unknowable,
and ULTIMATE resolved it the same way; assuming the target would manufacture an
edge out of missing information, and it is the single easiest way to make a
backtest look good.

ENTRIES FILL ON THE NEXT BAR'S OPEN, not the signal bar's close. The close that
triggered the signal is knowable only once that bar has finished, at which point
it can no longer be traded. Filling at it is lookahead.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from kabali.execution.broker import BUY, SELL, Fill, OrderIntent
from kabali.risk.book import Position, SessionBook
from kabali.risk.circuit import CircuitBreaker
from kabali.risk.sizing import _round_trip_rate, size_position
from kabali.strategies.base import SymbolContext
from kabali.strategies.registry import route

log = logging.getLogger(__name__)

EXIT_STOP = "stop"
EXIT_TARGET = "target"
EXIT_SQUAREOFF = "squareoff"
EXIT_CIRCUIT = "circuit"


@dataclass
class SessionResult:
    """Everything one session produced."""

    trade_date: dt.date
    book: SessionBook
    regime_label: str
    universe_size: int
    timesteps: int = 0
    signals_generated: int = 0
    entries: int = 0
    rejected_by_risk: int = 0
    halted: bool = False
    halt_reason: str = ""
    halt_kind: str = ""
    modelled_slippage: float = 0.0
    observed_slippage: float = 0.0
    loss_limit: float = 0.0

    @property
    def net_pnl(self) -> float:
        return self.book.realised_pnl

    def record_row(self) -> dict:
        """One row of the paper record the live gate is evaluated against."""
        b = self.book
        return {
            "trade_date": self.trade_date,
            "regime": self.regime_label,
            "universe_size": self.universe_size,
            "trades": len(b.closed),
            "wins": b.wins,
            "gross_pnl": round(sum(t.gross_pnl for t in b.closed), 2),
            "costs": round(sum(t.costs for t in b.closed), 2),
            "net_pnl": round(b.realised_pnl, 2),
            "signals": self.signals_generated,
            "entries": self.entries,
            "rejected_by_risk": self.rejected_by_risk,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "halt_kind": self.halt_kind,
            "modelled_slippage": round(self.modelled_slippage, 4),
            "observed_slippage": round(self.observed_slippage, 4),
            "loss_limit": round(self.loss_limit, 2),
        }

    def render(self) -> str:
        b = self.book
        lines = [
            f"session {self.trade_date} | regime {self.regime_label} | "
            f"universe {self.universe_size}",
            f"  {self.timesteps} timesteps | {self.signals_generated} signals | "
            f"{self.entries} entries | {self.rejected_by_risk} blocked by risk",
            f"  {b.summary()}",
        ]
        if self.halted:
            lines.append(f"  HALTED: {self.halt_reason}")
        if b.closed:
            per_strategy = {}
            for t in b.closed:
                s = per_strategy.setdefault(t.strategy, [0, 0.0])
                s[0] += 1
                s[1] += t.net_pnl
            lines.append("  by strategy:")
            for name, (n, pnl) in sorted(per_strategy.items(), key=lambda kv: -kv[1][1]):
                lines.append(f"    {name:<38} {n:>3} trades  Rs {pnl:>10,.0f}")
        return "\n".join(lines)


class SessionEngine:
    """Runs one trading day over a set of per-symbol intraday bar frames."""

    def __init__(self, cfg, registry, broker, regime, instruments: dict,
                 daily_context: dict | None = None):
        self.cfg = cfg
        self.registry = registry
        self.broker = broker
        self.regime = regime
        self.instruments = instruments          # symbol -> Instrument
        self.daily = daily_context or {}        # symbol -> {prev_close, atr_daily, median_volume}
        self.circuit = CircuitBreaker(cfg)
        # One leg of the round trip, so the book can value itself at liquidation.
        self.exit_cost_rate = _round_trip_rate(cfg) / 2.0

    # ------------------------------------------------------------------- run
    def run(self, trade_date: dt.date, bars: dict[str, pd.DataFrame]) -> SessionResult:
        """Drive one session. `bars` maps symbol -> that day's intraday bars."""
        book = SessionBook(trade_date=trade_date, capital=self.cfg.capital,
                           exit_cost_rate=self.exit_cost_rate)
        self.circuit.reset()
        result = SessionResult(
            trade_date=trade_date, book=book,
            regime_label=getattr(self.regime, "label", "unknown"),
            universe_size=len(bars),
            loss_limit=self.cfg.risk.daily_loss_limit(self.cfg.capital),
        )

        bars = {s: d for s, d in bars.items() if d is not None and len(d)}
        if not bars:
            return result

        # Normalise timestamps once, then index by POSITION for the rest of the
        # session. The obvious `d[d.timestamp <= now]` inside the loop is
        # O(symbols x bars^2) -- at 250 names and 75 bars it dominates the run
        # and does nothing an integer slice does not.
        prepared: dict[str, pd.DataFrame] = {}
        cursor: dict[str, dict] = {}
        for sym, d in bars.items():
            d = d.copy()
            d["timestamp"] = pd.to_datetime(d["timestamp"])
            d = d.sort_values("timestamp").reset_index(drop=True)
            prepared[sym] = d
            cursor[sym] = {ts: i for i, ts in enumerate(d["timestamp"])}
        bars = prepared

        timeline = sorted({ts for d in bars.values() for ts in d["timestamp"]})
        # Signals decided at bar i are filled at bar i+1's open, so the last bar
        # can never produce an entry -- only an exit.
        pending: list[tuple] = []
        # Last position seen per symbol, so a symbol that did not print on this
        # timestamp still marks at its most recent confirmed bar rather than
        # vanishing from the book.
        last_idx: dict[str, int] = {}

        for idx, now in enumerate(timeline):
            result.timesteps += 1
            frames, current = {}, {}
            for sym, d in bars.items():
                i = cursor[sym].get(now)
                if i is None:
                    i = last_idx.get(sym)
                    if i is None:
                        continue
                else:
                    last_idx[sym] = i
                frames[sym] = d.iloc[: i + 1]
                current[sym] = d.iloc[i]
            marks = {s: float(r["close"]) for s, r in current.items()}

            # 1-2. exits first, on this bar's range
            self._manage_exits(book, current, now, result)

            # fill anything decided on the previous bar, at this bar's open
            if pending:
                self._fill_pending(pending, book, current, now, result)
                pending = []

            # 3. session-level circuit
            decision = self.circuit.check_session(book, marks, now)
            if decision.halted:
                # Square-off is how every normal session ends. Counting it as a
                # halt would report 100% halted sessions and bury the ones that
                # actually tripped a risk limit.
                if decision.kind != "square_off":
                    result.halted = True
                    result.halt_reason = decision.reason
                    result.halt_kind = decision.kind
                if decision.flatten:
                    reason = (EXIT_SQUAREOFF if decision.kind == "square_off"
                              else EXIT_CIRCUIT)
                    self._flatten(book, current, now, reason, result)
                    break
                continue

            # 4. entries -- decided now, filled next bar
            if idx < len(timeline) - 1:
                pending = self._scan(book, frames, current, marks, now, result)

        # Nothing may be carried overnight. This is an intraday bot; an open
        # position at the close is a bug, not a swing trade.
        if book.positions:
            last = timeline[-1]
            final = {s: d.iloc[-1] for s, d in bars.items() if len(d)}
            self._flatten(book, final, last, EXIT_SQUAREOFF, result)
            if book.positions:
                log.error("%d position(s) still open after square-off: %s",
                          len(book.positions), list(book.positions))

        return result

    # ---------------------------------------------------------------- stages
    def _manage_exits(self, book, current, now, result) -> None:
        for symbol in list(book.positions):
            pos = book.positions[symbol]
            # A reservation is a slot held between decision and fill, not a
            # position. It has no stop to hit and nothing to exit; treating it
            # as one crashes the session at the first timestep after a signal.
            if isinstance(pos, _Reservation):
                continue
            row = current.get(symbol)
            if row is None:
                continue
            high, low = float(row["high"]), float(row["low"])
            pos.update_excursion(high, low)

            hit_stop = pos.stop_hit(low, high)
            hit_target = pos.target_hit(low, high)

            # Pessimistic: when both are inside one bar, the stop wins.
            if hit_stop:
                self._exit(book, pos, pos.stop, now, EXIT_STOP, result)
            elif hit_target:
                self._exit(book, pos, pos.target, now, EXIT_TARGET, result)
            elif pd.Timestamp(now).time() >= self.cfg.session.square_off:
                self._exit(book, pos, float(row["close"]), now, EXIT_SQUAREOFF, result)

    def _scan(self, book, frames, current, marks, now, result) -> list[tuple]:
        pending: list[tuple] = []
        # One margin read per timestep. Per-signal would be an HTTP round trip
        # per candidate against the live broker -- 250 of them per five minutes.
        available_margin = self.broker.state().available_margin
        for symbol, frame in frames.items():
            if not len(frame):
                continue
            gate = self.circuit.check_entry(book, marks, now, symbol)
            if not gate.allowed:
                if gate.halted:
                    break
                continue

            ctx = self._context(symbol, frame, now)
            if ctx is None:
                continue
            signals = route(ctx, self.registry)
            if not signals:
                continue
            result.signals_generated += len(signals)

            for signal in signals:
                ctxinfo = self.daily.get(symbol, {})
                decision = size_position(
                    signal, self.cfg, book,
                    median_volume=ctxinfo.get("median_volume"),
                    available_margin=available_margin,
                )
                if not decision.approved:
                    result.rejected_by_risk += 1
                    book.reject(symbol, signal.strategy, decision.reason, now)
                    continue
                pending.append((signal, decision))
                # Reserve the slot immediately. Without this, a later symbol in
                # the same timestep would see the pre-entry book and the position
                # count could be exceeded before any of them fill.
                book.positions[symbol] = _Reservation(symbol, signal, decision, now)
        return pending

    def _fill_pending(self, pending, book, current, now, result) -> None:
        for signal, decision in pending:
            symbol = signal.symbol
            book.positions.pop(symbol, None)   # release the slot reservation
            row = current.get(symbol)
            if row is None:
                continue
            fill_ref = float(row["open"])       # next bar's open, as promised

            # The level that invalidated the idea may already be through by the
            # time we can act. Entering anyway would be entering a trade whose
            # stop is behind the price.
            if signal.side == "long" and fill_ref <= signal.stop:
                book.reject(symbol, signal.strategy, "opened through the stop", now)
                result.rejected_by_risk += 1
                continue
            if signal.side == "short" and fill_ref >= signal.stop:
                book.reject(symbol, signal.strategy, "opened through the stop", now)
                result.rejected_by_risk += 1
                continue

            # Re-size against the price we are actually filling at.
            #
            # The sizer ran on the signal bar's close; the fill happens at the
            # NEXT bar's open, and on a five-minute bar those differ. If the open
            # is against us the distance to the unchanged stop is wider than the
            # quantity was sized for, and the position silently carries more than
            # its share of the daily budget. Re-pricing keeps the rupee risk the
            # limit assumes, and skips the trade when the move has eaten the setup.
            try:
                repriced = replace(signal, entry=fill_ref)
            except (ValueError, TypeError):
                book.reject(symbol, signal.strategy, "next-bar open invalidated the setup", now)
                result.rejected_by_risk += 1
                continue

            decision = size_position(
                repriced, self.cfg, book,
                median_volume=self.daily.get(symbol, {}).get("median_volume"),
                available_margin=self.broker.state().available_margin,
            )
            if not decision.approved:
                book.reject(symbol, signal.strategy, f"at fill price: {decision.reason}", now)
                result.rejected_by_risk += 1
                continue
            signal = repriced

            side = BUY if signal.side == "long" else SELL
            intent = OrderIntent(
                symbol=symbol, security_id=self._security_id(symbol), side=side,
                quantity=decision.quantity, reference_price=fill_ref,
                reason=f"entry:{signal.strategy}", at=now,
                stop=signal.stop, target=signal.target,
            )
            fill = self.broker.submit(intent)
            if not fill.ok:
                book.reject(symbol, signal.strategy, f"broker: {fill.note}", now)
                result.rejected_by_risk += 1
                continue

            book.add(Position(
                symbol=symbol, security_id=intent.security_id, side=signal.side,
                quantity=fill.quantity, entry_price=fill.price, stop=signal.stop,
                target=signal.target, strategy=signal.strategy, opened_at=now,
                entry_cost=fill.cost, confidence=signal.confidence,
                planned_risk=decision.risk_amount, order_id=fill.order_id,
            ))
            result.entries += 1
            self._account_slippage(fill, fill_ref, result)

    def _exit(self, book, pos, price, now, reason, result) -> None:
        side = SELL if pos.side == "long" else BUY
        intent = OrderIntent(
            symbol=pos.symbol, security_id=pos.security_id, side=side,
            quantity=pos.quantity, reference_price=float(price),
            reason=f"exit:{reason}", at=now,
        )
        fill = self.broker.submit(intent)
        if not fill.ok:
            # An exit that will not fill is the dangerous case: the position is
            # still live and unmanaged. Log loudly; the square-off pass retries.
            log.error("EXIT REJECTED %s (%s): %s -- position still open",
                      pos.symbol, reason, fill.note)
            return
        book.close(pos.symbol, fill.price, now, fill.cost, reason)
        self._account_slippage(fill, float(price), result)

    def _flatten(self, book, current, now, reason, result) -> None:
        for symbol in list(book.positions):
            pos = book.positions[symbol]
            if isinstance(pos, _Reservation):
                book.positions.pop(symbol, None)
                continue
            row = current.get(symbol)
            price = float(row["close"]) if row is not None else pos.entry_price
            self._exit(book, pos, price, now, reason, result)

    # ---------------------------------------------------------------- detail
    def _context(self, symbol, frame, now) -> SymbolContext | None:
        info = self.daily.get(symbol, {})
        inst = self.instruments.get(symbol)
        return SymbolContext(
            symbol=symbol,
            security_id=self._security_id(symbol),
            now=pd.Timestamp(now),
            intraday=frame,
            daily=info.get("daily", pd.DataFrame()),
            regime=self.regime,
            session=self.cfg.session,
            prev_close=float(info.get("prev_close", np.nan)),
            atr_daily=float(info.get("atr_daily", np.nan)),
            tick_size=float(getattr(inst, "tick_size", 0.05) or 0.05),
        )

    def _security_id(self, symbol: str) -> str:
        inst = self.instruments.get(symbol)
        return str(getattr(inst, "security_id", symbol))

    def _account_slippage(self, fill: Fill, reference: float, result: SessionResult) -> None:
        """Accumulate realised vs predicted slippage, in rupees.

        In a paper session these are equal by construction -- PaperBroker applies
        exactly the modelled slippage -- so the gate's slippage-fidelity check is
        vacuous until it sees LIVE fills. That is the intended behaviour, not an
        oversight: the check exists to catch the model being wrong about the real
        world, and a simulator cannot falsify itself.
        """
        result.observed_slippage += abs(fill.slippage) * fill.quantity
        result.modelled_slippage += (
            abs(reference) * fill.quantity * self.cfg.costs.slippage_bps / 10_000.0)


class _Reservation:
    """Placeholder occupying a position slot between decision and fill.

    It exists so that limits measured across a timestep are correct. Several
    symbols are scanned at one timestamp, and each `check_entry` reads the book
    as it stands -- so a reservation must report the risk and exposure of the
    position it is about to become.

    Reporting zero (the obvious first cut) breaks the daily loss limit outright:
    four entries decided on the same bar each see an unchanged `open_risk`, all
    four pass the remaining-budget test against the same free budget, and the
    session opens Rs 800 of risk while believing it opened Rs 200. A paper replay
    caught exactly that -- three sessions closed beyond the daily limit with the
    breaker never firing.

    It still carries no P&L: nothing has filled yet, so there is nothing to mark.
    """

    __slots__ = ("symbol", "signal", "decision", "at")

    def __init__(self, symbol, signal, decision, at):
        self.symbol = symbol
        self.signal = signal
        self.decision = decision
        self.at = at

    @property
    def notional(self) -> float:
        return float(self.decision.notional)

    @property
    def risk_amount(self) -> float:
        return float(self.decision.risk_amount)

    @property
    def price_risk(self) -> float:
        return float(self.decision.price_risk)

    @property
    def entry_cost(self) -> float:
        return 0.0          # not filled yet, so nothing has been paid

    def unrealised(self, price: float) -> float:
        return 0.0
