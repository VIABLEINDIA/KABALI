"""The swing portfolio: cash, positions, and the daily loop that moves them.

Everything here is an accounting decision that the intraday replay proved is
easy to get wrong in a way that flatters the result. The five bugs that replay
caught were all of this kind, and the same five have been written into the
design rather than rediscovered:

DECIDE ON THE CLOSE, FILL AT THE NEXT OPEN. A rule that reads Wednesday's close
cannot also transact at it. Orders decided on day d are queued and filled at
d+1's open, with slippage. This is the single largest source of fake edge in a
daily-bar study.

RE-SIZE AT THE FILL, NOT AT THE DECISION. The overnight gap between the two is
exactly where "sized at signal price, filled at next open" went wrong intraday.
The stop is fixed the night before, from that night's ATR; the quantity is
recomputed against the price actually paid, and the order is cancelled outright
if the market opened through the stop.

CASH IS FINITE AND COSTS LEAVE IT IMMEDIATELY. CNC delivery is settled in cash:
there is no MIS leverage to borrow, so the book cannot exceed the account. Entry
cost is deducted on entry, not netted at exit.

A GAP THROUGH THE STOP FILLS AT THE OPEN. If the session opens below a long's
stop, the exit is the open, worse than the stop. Filling such a day at the stop
price would credit the strategy with an exit that was not available.

UNREALISED PROFIT IS NOT BUDGET. Room for a new position is judged on cash and
the worst case of what is already open, never on marks. This is the subtlest of
the intraday bugs and the one that breached the daily limit.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from kabali.core import compute_metrics, get_cost_model
from kabali.risk.book import ClosedTrade
from kabali.risk.sizing import round_trip_rate, size_position
from kabali.strategies.base import Signal

log = logging.getLogger(__name__)


@dataclass
class SwingPosition:
    """One held position. No target: the exit is a rule, not a price."""

    symbol: str
    security_id: str
    quantity: int
    entry_price: float
    stop: float
    opened_at: pd.Timestamp
    strategy: str
    entry_cost: float = 0.0
    planned_risk: float = float("nan")
    initial_stop: float = float("nan")

    @property
    def notional(self) -> float:
        return self.quantity * self.entry_price

    @property
    def risk_amount(self) -> float:
        """What a stop-out costs the account, costs included."""
        if np.isfinite(self.planned_risk):
            return float(self.planned_risk)
        return float(abs(self.entry_price - self.stop) * self.quantity)

    def unrealised(self, price: float) -> float:
        return float((price - self.entry_price) * self.quantity)


@dataclass
class PendingOrder:
    """An order decided on one close, to be filled at the next open."""

    symbol: str
    action: str                    # buy | sell
    decided_on: pd.Timestamp
    stop: float = float("nan")     # buys only; fixed the night before
    reason: str = ""


@dataclass
class SwingBook:
    """Cash, holdings and the realised record. The single source of truth."""

    capital: float
    cash: float = field(init=False)
    positions: dict[str, SwingPosition] = field(default_factory=dict)
    closed: list[ClosedTrade] = field(default_factory=list)
    costs_paid: float = 0.0

    def __post_init__(self):
        self.cash = float(self.capital)

    @property
    def realised_pnl(self) -> float:
        return float(sum(t.net_pnl for t in self.closed))

    @property
    def gross_exposure(self) -> float:
        """Book value at entry -- what the sizing caps are measured against."""
        return float(sum(p.notional for p in self.positions.values()))

    @property
    def open_risk(self) -> float:
        """Worst case across everything held, if every stop filled exactly."""
        return float(sum(p.risk_amount for p in self.positions.values()))

    def holdings_value(self, marks: dict[str, float]) -> float:
        total = 0.0
        for sym, pos in self.positions.items():
            px = marks.get(sym)
            if px is None or not np.isfinite(px):
                px = pos.entry_price       # no bar today: carry at cost, never at a guess
            total += pos.quantity * px
        return float(total)

    def equity(self, marks: dict[str, float]) -> float:
        return float(self.cash + self.holdings_value(marks))

    def buy(self, pos: SwingPosition) -> None:
        self.positions[pos.symbol] = pos
        self.cash -= pos.notional + pos.entry_cost
        self.costs_paid += pos.entry_cost

    def sell(self, symbol: str, price: float, when: pd.Timestamp,
             exit_cost: float, reason: str) -> ClosedTrade:
        pos = self.positions.pop(symbol)
        self.cash += pos.quantity * price - exit_cost
        self.costs_paid += exit_cost
        trade = ClosedTrade(
            symbol=symbol, side="long", quantity=pos.quantity,
            entry_price=pos.entry_price, exit_price=price, strategy=pos.strategy,
            opened_at=pos.opened_at, closed_at=pd.Timestamp(when),
            entry_cost=pos.entry_cost, exit_cost=exit_cost, reason=reason,
        )
        self.closed.append(trade)
        return trade


@dataclass
class SwingResult:
    """Everything a run produced, in the shape `research.metrics` reads."""

    equity: pd.Series
    trades: list[ClosedTrade]
    initial_capital: float
    first_bar: pd.Timestamp
    last_bar: pd.Timestamp
    bars_tested: int
    rejections: dict[str, int] = field(default_factory=dict)
    strategy: str = ""
    exposure_days: int = 0

    @property
    def trade_count(self) -> int:
        return len(self.trades)

    def trades_frame(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame(columns=[
                "symbol", "side", "quantity", "entry_price", "exit_price", "strategy",
                "opened_at", "closed_at", "entry_cost", "exit_cost", "costs",
                "gross_pnl", "net_pnl", "bars_held", "exit_reason"])
        rows = []
        for t in self.trades:
            rows.append({
                "symbol": t.symbol, "side": t.side, "quantity": t.quantity,
                "entry_price": t.entry_price, "exit_price": t.exit_price,
                "strategy": t.strategy, "opened_at": t.opened_at, "closed_at": t.closed_at,
                "entry_cost": t.entry_cost, "exit_cost": t.exit_cost, "costs": t.costs,
                "gross_pnl": t.gross_pnl, "net_pnl": t.net_pnl,
                "bars_held": max((t.closed_at - t.opened_at).days, 0),
                "exit_reason": t.reason,
            })
        return pd.DataFrame(rows)

    def metrics(self):
        return compute_metrics(self, timeframe="D")

    @property
    def gross_pnl(self) -> float:
        return float(sum(t.gross_pnl for t in self.trades))

    @property
    def costs(self) -> float:
        return float(sum(t.costs for t in self.trades))

    @property
    def net_pnl(self) -> float:
        return float(sum(t.net_pnl for t in self.trades))


class SwingEngine:
    """Walk a panel forward one session at a time, holding across days."""

    def __init__(self, panel, strategy, cfg):
        self.panel = panel
        self.strategy = strategy
        self.cfg = cfg
        self.swing = cfg.swing
        self.model = get_cost_model(self.swing.costs.profile)
        self.slippage = self.swing.costs.slippage_bps / 10_000.0
        self.cost_rate = round_trip_rate(self.swing.costs)
        # Sizing scales with equity, not with the opening balance. Over four
        # years the difference is not a detail: fixed-fractional sizing on the
        # initial capital reports a compounding strategy's returns as if it had
        # been forbidden to compound, which flatters a losing run and penalises
        # a winning one. `_sizing_cfg` rebuilds the config view against this.
        self._equity_basis = float(cfg.capital)

    def _sizing_cfg(self):
        """The bot config with `capital` set to current equity.

        `size_position` reads `cfg.capital` for every cap it applies. Handing it
        a view rather than a second sizing function keeps one implementation of
        the caps, which is the only way they stay in agreement.
        """
        from dataclasses import replace
        if abs(self._equity_basis - self.cfg.capital) < 1e-9:
            return self.cfg
        return replace(self.cfg, account=replace(self.cfg.account,
                                                 capital=self._equity_basis))

    # ------------------------------------------------------------------- fills
    def _fill_price(self, price: float, side: str) -> float:
        """Slippage always moves against the order, on both legs."""
        return float(price * (1.0 + self.slippage) if side == "buy"
                     else price * (1.0 - self.slippage))

    def _charge(self, price: float, quantity: int, side: str) -> float:
        return float(self.model.charge(price, quantity, side))

    # -------------------------------------------------------------------- loop
    def run(self, start=None, end=None) -> SwingResult:
        panel = self.panel
        dates = panel.dates
        if start is not None:
            dates = dates[dates >= pd.Timestamp(start)]
        if end is not None:
            dates = dates[dates <= pd.Timestamp(end)]
        if len(dates) < 2:
            raise ValueError(f"swing run needs at least 2 sessions, got {len(dates)}")

        book = SwingBook(capital=self.cfg.capital)
        pending: list[PendingOrder] = []
        rejections: dict[str, int] = {}
        equity_rows: list[tuple[pd.Timestamp, float]] = []
        peak_equity = float(self.cfg.capital)
        exposure_days = 0

        opens, lows, closes = panel.open, panel.low, panel.close

        for date in dates:
            row_open = opens.loc[date]
            row_close = closes.loc[date]

            # 1. Yesterday's decisions become today's fills. Sells first: they
            #    release the cash the buys are then measured against.
            for order in sorted(pending, key=lambda o: 0 if o.action == "sell" else 1):
                if order.action == "sell":
                    self._execute_sell(book, order, date, row_open, row_close)
                else:
                    self._execute_buy(book, order, date, row_open, rejections)
            pending = []

            # 2. Stops, against today's actual path. A position opened at this
            #    morning's fill is included -- it can stop out the same day.
            self._apply_stops(book, date, row_open, lows)

            # 3. Trail what survived, from today's close.
            for pos in book.positions.values():
                px = row_close.get(pos.symbol, np.nan)
                if np.isfinite(px):
                    pos.stop = self.strategy.trail_stop(pos.symbol, date, float(px), pos.stop)

            marks = {s: float(row_close[s]) for s in book.positions
                     if s in row_close.index and np.isfinite(row_close[s])}
            equity = book.equity(marks)
            self._equity_basis = equity
            peak_equity = max(peak_equity, equity)
            equity_rows.append((date, equity))
            if book.positions:
                exposure_days += 1

            # 4. Rebalance decisions, on today's close, for tomorrow's open.
            if self._is_rebalance(date):
                pending = self._rebalance(book, date, equity, peak_equity, rejections)

        # Close whatever is still open at the final close, so the record is a
        # set of completed round trips and not a marked-to-market claim.
        last = dates[-1]
        for symbol in list(book.positions):
            px = closes.at[last, symbol]
            if not np.isfinite(px):
                px = book.positions[symbol].entry_price
            self._close(book, symbol, float(px), last, "end_of_study")
        if equity_rows:
            equity_rows[-1] = (last, book.equity({}))

        eq = pd.Series(dict(equity_rows)).sort_index()
        eq.index.name = "date"
        return SwingResult(
            equity=eq, trades=book.closed, initial_capital=float(self.cfg.capital),
            first_bar=dates[0], last_bar=dates[-1], bars_tested=len(dates),
            rejections=rejections, strategy=self.strategy.name,
            exposure_days=exposure_days,
        )

    # --------------------------------------------------------------- mechanics
    def _is_rebalance(self, date: pd.Timestamp) -> bool:
        return int(pd.Timestamp(date).weekday()) == self.swing.rules.rebalance_weekday

    def _execute_sell(self, book, order, date, row_open, row_close) -> None:
        if order.symbol not in book.positions:
            return                                   # already stopped out overnight
        px = row_open.get(order.symbol, np.nan)
        if not np.isfinite(px):
            px = row_close.get(order.symbol, np.nan)
        if not np.isfinite(px):
            return                                   # no bar to trade against; carry it
        self._close(book, order.symbol, float(px), date, order.reason or "rule")

    def _close(self, book, symbol, raw_price, date, reason) -> None:
        pos = book.positions[symbol]
        fill = self._fill_price(raw_price, "sell")
        cost = self._charge(fill, pos.quantity, "sell")
        book.sell(symbol, fill, date, cost, reason)

    def _execute_buy(self, book, order, date, row_open, rejections) -> None:
        raw = row_open.get(order.symbol, np.nan)
        if not np.isfinite(raw):
            rejections["no_open_price"] = rejections.get("no_open_price", 0) + 1
            return
        fill = self._fill_price(float(raw), "buy")

        # The stop was set last night. If the market opened through it, the
        # trade no longer exists -- taking it would be entering a position that
        # is already beyond the risk that justified it.
        if not np.isfinite(order.stop) or fill <= order.stop:
            rejections["gapped_through_stop"] = rejections.get("gapped_through_stop", 0) + 1
            return

        # `Signal` is built here for its invariants -- finite prices, stop the
        # right side of entry -- which are exactly the checks worth running on a
        # price that arrived overnight. The target is synthetic and unread: this
        # strategy exits on a rule, not at a level, and the sizer only ever
        # touches `entry` and `risk_per_unit`.
        signal = Signal(symbol=order.symbol, side="long", entry=fill, stop=order.stop,
                        target=fill + 2.0 * (fill - order.stop),
                        strategy=self.strategy.name, at=pd.Timestamp(date))
        decision = size_position(
            signal, self._sizing_cfg(), book,
            median_volume=self._median_volume(order.symbol, date),
            available_margin=book.cash,
            risk_limits=self.swing.risk, cost_rate=self.cost_rate,
        )
        if not decision.approved:
            key = decision.binding_cap or "sizing"
            rejections[key] = rejections.get(key, 0) + 1
            return

        entry_cost = self._charge(fill, decision.quantity, "buy")
        if decision.quantity * fill + entry_cost > book.cash:
            rejections["cash"] = rejections.get("cash", 0) + 1
            return

        book.buy(SwingPosition(
            symbol=order.symbol,
            security_id=self.panel.security_ids.get(order.symbol, ""),
            quantity=decision.quantity, entry_price=fill, stop=order.stop,
            opened_at=pd.Timestamp(date), strategy=self.strategy.name,
            entry_cost=entry_cost, planned_risk=decision.risk_amount,
            initial_stop=order.stop,
        ))

    def _apply_stops(self, book, date, row_open, lows) -> None:
        for symbol in list(book.positions):
            pos = book.positions[symbol]
            op = row_open.get(symbol, np.nan)
            lo = lows.at[date, symbol] if symbol in lows.columns else np.nan
            if not np.isfinite(lo):
                continue
            if np.isfinite(op) and op <= pos.stop:
                # Opened through the stop: the exit is the open, not the stop.
                self._close(book, symbol, float(op), date, "stop_gap")
            elif lo <= pos.stop:
                self._close(book, symbol, float(pos.stop), date, "stop")

    def _median_volume(self, symbol: str, date: pd.Timestamp) -> float:
        try:
            v = self.panel.volume[symbol].loc[:date].tail(
                self.swing.rules.turnover_window).median()
            return float(v)
        except (KeyError, IndexError):
            return float("nan")

    # -------------------------------------------------------------- rebalance
    def _rebalance(self, book, date, equity, peak_equity, rejections) -> list[PendingOrder]:
        ranking = self.strategy.rank(date)
        orders: list[PendingOrder] = []

        for symbol in list(book.positions):
            reason = self.strategy.exit_reason(symbol, date, ranking)
            if reason:
                orders.append(PendingOrder(symbol, "sell", date, reason=reason))

        leaving = {o.symbol for o in orders}
        room = self.swing.risk.max_concurrent_positions - (len(book.positions) - len(leaving))
        if room <= 0:
            return orders
        if not ranking.market_ok:
            rejections["index_filter"] = rejections.get("index_filter", 0) + 1
            return orders

        # Drawdown pause. Judged on equity against its peak, and it stops NEW
        # risk only -- open positions keep their stops, because closing them
        # here would be a second bet the study never tested.
        if equity < peak_equity - self.swing.risk.drawdown_pause(peak_equity):
            rejections["drawdown_pause"] = rejections.get("drawdown_pause", 0) + 1
            return orders

        held = set(book.positions) - leaving
        for symbol in ranking.entries:
            if room <= 0:
                break
            if symbol in held:
                continue
            close = self.panel.close.at[date, symbol]
            stop = self.strategy.initial_stop(symbol, date, float(close))
            if not np.isfinite(stop) or stop <= 0:
                rejections["no_stop"] = rejections.get("no_stop", 0) + 1
                continue
            orders.append(PendingOrder(symbol, "buy", date, stop=stop, reason="rank_entry"))
            room -= 1
        return orders
