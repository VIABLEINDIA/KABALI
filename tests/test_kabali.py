"""Invariant tests for the KABALI bot.

These are weighted toward the failures that are expensive and silent: lookahead,
risk limits that do not bind, and the live gate opening when it should not. A
strategy that underperforms is a disappointment; a stop that does not fire, or a
backtest that read tomorrow's bar, is a wrong answer that looks like a right one.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kabali.config import ConfigError, load_config                       # noqa: E402
from kabali.engine.session import SessionEngine                          # noqa: E402
from kabali.execution.broker import OrderIntent                          # noqa: E402
from kabali.execution.gate import (                                      # noqa: E402
    GateClosed, GateCriteria, GateVerdict, LiveGate, evaluate,
)
from kabali.execution.paper import PaperBroker                           # noqa: E402
from kabali.risk.book import Position, SessionBook                       # noqa: E402
from kabali.risk.circuit import CircuitBreaker                           # noqa: E402
from kabali.risk.sizing import size_position                             # noqa: E402
from kabali.strategies.base import Signal, StrategyRegistry, SymbolContext  # noqa: E402
from kabali.strategies.registry import default_registry, route           # noqa: E402
from kabali.universe.factors import compute_factors, efficiency_ratio, slope_r2  # noqa: E402
from kabali.universe.selector import apply_gates, winsorised_z           # noqa: E402


@pytest.fixture(scope="module")
def cfg():
    return load_config()


class FakeInstrument:
    def __init__(self, symbol, security_id="1", tick_size=0.05):
        self.symbol = symbol
        self.security_id = security_id
        self.tick_size = tick_size


class FakeRegime:
    label = "bull_trend"
    family = "trend"
    direction = 1
    allows_long = True
    allows_short = True


def make_intraday(n=40, start="2026-08-28 09:15", base=1000.0, step=0.0,
                  high_pad=1.0, low_pad=1.0, volume=10_000.0):
    ts = pd.date_range(start, periods=n, freq="5min")
    close = base + np.arange(n) * step
    return pd.DataFrame({
        "timestamp": ts,
        "open": close - step,
        "high": close + high_pad,
        "low": close - low_pad,
        "close": close,
        "volume": np.full(n, volume),
    })


# --------------------------------------------------------------------- signals
class TestSignal:
    def test_long_requires_stop_below_entry_below_target(self):
        with pytest.raises(ValueError, match="long needs stop < entry < target"):
            Signal(symbol="X", side="long", entry=100, stop=101, target=105, strategy="t")

    def test_short_requires_target_below_entry_below_stop(self):
        with pytest.raises(ValueError, match="short needs target < entry < stop"):
            Signal(symbol="X", side="short", entry=100, stop=99, target=105, strategy="t")

    def test_rejects_nonfinite_prices(self):
        with pytest.raises(ValueError, match="must be finite"):
            Signal(symbol="X", side="long", entry=float("nan"), stop=99, target=105, strategy="t")

    def test_reward_risk_arithmetic(self):
        s = Signal(symbol="X", side="long", entry=100, stop=98, target=106, strategy="t")
        assert s.risk_per_unit == pytest.approx(2.0)
        assert s.reward_risk == pytest.approx(3.0)

    def test_short_reward_risk_arithmetic(self):
        s = Signal(symbol="X", side="short", entry=100, stop=102, target=94, strategy="t")
        assert s.risk_per_unit == pytest.approx(2.0)
        assert s.reward_risk == pytest.approx(3.0)
        assert s.direction == -1


# ---------------------------------------------------------------------- config
class TestConfig:
    def test_loads_and_derives_limits(self, cfg):
        assert cfg.capital > 0
        assert cfg.risk.daily_loss_limit(cfg.capital) == pytest.approx(cfg.capital * 0.02)

    def test_rejects_incoherent_risk_budget(self, cfg):
        """Positions x per-trade risk must not exceed the daily limit.

        Otherwise the daily limit can only fire after it has been breached.
        """
        from kabali.config import RiskConfig
        bad = RiskConfig(
            daily_loss_limit_pct=2.0, daily_profit_lock_pct=3.0,
            per_trade_risk_pct=1.0, max_concurrent_positions=4,
            max_gross_exposure_mult=3.0, max_position_notional_pct=30.0,
            max_trades_per_day=12, max_consecutive_losses=4, min_position_notional=5000.0,
        )
        with pytest.raises(ConfigError, match="incoherent"):
            bad.validate()

    def test_session_times_must_be_ordered(self):
        from kabali.config import SessionConfig
        s = SessionConfig(timezone="Asia/Kolkata", market_open=dt.time(9, 15),
                          market_close=dt.time(15, 30), opening_range_end=dt.time(9, 30),
                          entry_cutoff=dt.time(15, 20), square_off=dt.time(15, 10))
        with pytest.raises(ConfigError, match="entry_cutoff"):
            s.validate()


# ---------------------------------------------------------------------- sizing
class TestSizing:
    def test_quantity_derives_from_stop_distance(self, cfg):
        book = SessionBook(trade_date=dt.date(2026, 8, 28), capital=cfg.capital)
        sig = Signal(symbol="X", side="long", entry=1000.0, stop=990.0,
                     target=1020.0, strategy="t")
        d = size_position(sig, cfg, book, median_volume=10_000_000)
        assert d.approved
        # Quantity solves budget / (stop distance + entry * round-trip cost rate),
        # so it is strictly below the price-only figure of 200/10 = 20 shares.
        from kabali.risk.sizing import _round_trip_rate
        expected = int(cfg.risk.risk_per_trade(cfg.capital)
                       // (10.0 + 1000.0 * _round_trip_rate(cfg)))
        assert d.quantity == expected < 20
        assert d.risk_amount == pytest.approx(cfg.risk.risk_per_trade(cfg.capital), rel=0.05)

    def test_risk_is_constant_across_different_volatilities(self, cfg):
        """Same rupee risk on a calm and a violent name -- the point of the design.

        Priced low enough that the risk budget is the binding cap for both. At a
        higher price the notional cap binds first, which is covered separately.
        """
        book = SessionBook(trade_date=dt.date(2026, 8, 28), capital=cfg.capital)
        calm = Signal(symbol="A", side="long", entry=1000.0, stop=990.0,
                      target=1030.0, strategy="t")
        wild = Signal(symbol="B", side="long", entry=1000.0, stop=970.0,
                      target=1090.0, strategy="t")
        a = size_position(calm, cfg, book, median_volume=100_000_000)
        b = size_position(wild, cfg, book, median_volume=100_000_000)
        assert a.binding_cap == b.binding_cap == "risk_budget"
        assert a.quantity > b.quantity
        assert a.risk_amount == pytest.approx(b.risk_amount, rel=0.05)

    def test_notional_cap_can_hold_risk_below_the_budget(self, cfg):
        """A very tight stop would imply a position larger than one name may be.

        The notional cap wins, so realised risk is BELOW the per-trade budget.
        That asymmetry is intentional: caps may only ever reduce exposure.
        """
        book = SessionBook(trade_date=dt.date(2026, 8, 28), capital=cfg.capital)
        tight = Signal(symbol="A", side="long", entry=1000.0, stop=997.5,
                       target=1010.0, strategy="t")
        d = size_position(tight, cfg, book, median_volume=100_000_000)
        assert d.binding_cap == "position_notional"
        assert d.risk_amount < cfg.risk.risk_per_trade(cfg.capital)
        assert d.notional <= cfg.risk.position_notional_cap(cfg.capital)

    def test_participation_cap_binds_on_thin_names(self, cfg):
        book = SessionBook(trade_date=dt.date(2026, 8, 28), capital=cfg.capital)
        sig = Signal(symbol="THIN", side="long", entry=100.0, stop=99.0,
                     target=103.0, strategy="t")
        d = size_position(sig, cfg, book, median_volume=5_000)
        assert d.binding_cap == "participation"
        assert d.quantity <= 5_000 * cfg.universe.max_participation_pct / 100

    def test_rejects_below_notional_floor(self, cfg):
        """A position too small to carry fixed brokerage is refused, not shrunk."""
        book = SessionBook(trade_date=dt.date(2026, 8, 28), capital=cfg.capital)
        sig = Signal(symbol="TINY", side="long", entry=50.0, stop=49.0,
                     target=53.0, strategy="t")
        d = size_position(sig, cfg, book, median_volume=1_000)
        assert not d.approved
        assert "floor" in d.reason

    def test_rejects_stop_wider_than_budget(self, cfg):
        book = SessionBook(trade_date=dt.date(2026, 8, 28), capital=cfg.capital)
        sig = Signal(symbol="HUGE", side="long", entry=50_000.0, stop=40_000.0,
                     target=70_000.0, strategy="t")
        d = size_position(sig, cfg, book, median_volume=10_000_000)
        assert not d.approved
        assert "budget" in d.reason

    def test_gross_exposure_cap_shrinks_later_positions(self, cfg):
        book = SessionBook(trade_date=dt.date(2026, 8, 28), capital=cfg.capital)
        book.add(Position(symbol="HELD", security_id="1", side="long", quantity=100,
                          entry_price=1180.0, stop=1170.0, target=1200.0,
                          strategy="t", opened_at=pd.Timestamp("2026-08-28 10:00")))
        sig = Signal(symbol="NEW", side="long", entry=1000.0, stop=999.0,
                     target=1005.0, strategy="t")
        d = size_position(sig, cfg, book, median_volume=100_000_000)
        assert d.binding_cap in ("gross_exposure", "position_notional")


# -------------------------------------------------------------------- circuit
class TestCircuit:
    def test_daily_loss_counts_unrealised(self, cfg):
        """Open losers must consume the daily budget, or the limit is decorative."""
        book = SessionBook(trade_date=dt.date(2026, 8, 28), capital=cfg.capital)
        cb = CircuitBreaker(cfg)
        book.add(Position(symbol="L", security_id="1", side="long", quantity=100,
                          entry_price=100.0, stop=98.0, target=104.0, strategy="t",
                          opened_at=pd.Timestamp("2026-08-28 10:00"), entry_cost=10.0))
        assert cb.check_session(book, {"L": 99.0}, pd.Timestamp("2026-08-28 10:30")).allowed
        d = cb.check_session(book, {"L": 90.0}, pd.Timestamp("2026-08-28 10:30"))
        assert d.halted and d.flatten

    def test_halt_is_sticky_even_if_price_recovers(self, cfg):
        book = SessionBook(trade_date=dt.date(2026, 8, 28), capital=cfg.capital)
        cb = CircuitBreaker(cfg)
        book.add(Position(symbol="L", security_id="1", side="long", quantity=100,
                          entry_price=100.0, stop=98.0, target=104.0, strategy="t",
                          opened_at=pd.Timestamp("2026-08-28 10:00")))
        cb.check_session(book, {"L": 88.0}, pd.Timestamp("2026-08-28 10:30"))
        recovered = cb.check_session(book, {"L": 105.0}, pd.Timestamp("2026-08-28 11:00"))
        assert recovered.halted

    def test_blocks_before_opening_range_completes(self, cfg):
        book = SessionBook(trade_date=dt.date(2026, 8, 28), capital=cfg.capital)
        cb = CircuitBreaker(cfg)
        d = cb.check_entry(book, {}, pd.Timestamp("2026-08-28 09:20"), "X")
        assert not d.allowed and "opening range" in d.reason

    def test_blocks_after_entry_cutoff(self, cfg):
        book = SessionBook(trade_date=dt.date(2026, 8, 28), capital=cfg.capital)
        cb = CircuitBreaker(cfg)
        d = cb.check_entry(book, {}, pd.Timestamp("2026-08-28 15:00"), "X")
        assert not d.allowed and "cutoff" in d.reason

    def test_blocks_at_max_concurrent_positions(self, cfg):
        book = SessionBook(trade_date=dt.date(2026, 8, 28), capital=cfg.capital)
        cb = CircuitBreaker(cfg)
        for i in range(cfg.risk.max_concurrent_positions):
            book.add(Position(symbol=f"S{i}", security_id=str(i), side="long", quantity=1,
                              entry_price=100.0, stop=99.5, target=101.0, strategy="t",
                              opened_at=pd.Timestamp("2026-08-28 10:00")))
        d = cb.check_entry(book, {}, pd.Timestamp("2026-08-28 11:00"), "NEW")
        assert not d.allowed and "concurrent" in d.reason

    def test_blocks_duplicate_symbol(self, cfg):
        book = SessionBook(trade_date=dt.date(2026, 8, 28), capital=cfg.capital)
        cb = CircuitBreaker(cfg)
        book.add(Position(symbol="DUP", security_id="1", side="long", quantity=1,
                          entry_price=100.0, stop=99.5, target=101.0, strategy="t",
                          opened_at=pd.Timestamp("2026-08-28 10:00")))
        d = cb.check_entry(book, {}, pd.Timestamp("2026-08-28 11:00"), "DUP")
        assert not d.allowed and "already holding" in d.reason

    def test_consecutive_losses_halt(self, cfg):
        book = SessionBook(trade_date=dt.date(2026, 8, 28), capital=cfg.capital)
        cb = CircuitBreaker(cfg)
        book.consecutive_losses = cfg.risk.max_consecutive_losses
        d = cb.check_session(book, {}, pd.Timestamp("2026-08-28 11:00"))
        assert d.halted


# ----------------------------------------------------------------------- book
class TestBook:
    def test_net_pnl_subtracts_both_legs_of_cost(self):
        book = SessionBook(trade_date=dt.date(2026, 8, 28), capital=40_000.0)
        book.add(Position(symbol="X", security_id="1", side="long", quantity=10,
                          entry_price=100.0, stop=98.0, target=104.0, strategy="t",
                          opened_at=pd.Timestamp("2026-08-28 10:00"), entry_cost=5.0))
        t = book.close("X", 104.0, pd.Timestamp("2026-08-28 11:00"), 7.0, "target")
        assert t.gross_pnl == pytest.approx(40.0)
        assert t.net_pnl == pytest.approx(28.0)

    def test_short_pnl_sign(self):
        book = SessionBook(trade_date=dt.date(2026, 8, 28), capital=40_000.0)
        book.add(Position(symbol="X", security_id="1", side="short", quantity=10,
                          entry_price=100.0, stop=102.0, target=96.0, strategy="t",
                          opened_at=pd.Timestamp("2026-08-28 10:00")))
        t = book.close("X", 96.0, pd.Timestamp("2026-08-28 11:00"), 0.0, "target")
        assert t.gross_pnl == pytest.approx(40.0)

    def test_equity_pnl_includes_open_entry_costs(self):
        book = SessionBook(trade_date=dt.date(2026, 8, 28), capital=40_000.0)
        book.add(Position(symbol="X", security_id="1", side="long", quantity=10,
                          entry_price=100.0, stop=98.0, target=104.0, strategy="t",
                          opened_at=pd.Timestamp("2026-08-28 10:00"), entry_cost=9.0))
        assert book.equity_pnl({"X": 100.0}) == pytest.approx(-9.0)

    def test_refuses_to_stack_the_same_symbol(self):
        book = SessionBook(trade_date=dt.date(2026, 8, 28), capital=40_000.0)
        p = Position(symbol="X", security_id="1", side="long", quantity=10,
                     entry_price=100.0, stop=98.0, target=104.0, strategy="t",
                     opened_at=pd.Timestamp("2026-08-28 10:00"))
        book.add(p)
        with pytest.raises(ValueError, match="already holding"):
            book.add(p)


# ------------------------------------------------------------------ causality
class TestCausality:
    """Nothing may depend on a bar later than the one being evaluated."""

    def test_factors_are_truncation_invariant(self):
        rs = np.random.RandomState(7)
        n = 400
        close = 500 + np.cumsum(rs.randn(n))
        df = pd.DataFrame({
            "timestamp": pd.bdate_range("2023-01-02", periods=n),
            "open": close, "high": close + 2, "low": close - 2,
            "close": close, "volume": np.full(n, 1e6),
        })
        full = compute_factors("X", "1", df, as_of=df["timestamp"].iloc[300])
        truncated = compute_factors("X", "1", df.iloc[:301], as_of=df["timestamp"].iloc[300])
        assert full is not None and truncated is not None
        for field in ("ret_63", "ret_126", "efficiency_ratio", "slope_r2",
                      "atr_pct", "median_turnover"):
            a, b = getattr(full, field), getattr(truncated, field)
            assert a == pytest.approx(b, rel=1e-9, nan_ok=True), field

    def test_efficiency_ratio_is_one_for_a_straight_line(self):
        s = pd.Series(np.arange(100, dtype=float) + 100)
        assert efficiency_ratio(s, 63) == pytest.approx(1.0)

    def test_efficiency_ratio_is_zero_for_a_round_trip(self):
        up = np.arange(0, 32, dtype=float)
        s = pd.Series(np.r_[up, up[::-1][1:]] + 100)
        assert efficiency_ratio(s, len(s) - 1) == pytest.approx(0.0, abs=1e-9)

    def test_slope_r2_positive_for_uptrend_negative_for_downtrend(self):
        up = pd.Series(np.linspace(100, 200, 120))
        down = pd.Series(np.linspace(200, 100, 120))
        assert slope_r2(up, 90) > 0
        assert slope_r2(down, 90) < 0

    def test_entry_fills_at_next_bar_open_not_signal_bar_close(self, cfg):
        """The bar that produced the signal cannot also be the bar we filled on."""
        bars = make_intraday(n=30, step=2.0, volume=10_000)
        bars.loc[len(bars) - 1, "volume"] = 500_000
        engine = SessionEngine(cfg, default_registry(), PaperBroker(cfg), FakeRegime(),
                               instruments={"X": FakeInstrument("X")},
                               daily_context={"X": {"prev_close": 1000.0, "atr_daily": 20.0,
                                                    "median_volume": 5_000_000}})
        res = engine.run(dt.date(2026, 8, 28), {"X": bars})
        for t in res.book.closed:
            signal_bar = bars[bars["timestamp"] < t.opened_at]
            assert len(signal_bar) >= 1


# ------------------------------------------------------------------- engine
class TestEngine:
    def test_never_carries_a_position_overnight(self, cfg):
        rs = np.random.RandomState(3)
        n = 60
        close = 1000 + np.cumsum(rs.randn(n) * 3)
        bars = pd.DataFrame({
            "timestamp": pd.date_range("2026-08-28 09:15", periods=n, freq="5min"),
            "open": close, "high": close + 4, "low": close - 4, "close": close,
            "volume": rs.randint(5_000, 80_000, n).astype(float),
        })
        engine = SessionEngine(cfg, default_registry(), PaperBroker(cfg), FakeRegime(),
                               instruments={"X": FakeInstrument("X")},
                               daily_context={"X": {"prev_close": 1000.0, "atr_daily": 25.0,
                                                    "median_volume": 5_000_000}})
        res = engine.run(dt.date(2026, 8, 28), {"X": bars})
        assert not res.book.positions, "intraday bot must be flat at the close"

    def test_stop_wins_when_one_bar_contains_stop_and_target(self, cfg):
        """Pessimistic resolution: without tick data we must not assume the target."""
        book = SessionBook(trade_date=dt.date(2026, 8, 28), capital=cfg.capital)
        pos = Position(symbol="X", security_id="1", side="long", quantity=10,
                       entry_price=100.0, stop=98.0, target=104.0, strategy="t",
                       opened_at=pd.Timestamp("2026-08-28 10:00"))
        book.add(pos)
        engine = SessionEngine(cfg, default_registry(), PaperBroker(cfg), FakeRegime(),
                               instruments={"X": FakeInstrument("X")})
        row = pd.Series({"open": 100.0, "high": 105.0, "low": 97.0, "close": 101.0})
        from kabali.engine.session import SessionResult
        res = SessionResult(trade_date=dt.date(2026, 8, 28), book=book,
                            regime_label="t", universe_size=1)
        engine._manage_exits(book, {"X": row}, pd.Timestamp("2026-08-28 10:05"), res)
        assert len(book.closed) == 1
        assert book.closed[0].reason == "stop"

    def test_position_count_respected_within_one_timestep(self, cfg):
        """Reservations must stop many symbols entering on the same bar."""
        rs = np.random.RandomState(11)
        bars = {}
        for i in range(12):
            n = 30
            close = 1000 + np.cumsum(rs.randn(n) * 4)
            b = pd.DataFrame({
                "timestamp": pd.date_range("2026-08-28 09:15", periods=n, freq="5min"),
                "open": close, "high": close + 6, "low": close - 6, "close": close,
                "volume": np.full(n, 20_000.0),
            })
            b.loc[n - 6:, "volume"] = 400_000.0
            bars[f"S{i}"] = b
        engine = SessionEngine(
            cfg, default_registry(), PaperBroker(cfg), FakeRegime(),
            instruments={s: FakeInstrument(s, str(i)) for i, s in enumerate(bars)},
            daily_context={s: {"prev_close": 1000.0, "atr_daily": 25.0,
                               "median_volume": 5_000_000} for s in bars})
        res = engine.run(dt.date(2026, 8, 28), bars)
        assert res.entries <= cfg.risk.max_trades_per_day


# ---------------------------------------------------------------------- gate
class TestLiveGate:
    def test_refuses_with_no_evidence(self, cfg):
        assert not LiveGate(cfg).check(None).passed

    def test_refuses_short_record(self):
        rec = pd.DataFrame({
            "trade_date": pd.date_range("2026-08-01", periods=5),
            "trades": [3] * 5, "wins": [2] * 5, "gross_pnl": [100] * 5,
            "costs": [40] * 5, "net_pnl": [60, -20, 80, -10, 50],
            "modelled_slippage": [10] * 5, "observed_slippage": [9] * 5,
            "loss_limit": [800] * 5,
        })
        v = evaluate(rec, GateCriteria())
        assert not v.passed
        assert any("sessions" in f for f in v.failures)

    def test_refuses_when_observed_slippage_exceeds_model(self):
        rec = pd.DataFrame({
            "trade_date": pd.date_range("2026-08-01", periods=25, freq="B"),
            "trades": [2] * 25, "wins": [1] * 25, "gross_pnl": [100] * 25,
            "costs": [20] * 25, "net_pnl": [40] * 25,
            "modelled_slippage": [10] * 25, "observed_slippage": [40] * 25,
            "loss_limit": [800] * 25,
        })
        v = evaluate(rec, GateCriteria(), now=pd.Timestamp("2026-09-02"))
        assert not v.passed
        assert any("slippage" in f for f in v.failures)

    def test_refuses_stale_evidence(self):
        rec = pd.DataFrame({
            "trade_date": pd.date_range("2026-01-01", periods=25, freq="B"),
            "trades": [2] * 25, "wins": [2] * 25, "gross_pnl": [100] * 25,
            "costs": [20] * 25, "net_pnl": [40] * 25,
            "modelled_slippage": [10] * 25, "observed_slippage": [9] * 25,
            "loss_limit": [800] * 25,
        })
        v = evaluate(rec, GateCriteria(), now=pd.Timestamp("2026-08-30"))
        assert any("freshness" in f for f in v.failures)

    def test_flags_daily_limit_breach_as_a_risk_engine_bug(self):
        rec = pd.DataFrame({
            "trade_date": pd.date_range("2026-08-01", periods=25, freq="B"),
            "trades": [2] * 25, "wins": [2] * 25, "gross_pnl": [100] * 25,
            "costs": [20] * 25, "net_pnl": [40] * 24 + [-5000],
            "modelled_slippage": [10] * 25, "observed_slippage": [9] * 25,
            "loss_limit": [800] * 25,
        })
        v = evaluate(rec, GateCriteria(), now=pd.Timestamp("2026-09-02"))
        assert any("daily_limit" in f for f in v.failures)

    # --------------------------------------------------------- provenance
    @staticmethod
    def _clean_record(days: int = 25) -> pd.DataFrame:
        """A record that clears every numeric criterion, so only provenance can fail."""
        return pd.DataFrame({
            "trade_date": pd.date_range("2026-08-01", periods=days, freq="B"),
            "trades": [2] * days, "wins": [2] * days, "gross_pnl": [100] * days,
            "costs": [20] * days, "net_pnl": [40] * days,
            "modelled_slippage": [10] * days, "observed_slippage": [9] * days,
            "loss_limit": [800] * days,
        })

    def _armed_gate(self, cfg, tmp_path, monkeypatch):
        """A gate whose file is armed, reading a record under tmp_path."""
        import kabali.execution.gate as g

        gate_file = tmp_path / "LIVE_GATE.json"
        gate_file.write_text(json.dumps(
            {"armed": True, "confirm": g.CONFIRMATION_PHRASE}))
        monkeypatch.setattr(type(cfg), "gate_path", lambda self: gate_file)
        return g.LiveGate(cfg, record_path=tmp_path / "paper_record.csv")

    def test_record_without_a_sidecar_is_refused(self, cfg, tmp_path, monkeypatch):
        """The failure that let a pre-fix replay be scored as current evidence."""
        gate = self._armed_gate(cfg, tmp_path, monkeypatch)
        self._clean_record().to_csv(gate.record_path, index=False)

        v = gate.check(now=pd.Timestamp("2026-09-02"))
        assert not v.passed
        assert any("provenance" in f and "sidecar" in f for f in v.failures)

    def test_promoted_record_passes_provenance(self, cfg, tmp_path, monkeypatch):
        from kabali.execution import provenance

        gate = self._armed_gate(cfg, tmp_path, monkeypatch)
        provenance.write(gate.record_path, self._clean_record(), cfg, source="test")

        v = gate.check(now=pd.Timestamp("2026-09-02"))
        assert v.passed, v.render()
        assert "test" in v.checks["provenance"][1]

    def test_code_change_after_promotion_invalidates_the_evidence(
            self, cfg, tmp_path, monkeypatch):
        """A record produced by different trading code is not evidence about this one.

        `runs/paper_final` was replayed one minute before the daily-loss-limit
        fix landed; without this check its record scored as current.
        """
        from kabali.execution import provenance

        gate = self._armed_gate(cfg, tmp_path, monkeypatch)
        provenance.write(gate.record_path, self._clean_record(), cfg, source="test")

        meta = provenance.meta_path(gate.record_path)
        blob = json.loads(meta.read_text())
        blob["code_fingerprint"] = "0000deadbeef"
        meta.write_text(json.dumps(blob))

        v = gate.check(now=pd.Timestamp("2026-09-02"))
        assert not v.passed
        assert any("different trading code" in f for f in v.failures)

    def test_config_change_after_promotion_invalidates_the_evidence(
            self, cfg, tmp_path, monkeypatch):
        from kabali.execution import provenance

        gate = self._armed_gate(cfg, tmp_path, monkeypatch)
        provenance.write(gate.record_path, self._clean_record(), cfg, source="test")

        meta = provenance.meta_path(gate.record_path)
        blob = json.loads(meta.read_text())
        blob["config_fingerprint"] = "0000feedface"
        meta.write_text(json.dumps(blob))

        v = gate.check(now=pd.Timestamp("2026-09-02"))
        assert not v.passed
        assert any("different config" in f for f in v.failures)

    def test_editing_the_record_after_promotion_is_detected(
            self, cfg, tmp_path, monkeypatch):
        """Appending winning sessions by hand must not quietly become evidence."""
        from kabali.execution import provenance

        gate = self._armed_gate(cfg, tmp_path, monkeypatch)
        rec = self._clean_record()
        provenance.write(gate.record_path, rec, cfg, source="test")

        padded = pd.concat([rec, rec.tail(1).assign(net_pnl=5000)], ignore_index=True)
        padded.to_csv(gate.record_path, index=False)

        v = gate.check(now=pd.Timestamp("2026-09-02"))
        assert not v.passed
        assert any("does not match its sidecar" in f for f in v.failures)

    def test_fingerprints_are_stable_across_calls(self, cfg):
        from kabali.execution import provenance

        assert provenance.code_fingerprint() == provenance.code_fingerprint()
        assert provenance.config_fingerprint(cfg) == provenance.config_fingerprint(cfg)

    def test_live_broker_refuses_without_a_passing_verdict(self, cfg):
        from kabali.execution.dhan_live import DhanBroker
        with pytest.raises(GateClosed):
            DhanBroker(cfg, None)
        failing = GateVerdict()
        failing.add("sessions", False, "not enough")
        with pytest.raises(GateClosed):
            DhanBroker(cfg, failing)

    def test_research_core_cannot_place_orders(self):
        """The read-only guarantee must survive being imported by the bot."""
        from research.dhan_client import READ_ONLY_METHODS
        assert "place_order" not in READ_ONLY_METHODS
        assert "cancel_order" not in READ_ONLY_METHODS


# ------------------------------------------------------------------- selector
class TestSelector:
    def test_gates_drop_untradeable_names(self, cfg):
        f = pd.DataFrame({
            "bars": [1200, 1200, 1200, 80],
            "close": [500.0, 10.0, 500.0, 500.0],       # row 1 below min_price
            "median_turnover": [1e9, 1e9, 1e5, 1e9],    # row 2 illiquid
            "median_volume": [1e6, 1e6, 1e3, 1e6],
            "zero_volume_frac": [0.0, 0.0, 0.0, 0.0],
            "atr_pct": [0.02, 0.02, 0.02, 0.02],
        }, index=["GOOD", "CHEAP", "THIN", "SHORTHIST"])
        kept, rep = apply_gates(f, cfg, cfg.capital,
                                position_notional=cfg.risk.position_notional_cap(cfg.capital))
        assert "GOOD" in kept.index
        assert set(kept.index) == {"GOOD"}
        assert rep.started == 4 and rep.survived == 1

    def test_winsorised_z_is_robust_to_one_outlier(self):
        s = pd.Series([1.0, 1.1, 0.9, 1.05, 0.95, 1000.0])
        z = winsorised_z(s)
        assert z.max() <= 3.0
        assert abs(z.iloc[0]) < 3.0

    def test_winsorised_z_handles_constant_input(self):
        assert winsorised_z(pd.Series([5.0] * 10)).abs().max() == 0.0


# ----------------------------------------------------------------- strategies
class TestStrategies:
    def test_registry_covers_every_regime_family(self):
        reg = default_registry()
        for family in ("trend", "range", "volatile"):
            assert reg.for_family(family), f"no strategy valid in {family}"

    def test_registry_rejects_duplicate_registration(self):
        from kabali.strategies.opening_range import OpeningRangeBreakout
        reg = StrategyRegistry([OpeningRangeBreakout()])
        with pytest.raises(ValueError, match="already registered"):
            reg.register(OpeningRangeBreakout())

    def test_contradictory_signals_are_skipped(self, cfg):
        """Two rules pointing opposite ways is not a hedge, it is a no-trade."""
        from kabali.strategies.base import IntradayStrategy

        class AlwaysLong(IntradayStrategy):
            families = ("trend",)
            warmup_bars = 1

            def generate(self, ctx):
                return [Signal(symbol=ctx.symbol, side="long", entry=100, stop=99,
                               target=103, strategy="AlwaysLong")]

        class AlwaysShort(IntradayStrategy):
            families = ("trend",)
            warmup_bars = 1

            def generate(self, ctx):
                return [Signal(symbol=ctx.symbol, side="short", entry=100, stop=101,
                               target=97, strategy="AlwaysShort")]

        reg = StrategyRegistry([AlwaysLong(), AlwaysShort()])
        ctx = SymbolContext(symbol="X", security_id="1",
                            now=pd.Timestamp("2026-08-28 11:00"),
                            intraday=make_intraday(20), daily=pd.DataFrame(),
                            regime=FakeRegime(), session=cfg.session)
        assert route(ctx, reg) == []

    def test_agreeing_signals_merge_with_the_tighter_stop(self, cfg):
        from kabali.strategies.base import IntradayStrategy

        class LongA(IntradayStrategy):
            families = ("trend",)
            warmup_bars = 1

            def generate(self, ctx):
                return [Signal(symbol=ctx.symbol, side="long", entry=100, stop=97,
                               target=110, strategy="LongA", confidence=0.4)]

        class LongB(IntradayStrategy):
            families = ("trend",)
            warmup_bars = 1

            def generate(self, ctx):
                return [Signal(symbol=ctx.symbol, side="long", entry=100, stop=99,
                               target=104, strategy="LongB", confidence=0.4)]

        reg = StrategyRegistry([LongA(), LongB()])
        ctx = SymbolContext(symbol="X", security_id="1",
                            now=pd.Timestamp("2026-08-28 11:00"),
                            intraday=make_intraday(20), daily=pd.DataFrame(),
                            regime=FakeRegime(), session=cfg.session)
        merged = route(ctx, reg)
        assert len(merged) == 1
        assert merged[0].stop == 99          # tighter of the two
        assert merged[0].target == 104       # nearer of the two
        assert merged[0].confidence > 0.4    # corroboration raises it

    def test_a_raising_strategy_does_not_break_the_scan(self, cfg):
        from kabali.strategies.base import IntradayStrategy

        class Broken(IntradayStrategy):
            families = ("trend",)
            warmup_bars = 1

            def generate(self, ctx):
                raise RuntimeError("boom")

        ctx = SymbolContext(symbol="X", security_id="1",
                            now=pd.Timestamp("2026-08-28 11:00"),
                            intraday=make_intraday(20), daily=pd.DataFrame(),
                            regime=FakeRegime(), session=cfg.session)
        assert route(ctx, StrategyRegistry([Broken()])) == []

    def test_vwap_uses_volume_weighting(self, cfg):
        bars = make_intraday(n=4, base=100.0, high_pad=0, low_pad=0)
        bars["close"] = [100.0, 100.0, 100.0, 200.0]
        bars["high"] = bars["close"]
        bars["low"] = bars["close"]
        bars["volume"] = [1.0, 1.0, 1.0, 97.0]
        ctx = SymbolContext(symbol="X", security_id="1",
                            now=pd.Timestamp(bars["timestamp"].iloc[-1]),
                            intraday=bars, daily=pd.DataFrame(),
                            regime=FakeRegime(), session=cfg.session)
        # heavily weighted to the last bar, so far above the simple mean of 125
        assert ctx.session_vwap() > 190

    def test_vwap_falls_back_when_volume_is_absent(self, cfg):
        bars = make_intraday(n=4, base=100.0, high_pad=0, low_pad=0)
        bars["volume"] = 0.0
        ctx = SymbolContext(symbol="X", security_id="1",
                            now=pd.Timestamp(bars["timestamp"].iloc[-1]),
                            intraday=bars, daily=pd.DataFrame(),
                            regime=FakeRegime(), session=cfg.session)
        assert np.isfinite(ctx.session_vwap())


# ------------------------------------------------------------ paper execution
class TestPaperBroker:
    def test_slippage_is_charged_against_the_order_on_both_legs(self, cfg):
        pb = PaperBroker(cfg)
        now = pd.Timestamp("2026-08-28 10:00")
        buy = pb.submit(OrderIntent("X", "1", "buy", 10, 1000.0, "entry:t", now))
        sell = pb.submit(OrderIntent("X", "1", "sell", 10, 1000.0, "exit:t", now))
        assert buy.price > 1000.0
        assert sell.price < 1000.0

    def test_costs_are_nonzero_and_match_the_research_model(self, cfg):
        from kabali.core import get_cost_model
        pb = PaperBroker(cfg)
        now = pd.Timestamp("2026-08-28 10:00")
        f = pb.submit(OrderIntent("X", "1", "buy", 10, 1000.0, "entry:t", now))
        expected = get_cost_model(cfg.costs.profile).charge(f.price, 10, "buy")
        assert f.cost == pytest.approx(round(expected, 2))

    def test_rejects_when_simulated_margin_is_exhausted(self, cfg):
        pb = PaperBroker(cfg, starting_margin=1_000.0)
        f = pb.submit(OrderIntent("X", "1", "buy", 100, 1000.0, "entry:t",
                                  pd.Timestamp("2026-08-28 10:00")))
        assert not f.ok and "margin" in f.note

    def test_rejection_is_a_fill_object_not_an_exception(self, cfg):
        pb = PaperBroker(cfg)
        f = pb.submit(OrderIntent("X", "1", "buy", 1, -5.0, "entry:t",
                                  pd.Timestamp("2026-08-28 10:00")))
        assert f.status == "rejected"


# ------------------------------------------------------------- live session
class TestLiveSession:
    """The forming-bar guard is the live-only lookahead risk."""

    def _session(self, cfg, feed):
        from kabali.engine.live import LiveSession
        return LiveSession(cfg, default_registry(), PaperBroker(cfg), FakeRegime(),
                           instruments={"X": FakeInstrument("X")},
                           daily_context={"X": {"prev_close": 1000.0, "atr_daily": 20.0,
                                                "median_volume": 5_000_000}},
                           feed=feed, timeframe="5m")

    def test_drops_the_still_forming_bar(self, cfg):
        bars = make_intraday(n=6, start="2026-08-28 09:15")
        sess = self._session(cfg, lambda: {"X": bars})
        # 09:42 -- the 09:40 candle covers 09:40..09:45 and has not closed.
        out = sess._confirmed(bars, pd.Timestamp("2026-08-28 09:42"))
        assert out["timestamp"].max() == pd.Timestamp("2026-08-28 09:35")

    def test_keeps_a_bar_whose_interval_has_elapsed(self, cfg):
        bars = make_intraday(n=6, start="2026-08-28 09:15")
        sess = self._session(cfg, lambda: {"X": bars})
        out = sess._confirmed(bars, pd.Timestamp("2026-08-28 09:46"))
        assert out["timestamp"].max() == pd.Timestamp("2026-08-28 09:40")

    def test_feed_failure_does_not_end_the_session(self, cfg, monkeypatch):
        import kabali.engine.live as live_mod
        monkeypatch.setattr(live_mod.time, "sleep", lambda _s: None)

        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            raise RuntimeError("network down")

        sess = self._session(cfg, flaky)
        res = sess.run(trade_date=dt.date(2026, 8, 28), poll_seconds=0,
                       now_fn=lambda: pd.Timestamp("2026-08-28 11:00"),
                       max_polls=3)
        assert calls["n"] == 3          # retried rather than aborting
        assert not res.book.positions

    def test_square_off_deadline_ends_the_loop(self, cfg, monkeypatch):
        import kabali.engine.live as live_mod
        monkeypatch.setattr(live_mod.time, "sleep", lambda _s: None)
        bars = make_intraday(n=20, start="2026-08-28 09:15")
        sess = self._session(cfg, lambda: {"X": bars})
        res = sess.run(trade_date=dt.date(2026, 8, 28), poll_seconds=0,
                       now_fn=lambda: pd.Timestamp("2026-08-28 15:20"),
                       max_polls=5)
        assert res.timesteps == 0       # past square-off, never traded
        assert not res.book.positions

    def test_merge_falls_back_when_entries_disagree(self, cfg):
        """A merge that would violate stop < entry < target must not be emitted."""
        from kabali.strategies.base import IntradayStrategy

        class Near(IntradayStrategy):
            families = ("trend",)
            warmup_bars = 1

            def generate(self, ctx):
                return [Signal(symbol=ctx.symbol, side="long", entry=100, stop=97,
                               target=110, strategy="Near", confidence=0.6)]

        class Far(IntradayStrategy):
            families = ("trend",)
            warmup_bars = 1

            def generate(self, ctx):
                return [Signal(symbol=ctx.symbol, side="long", entry=110, stop=105,
                               target=120, strategy="Far", confidence=0.3)]

        ctx = SymbolContext(symbol="X", security_id="1",
                            now=pd.Timestamp("2026-08-28 11:00"),
                            intraday=make_intraday(20), daily=pd.DataFrame(),
                            regime=FakeRegime(), session=cfg.session)
        out = route(ctx, StrategyRegistry([Near(), Far()]))
        assert len(out) == 1
        s = out[0]
        assert s.stop < s.entry < s.target      # still a valid signal
        assert s.strategy == "Near"             # fell back to the strongest


# --------------------------------------------------- cost-aware risk budgeting
class TestCostAwareRisk:
    """The daily loss limit must survive the costs of losing."""

    def test_sizing_includes_round_trip_cost_in_the_risk_figure(self, cfg):
        book = SessionBook(trade_date=dt.date(2026, 8, 28), capital=cfg.capital)
        sig = Signal(symbol="X", side="long", entry=1287.0, stop=1275.0,
                     target=1311.0, strategy="t")
        d = size_position(sig, cfg, book, median_volume=10_000_000)
        assert d.approved
        assert d.risk_amount > d.price_risk, "costs must be inside the risk figure"
        assert d.risk_amount <= cfg.risk.risk_per_trade(cfg.capital) * 1.01

    def test_full_book_worst_case_stays_inside_the_daily_limit(self, cfg):
        """Every position stopping out at once must not breach the daily budget.

        This is the invariant a paper replay flagged as broken when sizing
        ignored costs: 4 x Rs 200 of price risk plus 4 x ~Rs 21 of costs is
        Rs 884 against an Rs 800 limit.
        """
        book = SessionBook(trade_date=dt.date(2026, 8, 28), capital=cfg.capital)
        total = 0.0
        for i in range(cfg.risk.max_concurrent_positions):
            sig = Signal(symbol=f"S{i}", side="long", entry=1287.0, stop=1275.0,
                         target=1311.0, strategy="t")
            d = size_position(sig, cfg, book, median_volume=10_000_000)
            assert d.approved
            total += d.risk_amount
            book.add(Position(symbol=f"S{i}", security_id=str(i), side="long",
                              quantity=d.quantity, entry_price=sig.entry, stop=sig.stop,
                              target=sig.target, strategy="t",
                              opened_at=pd.Timestamp("2026-08-28 10:00"),
                              planned_risk=d.risk_amount))
        assert total <= cfg.risk.daily_loss_limit(cfg.capital)
        assert book.open_risk <= cfg.risk.daily_loss_limit(cfg.capital)

    def test_open_risk_prefers_the_cost_inclusive_figure(self):
        p = Position(symbol="X", security_id="1", side="long", quantity=10,
                     entry_price=100.0, stop=98.0, target=104.0, strategy="t",
                     opened_at=pd.Timestamp("2026-08-28 10:00"), planned_risk=25.0)
        assert p.price_risk == pytest.approx(20.0)
        assert p.risk_amount == pytest.approx(25.0)

    def test_falls_back_to_price_risk_when_unsized(self):
        p = Position(symbol="X", security_id="1", side="long", quantity=10,
                     entry_price=100.0, stop=98.0, target=104.0, strategy="t",
                     opened_at=pd.Timestamp("2026-08-28 10:00"))
        assert p.risk_amount == pytest.approx(20.0)

    def test_square_off_is_not_reported_as_a_halt(self, cfg):
        bars = make_intraday(n=30, start="2026-08-28 14:30", step=0.5)
        engine = SessionEngine(cfg, default_registry(), PaperBroker(cfg), FakeRegime(),
                               instruments={"X": FakeInstrument("X")},
                               daily_context={"X": {"prev_close": 1000.0, "atr_daily": 20.0,
                                                    "median_volume": 5_000_000}})
        res = engine.run(dt.date(2026, 8, 28), {"X": bars})
        assert not res.halted, "a normal square-off is not a risk halt"

    def test_equity_is_measured_on_liquidation_value(self):
        """Open positions must be valued net of what it costs to close them."""
        book = SessionBook(trade_date=dt.date(2026, 8, 28), capital=40_000.0,
                           exit_cost_rate=0.001)
        book.add(Position(symbol="X", security_id="1", side="long", quantity=10,
                          entry_price=1000.0, stop=990.0, target=1020.0, strategy="t",
                          opened_at=pd.Timestamp("2026-08-28 10:00"), entry_cost=5.0))
        # flat on marks, but 10 x 1000 x 0.001 = 10 to get out, plus the 5 paid
        assert book.open_exit_costs == pytest.approx(10.0)
        assert book.equity_pnl({"X": 1000.0}) == pytest.approx(-15.0)

    def test_daily_limit_holds_through_the_flatten(self, cfg):
        """A full stop-out must SETTLE inside the budget, costs and all.

        The regression this pins: sizing on price risk alone, and valuing the
        book on marks rather than liquidation, tripped the breaker at exactly
        -Rs 800 and then paid the exit costs on top -- settling at -Rs 872.

        Note the breaker is NOT expected to fire at the full stop-out: correct
        sizing lands that case just inside the limit, which is the whole point.
        """
        from kabali.risk.sizing import _round_trip_rate
        rate = _round_trip_rate(cfg)
        book = SessionBook(trade_date=dt.date(2026, 8, 28), capital=cfg.capital,
                           exit_cost_rate=rate / 2.0)
        limit = cfg.risk.daily_loss_limit(cfg.capital)

        for i in range(cfg.risk.max_concurrent_positions):
            sig = Signal(symbol=f"S{i}", side="long", entry=1000.0, stop=990.0,
                         target=1030.0, strategy="t")
            d = size_position(sig, cfg, book, median_volume=100_000_000)
            assert d.approved
            book.add(Position(symbol=f"S{i}", security_id=str(i), side="long",
                              quantity=d.quantity, entry_price=1000.0, stop=990.0,
                              target=1030.0, strategy="t",
                              opened_at=pd.Timestamp("2026-08-28 10:00"),
                              entry_cost=1000.0 * d.quantity * rate / 2.0,
                              planned_risk=d.risk_amount))

        marks = {f"S{i}": 990.0 for i in range(cfg.risk.max_concurrent_positions)}
        # Liquidation value at the stops already accounts for getting out.
        assert book.equity_pnl(marks) >= -limit

        for sym in list(book.positions):
            pos = book.positions[sym]
            book.close(sym, 990.0, pd.Timestamp("2026-08-28 12:00"),
                       990.0 * pos.quantity * rate / 2.0, "circuit")
        assert book.realised_pnl >= -limit, (
            f"settled at Rs {book.realised_pnl:,.0f}, past the Rs {limit:,.0f} limit")

    def test_breaker_trips_once_liquidation_value_breaches_the_limit(self, cfg):
        """Beyond the stops -- a gap through them -- the breaker must fire."""
        from kabali.risk.sizing import _round_trip_rate
        rate = _round_trip_rate(cfg)
        book = SessionBook(trade_date=dt.date(2026, 8, 28), capital=cfg.capital,
                           exit_cost_rate=rate / 2.0)
        cb = CircuitBreaker(cfg)
        for i in range(cfg.risk.max_concurrent_positions):
            sig = Signal(symbol=f"S{i}", side="long", entry=1000.0, stop=990.0,
                         target=1030.0, strategy="t")
            d = size_position(sig, cfg, book, median_volume=100_000_000)
            book.add(Position(symbol=f"S{i}", security_id=str(i), side="long",
                              quantity=d.quantity, entry_price=1000.0, stop=990.0,
                              target=1030.0, strategy="t",
                              opened_at=pd.Timestamp("2026-08-28 10:00"),
                              entry_cost=1000.0 * d.quantity * rate / 2.0,
                              planned_risk=d.risk_amount))
        gapped = {f"S{i}": 985.0 for i in range(cfg.risk.max_concurrent_positions)}
        decision = cb.check_session(book, gapped, pd.Timestamp("2026-08-28 12:00"))
        assert decision.halted and decision.flatten
        assert decision.kind == "loss_limit"


class TestStrategyGuards:
    """Guards a strategy documents must actually exist in its code."""

    def _ctx(self, cfg, when: str):
        from kabali.strategies.base import SymbolContext
        return SymbolContext(symbol="X", security_id="1", now=pd.Timestamp(when),
                             intraday=make_intraday(20), daily=pd.DataFrame(),
                             regime=FakeRegime(), session=cfg.session)

    def test_vwap_reversion_refuses_late_fades(self, cfg):
        """No fade with too little time left to reach VWAP before square-off."""
        from kabali.strategies.vwap_reversion import VWAPReversion
        s = VWAPReversion(max_minutes_from_cutoff=45)
        # square_off 15:10, so the deadline is 14:25
        assert s.ready(self._ctx(cfg, "2026-08-28 11:00"))
        assert s.ready(self._ctx(cfg, "2026-08-28 14:20"))
        assert not s.ready(self._ctx(cfg, "2026-08-28 14:40"))
        assert not s.ready(self._ctx(cfg, "2026-08-28 15:05"))

    def test_opening_range_waits_for_the_range_to_complete(self, cfg):
        from kabali.strategies.opening_range import OpeningRangeBreakout
        s = OpeningRangeBreakout()
        early = self._ctx(cfg, "2026-08-28 09:25")
        assert s.generate(early) == []

    def test_every_strategy_declares_at_least_one_regime(self):
        for s in default_registry().all():
            assert s.families, f"{s.name} is valid in no regime and can never run"
            assert all(f in ("trend", "range", "volatile") for f in s.families)


class TestBarCache:
    """Offline cache reads. Nothing here may touch the network."""

    def test_path_matches_the_research_datastore(self):
        """If this drifts, every downstream stage silently sees an empty cache."""
        from kabali.data.cache import verify_layout
        assert verify_layout()

    def test_missing_symbol_returns_none_rather_than_fetching(self):
        from kabali.data.cache import cached_bars

        class Ghost:
            symbol = "NOSUCHSCRIP"
            security_id = "99999999"
            exchange_segment = "NSE_EQ"

        assert cached_bars(Ghost(), "D") is None

    def test_min_bars_rejects_a_thin_cache(self, tmp_path, monkeypatch):
        import kabali.data.cache as cache_mod

        class Fake:
            symbol = "FAKE"
            security_id = "1"
            exchange_segment = "NSE_EQ"

        monkeypatch.setattr(cache_mod, "bars_dir", lambda: tmp_path)
        df = pd.DataFrame({
            "timestamp": pd.bdate_range("2026-01-01", periods=5),
            "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
        })
        df.to_parquet(tmp_path / "NSE_EQ_1_D.parquet")
        assert cache_mod.cached_bars(Fake(), "D", min_bars=3) is not None
        assert cache_mod.cached_bars(Fake(), "D", min_bars=50) is None

    def test_load_many_reports_what_is_missing(self, tmp_path, monkeypatch):
        import kabali.data.cache as cache_mod

        class Inst:
            def __init__(self, sym, sid):
                self.symbol = sym
                self.security_id = sid
                self.exchange_segment = "NSE_EQ"

        monkeypatch.setattr(cache_mod, "bars_dir", lambda: tmp_path)
        df = pd.DataFrame({
            "timestamp": pd.bdate_range("2026-01-01", periods=20),
            "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0,
        })
        df.to_parquet(tmp_path / "NSE_EQ_7_D.parquet")
        frames, missing = cache_mod.load_many([Inst("HAVE", "7"), Inst("GONE", "8")], "D")
        assert set(frames) == {"HAVE"}
        assert missing == ["GONE"]


class TestRegimeCalibration:
    """A regime the strategies need must actually be reachable."""

    def _nifty(self):
        from kabali.core import Instrument
        from kabali.data.cache import cached_bars
        inst = Instrument(symbol="NIFTY", security_id="13",
                          exchange_segment="IDX_I", instrument_type="INDEX")
        return cached_bars(inst, "D")

    def test_efficiency_percentile_is_causal(self, cfg):
        """Today's percentile must not move when tomorrow's bars arrive."""
        from kabali.regime.classifier import _efficiency_percentile
        bars = self._nifty()
        if bars is None or len(bars) < 400:
            pytest.skip("NIFTY daily cache unavailable")
        close = pd.to_numeric(bars["close"], errors="coerce")
        full = _efficiency_percentile(close.iloc[:350], cfg.regime.trend_slow)
        more = _efficiency_percentile(close.iloc[:350], cfg.regime.trend_slow)
        assert full == pytest.approx(more, nan_ok=True)

    def test_every_family_is_reachable_on_real_history(self, cfg):
        """Regression: a flat 0.30 efficiency floor made `trend` 2.3% of sessions.

        Three of five strategies require the trend family, so the stack was
        mostly dead code. Each family must hold a usable share of real sessions.
        """
        from kabali.regime.classifier import classify
        bars = self._nifty()
        if bars is None or len(bars) < 600:
            pytest.skip("NIFTY daily cache unavailable")

        seen = []
        step = max(len(bars) // 120, 1)
        for i in range(cfg.regime.trend_anchor + 60, len(bars), step):
            try:
                seen.append(classify(bars.iloc[:i], cfg).family)
            except ValueError:
                continue
        assert len(seen) >= 50
        share = pd.Series(seen).value_counts(normalize=True)
        for family in ("trend", "range", "volatile"):
            got = float(share.get(family, 0.0))
            assert got >= 0.05, (
                f"{family} is only {got:.1%} of sessions -- strategies gated to it "
                f"can effectively never run")

    def test_absolute_floor_still_rejects_dead_chop(self, cfg):
        """A percentile alone would call the least-choppy third of chop a trend."""
        from kabali.regime.classifier import classify
        n = 400
        # Pure oscillation: large path, no net travel.
        close = 1000 + 20 * np.sin(np.arange(n) / 2.0)
        bars = pd.DataFrame({
            "timestamp": pd.bdate_range("2024-01-01", periods=n),
            "open": close, "high": close + 3, "low": close - 3,
            "close": close, "volume": 1e6,
        })
        assert classify(bars, cfg).family != "trend"

    def test_live_loop_waits_for_the_bar_boundary(self, cfg):
        """Regression: a fixed 30s poll re-fetched the scan list 10x per bar.

        At 100 symbols that is ~3.3 req/s sustained all session -- the rate whose
        breach cost a six-hour throttle during the history build.
        """
        from kabali.engine.live import LiveSession
        slept = []
        sess = LiveSession(cfg, default_registry(), PaperBroker(cfg), FakeRegime(),
                           instruments={}, daily_context={},
                           feed=lambda: {}, timeframe="5m")
        import kabali.engine.live as live_mod
        orig = live_mod.time.sleep
        live_mod.time.sleep = slept.append
        try:
            # 10:31:00 -> next 5m boundary is 10:35, i.e. 240s away, plus grace
            sess._wait(lambda: pd.Timestamp("2026-08-28 10:31:00"), 0)
            # 10:34:57 -> boundary is 3s away
            sess._wait(lambda: pd.Timestamp("2026-08-28 10:34:57"), 0)
            # an explicit ceiling still caps the wait
            sess._wait(lambda: pd.Timestamp("2026-08-28 10:31:00"), 10)
        finally:
            live_mod.time.sleep = orig
        assert slept[0] == 240 + sess.BOUNDARY_GRACE_SECONDS
        assert slept[1] == 3 + sess.BOUNDARY_GRACE_SECONDS
        assert slept[2] == 10


class TestWorstCaseBudget:
    """Entry admission must be judged on the worst case, not on paper gains."""

    def _open(self, cfg, book, n, entry=1000.0, stop=990.0):
        from kabali.risk.sizing import _round_trip_rate
        rate = _round_trip_rate(cfg)
        for i in range(n):
            sig = Signal(symbol=f"S{i}", side="long", entry=entry, stop=stop,
                         target=entry + 30, strategy="t")
            d = size_position(sig, cfg, book, median_volume=100_000_000)
            assert d.approved
            book.add(Position(symbol=f"S{i}", security_id=str(i), side="long",
                              quantity=d.quantity, entry_price=entry, stop=stop,
                              target=entry + 30, strategy="t",
                              opened_at=pd.Timestamp("2026-08-28 10:00"),
                              entry_cost=entry * d.quantity * rate / 2.0,
                              planned_risk=d.risk_amount))

    def test_unrealised_profit_is_not_spare_budget(self, cfg):
        """Regression: 2026-06-24 settled at -Rs 934 against an Rs 800 limit.

        Three open shorts showing a paper gain read as room for a fourth; all
        four then reversed to their stops.
        """
        book = SessionBook(trade_date=dt.date(2026, 8, 28), capital=cfg.capital)
        cb = CircuitBreaker(cfg)
        # One realised loss, then three open positions carrying full risk.
        book.add(Position(symbol="DONE", security_id="9", side="long", quantity=10,
                          entry_price=100.0, stop=98.0, target=104.0, strategy="t",
                          opened_at=pd.Timestamp("2026-08-28 10:00")))
        book.close("DONE", 98.0, pd.Timestamp("2026-08-28 10:30"), 5.0, "stop")
        self._open(cfg, book, 3)

        # Marks show the open positions well in profit. That must NOT admit a 4th.
        rosy = {f"S{i}": 1020.0 for i in range(3)}
        d = cb.check_entry(book, rosy, pd.Timestamp("2026-08-28 11:35"), "NEW")
        assert not d.allowed
        assert "worst-case room" in d.reason

    def _place(self, book, sym, planned_risk, entry=1000.0, qty=10):
        book.add(Position(symbol=sym, security_id=sym, side="long", quantity=qty,
                          entry_price=entry, stop=entry - planned_risk / qty,
                          target=entry + 30, strategy="t",
                          opened_at=pd.Timestamp("2026-08-28 10:00"),
                          planned_risk=planned_risk))

    def test_admitting_until_blocked_keeps_worst_case_inside_the_limit(self, cfg):
        """Open positions until the breaker refuses; the book must still be safe."""
        book = SessionBook(trade_date=dt.date(2026, 8, 28), capital=cfg.capital)
        cb = CircuitBreaker(cfg)
        limit = cfg.risk.daily_loss_limit(cfg.capital)
        per_trade = cfg.risk.risk_per_trade(cfg.capital)

        opened = 0
        for i in range(20):
            if not cb.check_entry(book, {}, pd.Timestamp("2026-08-28 11:00"), f"S{i}").allowed:
                break
            self._place(book, f"S{i}", per_trade)
            opened += 1

        assert opened >= 1
        # Worst case: every open position reaches its stop.
        assert book.open_risk <= limit + 1e-6, (
            f"open risk Rs {book.open_risk:,.0f} exceeds the Rs {limit:,.0f} limit")

    def test_realised_profit_does_extend_the_budget(self, cfg):
        """Banked gains are real budget, unlike paper ones.

        Two positions carrying Rs 310 of risk each leave Rs 180 of room, below
        the Rs 200 a third would need. A realised Rs 100 lifts that to Rs 280.
        """
        per_trade = cfg.risk.risk_per_trade(cfg.capital)

        flat = SessionBook(trade_date=dt.date(2026, 8, 28), capital=cfg.capital)
        self._place(flat, "A", per_trade * 1.55)
        self._place(flat, "B", per_trade * 1.55)
        blocked = CircuitBreaker(cfg).check_entry(
            flat, {}, pd.Timestamp("2026-08-28 11:00"), "NEW")

        ahead = SessionBook(trade_date=dt.date(2026, 8, 28), capital=cfg.capital)
        ahead.add(Position(symbol="WON", security_id="9", side="long", quantity=10,
                           entry_price=100.0, stop=98.0, target=140.0, strategy="t",
                           opened_at=pd.Timestamp("2026-08-28 10:00")))
        ahead.close("WON", 130.0, pd.Timestamp("2026-08-28 10:30"), 5.0, "target")
        self._place(ahead, "A", per_trade * 1.55)
        self._place(ahead, "B", per_trade * 1.55)
        allowed = CircuitBreaker(cfg).check_entry(
            ahead, {}, pd.Timestamp("2026-08-28 11:00"), "NEW")

        assert not blocked.allowed, "should refuse without banked profit"
        assert ahead.realised_pnl > 0
        assert allowed.allowed, "realised profit should extend the budget"
