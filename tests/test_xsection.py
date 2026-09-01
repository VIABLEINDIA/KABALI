"""Tests for the cross-sectional layer.

Weighted towards the failures that would manufacture an edge rather than break a
run. A crash is cheap -- it announces itself. A backtest that quietly peeks one
day into the future, spends cash it never had, or holds fractional shares reports
a plausible number that is wrong, and nothing in the output says so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kabali.xsection import controls as ctl
from kabali.xsection import evaluate as ev
from kabali.xsection import momentum as mom
from kabali.xsection import portfolio as pf
from kabali.xsection.panel import bad_prints


# ------------------------------------------------------------------ data hygiene
class TestBadPrints:
    def test_catches_a_reverting_spike(self):
        # The HATSUN shape: 912, 468, 900. A split does not come back.
        close = pd.Series([900.0, 905.0, 912.0, 468.0, 900.0, 898.0])
        assert bad_prints(close) == [3]

    def test_leaves_a_genuine_split_alone(self):
        # SIRCA's shape: the halving is permanent, so it is a corporate action
        # for `research.corporate_actions` to handle, not a print to mask.
        close = pd.Series([620.0, 621.0, 304.0, 305.0, 307.0, 310.0])
        assert bad_prints(close) == []

    def test_leaves_a_real_crash_alone(self):
        # A 20% fall that keeps falling is news, not a bad tick.
        close = pd.Series([100.0, 99.0, 79.0, 76.0, 74.0, 73.0])
        assert bad_prints(close) == []

    def test_short_series_is_not_an_error(self):
        assert bad_prints(pd.Series([1.0, 2.0])) == []


# ----------------------------------------------------------------------- signal
class TestMomentumSignal:
    def test_uses_only_data_before_the_skip_window(self):
        """The score on date t must not move when prices after t-skip change.

        This is the lookahead test. If the last month's prices leak into the
        signal, doubling them changes the score, and the whole backtest is
        reporting returns nobody could have earned.
        """
        n = 400
        close = pd.DataFrame(
            {"A": np.linspace(100, 200, n)},
            index=pd.bdate_range("2022-01-03", periods=n))
        base = mom.momentum(close).iloc[-1, 0]

        tampered = close.copy()
        tampered.iloc[-mom.SKIP:, 0] *= 2.0        # rewrite the skipped month
        after = mom.momentum(tampered).iloc[-1, 0]

        assert base == pytest.approx(after), "signal leaked into the skip window"

    def test_is_return_over_the_formation_window(self):
        n = 300
        idx = pd.bdate_range("2022-01-03", periods=n)
        close = pd.DataFrame({"A": np.full(n, 100.0)}, index=idx)
        close.iloc[-mom.SKIP - 1, 0] = 150.0
        got = mom.momentum(close).iloc[-1, 0]
        assert got == pytest.approx(0.5)

    def test_requires_both_ends_of_the_window(self):
        n = 300
        idx = pd.bdate_range("2022-01-03", periods=n)
        close = pd.DataFrame({"A": np.linspace(100, 200, n)}, index=idx)
        close.iloc[0:5, 0] = np.nan                # not listed a year ago
        turnover = pd.DataFrame(np.full((n, 1), 1e9), index=idx, columns=["A"])
        mask = mom.eligible(close, turnover)
        assert not bool(mask.iloc[mom.LOOKBACK + 2, 0])


class TestEligibility:
    def _frames(self, n=320, price=100.0, turnover=1e9):
        idx = pd.bdate_range("2022-01-03", periods=n)
        close = pd.DataFrame({"A": np.full(n, price)}, index=idx)
        turn = pd.DataFrame({"A": np.full(n, turnover)}, index=idx)
        return close, turn

    def test_illiquid_names_are_excluded(self):
        close, turn = self._frames(turnover=1e5)
        assert not mom.eligible(close, turn).iloc[-1, 0]

    def test_penny_prices_are_excluded(self):
        close, turn = self._frames(price=5.0)
        assert not mom.eligible(close, turn).iloc[-1, 0]

    def test_a_liquid_formed_name_is_included(self):
        close, turn = self._frames()
        assert mom.eligible(close, turn).iloc[-1, 0]


class TestRanking:
    def test_returns_best_first_and_respects_the_mask(self):
        scores = pd.Series({"A": 0.5, "B": 0.9, "C": 0.7, "D": 0.99})
        mask = pd.Series({"A": True, "B": True, "C": True, "D": False})
        assert mom.rank_top(scores, mask, 2) == ["B", "C"]

    def test_is_deterministic_under_ties(self):
        scores = pd.Series({"B": 1.0, "A": 1.0, "C": 1.0})
        mask = pd.Series({"B": True, "A": True, "C": True})
        first = mom.rank_top(scores, mask, 2)
        assert first == mom.rank_top(scores.sample(frac=1, random_state=3), mask, 2)
        assert first == ["A", "B"]

    def test_empty_when_nothing_is_eligible(self):
        scores = pd.Series({"A": 0.5})
        assert mom.rank_top(scores, pd.Series({"A": False}), 5) == []


# -------------------------------------------------------------------- portfolio
def _panel(n_days=400, n_symbols=6, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n_days)
    syms = [f"S{i}" for i in range(n_symbols)]
    rets = rng.normal(0.0005, 0.02, size=(n_days, n_symbols))
    close = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=idx, columns=syms)
    open_ = close.shift(1).fillna(close.iloc[0])
    turn = pd.DataFrame(1e9, index=idx, columns=syms)
    return close, open_, turn


class TestPortfolioEngine:
    def _run(self, **kw):
        close, open_, turn = _panel()
        cfg = pf.XSecConfig(capital=40_000.0, n_positions=3, rebalance_days=21, **kw)
        res = pf.run(close, open_, mom.momentum(close), mom.eligible(close, turn), cfg)
        return close, open_, res

    def test_holds_whole_shares_only(self):
        _, _, res = self._run()
        assert len(res.trades), "no trades produced; the fixture is not exercising the engine"
        qty = res.trades["qty"]
        assert (qty == qty.astype(int)).all()
        assert (qty > 0).all()

    def test_cash_never_goes_negative(self):
        """A CNC delivery account cannot borrow. Not once, on any day."""
        _, _, res = self._run()
        assert res.equity.notna().all()
        assert (res.cash >= -1e-9).all(), f"account borrowed: min cash {res.cash.min():.2f}"

    def test_fills_come_from_the_execution_day_open(self):
        """Every buy price must be the next session's open, slipped -- never the
        close the decision was made on."""
        close, open_, res = self._run()
        buys = res.trades[res.trades.side == "buy"]
        for _, t in buys.iterrows():
            expected = open_.loc[t["date"], t["symbol"]] * (1 + 3.0 / 10_000)
            assert t["price"] == pytest.approx(round(expected, 2), abs=0.02)

    def test_costs_are_charged_on_every_trade(self):
        _, _, res = self._run()
        assert (res.trades["cost"] > 0).all()

    def test_position_count_never_exceeds_the_target(self):
        _, _, res = self._run()
        assert res.positions.max() <= res.config.n_positions
        assert res.positions.max() > 0, "the engine never took a position"

    def test_sells_are_funded_before_buys(self):
        """On any rebalance day the engine must not spend more than it holds.

        Buying first and selling afterwards would let the book spend proceeds it
        has not received -- accidental leverage, and in a delivery account,
        leverage that does not exist.
        """
        _, _, res = self._run()
        spend = res.trades[res.trades.side == "buy"].groupby("date").apply(
            lambda g: (g["price"] * g["qty"] + g["cost"]).sum(), include_groups=False)
        for date, outlay in spend.items():
            available = res.config.capital + res.equity.loc[:date].max()
            assert outlay <= available

    def test_a_later_price_cannot_change_an_earlier_decision(self):
        """Rewriting the future must leave the past of the equity curve intact.

        The sharpest available lookahead test: run once, then multiply the final
        fifty sessions by three and run again. Any date before the change whose
        equity moves is proof the engine read forward.
        """
        close, open_, turn = _panel()
        cfg = pf.XSecConfig(capital=40_000.0, n_positions=3, rebalance_days=21)
        args = (mom.eligible(close, turn), cfg)
        base = pf.run(close, open_, mom.momentum(close), *args).equity

        tampered_c, tampered_o = close.copy(), open_.copy()
        tampered_c.iloc[-50:] *= 3.0
        tampered_o.iloc[-50:] *= 3.0
        after = pf.run(tampered_c, tampered_o,
                       mom.momentum(tampered_c), *args).equity

        cut = base.index[-51]
        pd.testing.assert_series_equal(base.loc[:cut], after.loc[:cut])


# --------------------------------------------------------------------- evaluate
class TestAlphaBeta:
    def test_recovers_a_planted_beta(self):
        rng = np.random.default_rng(0)
        idx = pd.bdate_range("2022-01-03", periods=600)
        rb = rng.normal(0.0004, 0.01, 600)
        rs = 1.5 * rb                      # pure beta, zero alpha by construction
        bench = pd.Series(100 * np.cumprod(1 + rb), index=idx)
        strat = pd.Series(100 * np.cumprod(1 + rs), index=idx)
        got = ev.alpha_beta(strat, bench)
        assert got.beta == pytest.approx(1.5, abs=0.05)
        assert abs(got.t_alpha) < 2.0

    def test_detects_a_planted_alpha(self):
        rng = np.random.default_rng(1)
        idx = pd.bdate_range("2022-01-03", periods=600)
        rb = rng.normal(0.0004, 0.01, 600)
        rs = rb + 0.0008                   # a steady 20%/yr of pure alpha
        bench = pd.Series(100 * np.cumprod(1 + rb), index=idx)
        strat = pd.Series(100 * np.cumprod(1 + rs), index=idx)
        got = ev.alpha_beta(strat, bench)
        assert got.t_alpha > 3.0
        assert got.alpha_annual_pct > 10.0

    def test_verdict_refuses_to_claim_an_edge_without_significance(self):
        ab = ev.AlphaBeta(beta=1.2, alpha_annual_pct=4.0, t_alpha=0.5,
                          t_beta_vs_one=5.0, correlation=0.8, observations=900)
        assert "no significant alpha" in ab.verdict()

    def test_rejects_too_short_a_sample(self):
        idx = pd.bdate_range("2022-01-03", periods=10)
        s = pd.Series(np.linspace(100, 110, 10), index=idx)
        with pytest.raises(ValueError):
            ev.alpha_beta(s, s)


class TestSyntheticControls:
    def test_noise_panel_has_no_cross_sectional_persistence(self):
        close, _ = ctl.synthetic_panel(n_symbols=60, n_days=500, seed=5,
                                       momentum_strength=0.0)
        scores = mom.momentum(close).iloc[400].dropna()
        forward = (close.iloc[-1] / close.iloc[400] - 1).reindex(scores.index)
        assert abs(np.corrcoef(scores, forward)[0, 1]) < 0.30

    def test_planted_edge_is_actually_present(self):
        close, _ = ctl.synthetic_panel(n_symbols=60, n_days=500, seed=5,
                                       momentum_strength=0.20)
        scores = mom.momentum(close).iloc[400].dropna()
        forward = (close.iloc[-1] / close.iloc[400] - 1).reindex(scores.index)
        assert np.corrcoef(scores, forward)[0, 1] > 0.30

    def test_market_factor_gives_the_benchmark_real_variance(self):
        """Without a common factor the equal-weight benchmark is a straight line
        and every comparison against it is meaningless."""
        close, _ = ctl.synthetic_panel(n_symbols=200, n_days=500, seed=5)
        ew = (1 + close.pct_change(fill_method=None).mean(axis=1)).cumprod().dropna()
        annual_vol = ew.pct_change().std(ddof=1) * np.sqrt(252)
        assert 0.05 < annual_vol < 0.40

    def test_opens_are_not_equal_to_closes(self):
        close, open_ = ctl.synthetic_panel(n_symbols=20, n_days=300, seed=2)
        diff = (open_ - close.shift(1)).abs().to_numpy()
        assert np.nanmax(diff) > 0, "fills would be free if opens matched closes"
