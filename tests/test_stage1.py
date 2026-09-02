"""Validation for the Stage 1 estimators.

An estimator that has never been shown to detect a known edge, or to stay quiet
on a known null, cannot make a null result mean anything -- which is exactly the
mistake the first cross-sectional run made. So both are checked against
synthetic panels where the answer is constructed and therefore known.

These tests deliberately do NOT run the estimators on the real 3.9-year panel.
`docs/xsection_hypothesis.md` forbids it: that sample cannot resolve the
hypothesis, and producing a number from it invites the number being quoted.
Synthetic validation establishes the instrument works without spending the
hypothesis on a sample too short to answer it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kabali.xsection.momentum import LOOKBACK
from kabali.xsection.stage1 import (decile_spread, eligibility_mask,
                                    fama_macbeth, newey_west, run_stage1)

N_NAMES = 120
N_DAYS = LOOKBACK + 420
SIDE = 10


def _panel(edge: float, seed: int = 0):
    """A synthetic panel whose per-name drift is the only planted structure.

    With `edge` > 0 a name's past return predicts its future one, because both
    are driven by the same fixed drift. With `edge` == 0 the drifts are zero and
    momentum ranks pure noise.
    """
    rng = np.random.default_rng(seed)
    drift = rng.normal(0.0, edge, N_NAMES) if edge > 0 else np.zeros(N_NAMES)
    shocks = rng.normal(0.0, 0.015, (N_DAYS, N_NAMES)) + drift
    prices = 100.0 * np.exp(np.cumsum(shocks, axis=0))
    idx = pd.bdate_range("2005-01-03", periods=N_DAYS)
    cols = [f"N{i:03d}" for i in range(N_NAMES)]
    close = pd.DataFrame(prices, index=idx, columns=cols)
    turnover = pd.DataFrame(1e9, index=idx, columns=cols)
    return close, turnover


def test_newey_west_matches_plain_se_on_iid_data():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 4000)
    mu, se, t = newey_west(x, lags=21)
    plain = x.std(ddof=0) / np.sqrt(len(x))
    assert mu == pytest.approx(x.mean(), rel=1e-9)
    assert se == pytest.approx(plain, rel=0.35)     # same order, no correction


def test_newey_west_widens_se_under_positive_autocorrelation():
    """The whole reason for using it: overlapping windows must cost significance."""
    rng = np.random.default_rng(4)
    e = rng.normal(0, 1, 6000)
    x = np.convolve(e, np.ones(20) / 20, mode="same")   # strong positive AC
    _, nw_se, _ = newey_west(x, lags=21)
    plain = x.std(ddof=0) / np.sqrt(len(x))
    assert nw_se > 2.0 * plain


def test_spread_detects_a_planted_edge():
    close, turnover = _panel(edge=0.0012, seed=1)
    mask = eligibility_mask(close, turnover)
    res = decile_spread(close, mask, n_side=SIDE)
    assert res.rebalances > 5
    assert res.mean_daily > 0
    assert res.t_stat > 3.0, res.render()


def test_fama_macbeth_detects_a_planted_edge():
    close, turnover = _panel(edge=0.0012, seed=1)
    mask = eligibility_mask(close, turnover)
    res = fama_macbeth(close, mask)
    assert res.mean > 0
    assert res.t_stat > 2.0, res.render()


@pytest.mark.parametrize("seed", [11, 12, 13, 14])
def test_estimators_stay_quiet_on_a_null_panel(seed):
    """No planted edge: neither estimator may claim one.

    Run over several seeds because a single quiet draw proves nothing -- the
    failure this guards against is an estimator with a positive bias, which
    would show up as significance on most seeds rather than one.
    """
    close, turnover = _panel(edge=0.0, seed=seed)
    mask = eligibility_mask(close, turnover)
    assert abs(decile_spread(close, mask, n_side=SIDE).t_stat) < 3.0
    assert abs(fama_macbeth(close, mask).t_stat) < 3.0


def test_eligibility_screens_out_illiquid_and_penny_names():
    close, turnover = _panel(edge=0.0, seed=2)
    close.iloc[:, 0] = 5.0                       # below the price floor
    turnover.iloc[:, 1] = 1.0                    # below the turnover floor
    mask = eligibility_mask(close, turnover)
    assert not mask[-1, 0], "a Rs 5 share must not be rankable"
    assert not mask[-1, 1], "an illiquid name must not be rankable"
    assert mask[-1, 2], "a normal name must be rankable"


def test_short_sample_is_labelled_a_pilot_and_never_passes():
    """The pre-registered guard: a sample too short to resolve the hypothesis
    must not be able to report PASS, however the numbers land."""
    close, turnover = _panel(edge=0.0012, seed=1)
    res = run_stage1(close, turnover, n_side=SIDE)
    assert res.years < 15.0
    assert res.confirmatory is False
    assert "PILOT ONLY" in res.render()
