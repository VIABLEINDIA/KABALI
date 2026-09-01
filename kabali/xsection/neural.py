"""A neural cross-sectional return model, built so that it can fail.

WHY A NETWORK HERE AND NOWHERE ELSE
===================================
The intraday venue has 426 round trips. Fitting a network to that would not be
research, it would be an elaborate way of memorising three months. The daily
panel is a different object: 1,237 sessions x 606 names, roughly 750,000
name-days, which is enough that a small model has something to generalise from.

So this is the one place in the repo where a learned model is defensible, and
even here the burden is on the model to prove it, not on the reader to assume it.

WHAT WOULD MAKE THIS DISHONEST, AND WHAT PREVENTS IT
====================================================
CHRONOLOGY. Every split is by date, never shuffled. A random split lets the model
see next month while predicting this one, and cross-sectional return data is
correlated enough across names that the leak is enormous and invisible -- the
score simply comes back excellent.

FEATURE CAUSALITY. Features at date t use closes up to and including t; the
label is the forward return from t to t+horizon. The gap is enforced by
construction, and a `--shift-check` mode deliberately breaks it so the tests can
confirm the leak would have been detectable.

STANDARDISATION INSIDE THE FOLD. Means and scales are fitted on training rows
only. Fitting them on the whole panel leaks the test period's distribution into
training, which is the most common quiet mistake in financial ML.

THE TEST SET IS TOUCHED ONCE. Walk-forward folds are for development. The final
holdout is scored exactly once, at the end, and its number is the one that counts.

THE CONTROLS DECIDE WHETHER THE ANSWER MEANS ANYTHING
=====================================================
The same pipeline runs on two synthetic panels from `xsection.controls`:

  PLANTED EDGE  a panel with a real momentum effect. If the model cannot find
                this, a negative result on real data says nothing -- the
                instrument is broken, not the hypothesis.
  PURE NOISE    a random-walk panel with nothing to find. If the model reports
                skill here, it is fitting noise, and its real-data score is
                worth exactly as much.

ULTIMATE established this discipline: 6,036 configurations found zero robust
regions, and those negatives are credible only because the same engine found a
planted edge 20 times and rejected 243 noise configurations. A network gets held
to the same standard or it gets no standing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Forward horizon in sessions. Matches the cross-sectional rebalance, so a
#: prediction is answerable by the venue that would act on it.
HORIZON = 21

#: Feature lookbacks in sessions. Deliberately few and conventional -- a wide
#: feature search on one panel is the search that produced zero robust regions.
MOMENTUM_WINDOWS = (21, 63, 126, 252)
VOL_WINDOW = 63
SKIP = 21                      # the 12-1 convention: skip the reversal month


@dataclass
class FoldResult:
    fold: int
    train_end: str
    test_start: str
    test_end: str
    n_train: int
    n_test: int
    ic: float                  # rank correlation of prediction vs realised
    top_decile_ret: float
    bottom_decile_ret: float
    spread: float


@dataclass
class ModelReport:
    label: str
    folds: list[FoldResult] = field(default_factory=list)
    holdout: FoldResult | None = None

    @property
    def mean_ic(self) -> float:
        return float(np.mean([f.ic for f in self.folds])) if self.folds else float("nan")

    @property
    def ic_t(self) -> float:
        """t-statistic of the fold ICs against zero.

        Reported rather than a single pooled IC because folds are the independent
        unit here: a pooled correlation over overlapping windows overstates its
        own significance badly.
        """
        ics = [f.ic for f in self.folds]
        if len(ics) < 2:
            return float("nan")
        return float(np.mean(ics) / (np.std(ics, ddof=1) / np.sqrt(len(ics))))

    def render(self) -> str:
        lines = [f"{self.label}",
                 f"  folds {len(self.folds)} | mean IC {self.mean_ic:+.4f} "
                 f"| IC t-stat {self.ic_t:+.2f}"]
        for f in self.folds:
            lines.append(f"    [{f.fold}] {f.test_start}..{f.test_end}  "
                         f"IC {f.ic:+.4f}  decile spread {f.spread:+.2%}  "
                         f"n_train {f.n_train:,}")
        if self.holdout:
            h = self.holdout
            lines.append(f"  HOLDOUT {h.test_start}..{h.test_end}  "
                         f"IC {h.ic:+.4f}  decile spread {h.spread:+.2%}")
        return "\n".join(lines)


# ------------------------------------------------------------------ features

def build_features(close: pd.DataFrame, volume: pd.DataFrame | None = None,
                   leak: bool = False) -> tuple[pd.DataFrame, pd.Series]:
    """Cross-sectional features and the forward return they must predict.

    Returns a long frame indexed by (date, symbol). Features at t use data up to
    and including t; the label is the return from t to t+HORIZON.

    `leak` shifts the label backwards so it overlaps the feature window. It exists
    only so a test can assert that a leak would show up as an implausible score --
    a guard against the guard being wrong.
    """
    rets = close.pct_change()
    feats = {}

    for w in MOMENTUM_WINDOWS:
        # Skip the most recent month on the longest window only: that is the
        # 12-1 convention, and applying it everywhere would erase the short-term
        # reversal the shorter windows are there to capture.
        if w >= 252:
            feats[f"mom_{w}"] = close.shift(SKIP) / close.shift(w) - 1.0
        else:
            feats[f"mom_{w}"] = close / close.shift(w) - 1.0

    feats["vol_63"] = rets.rolling(VOL_WINDOW).std()
    feats["rev_5"] = close / close.shift(5) - 1.0
    if volume is not None:
        turn = (volume * close).rolling(21).mean()
        feats["turnover_log"] = np.log1p(turn)

    label = close.shift(-HORIZON) / close - 1.0
    if leak:
        label = close.shift(HORIZON) / close - 1.0      # deliberately backwards

    long = pd.concat({k: v.stack() for k, v in feats.items()}, axis=1)
    long["y"] = label.stack()
    long = long.replace([np.inf, -np.inf], np.nan).dropna()
    long.index.names = ["date", "symbol"]

    # Rank-normalise features WITHIN each date. A network fed raw momentum learns
    # the market's level, not the cross-section; the venue trades relative rank,
    # so that is what it should see.
    x = long.drop(columns="y")
    x = x.groupby(level="date").rank(pct=True) - 0.5
    return x, long["y"]


# --------------------------------------------------------------------- model

def _fit_predict(x_tr, y_tr, x_te, seed: int, hidden=(32, 16), max_iter: int = 60):
    """Small MLP. Deliberately small: capacity is the enemy here.

    Standardisation is fitted on training rows only -- see the module docstring.
    """
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=hidden, activation="relu",
                     alpha=1e-3, learning_rate_init=1e-3, max_iter=max_iter,
                     early_stopping=True, n_iter_no_change=5, random_state=seed),
    )
    model.fit(x_tr, y_tr)
    return model.predict(x_te)


def _score(pred: np.ndarray, actual: np.ndarray, dates: np.ndarray) -> tuple[float, float, float]:
    """Rank IC and decile spread, computed per date then averaged.

    Per date, not pooled: pooling mixes the cross-section with the time series and
    lets a few extreme days dominate a number that is supposed to describe rank
    skill within a day.
    """
    df = pd.DataFrame({"p": pred, "a": actual, "d": dates})
    ics, tops, bots = [], [], []
    for _, g in df.groupby("d"):
        if len(g) < 20:
            continue
        ics.append(g["p"].corr(g["a"], method="spearman"))
        k = max(1, len(g) // 10)
        order = g.sort_values("p")
        bots.append(order["a"].head(k).mean())
        tops.append(order["a"].tail(k).mean())
    if not ics:
        return float("nan"), float("nan"), float("nan")
    return (float(np.nanmean(ics)), float(np.nanmean(tops)), float(np.nanmean(bots)))


class InsufficientHistory(RuntimeError):
    """The panel is too short to build even one honest fold."""


def walk_forward(x: pd.DataFrame, y: pd.Series, label: str,
                 n_folds: int = 5, holdout_frac: float = 0.2,
                 seed: int = 0, min_train_dates: int = 100,
                 min_train_rows: int = 500, min_test_rows: int = 100) -> ModelReport:
    """Expanding-window walk-forward, then one scored holdout.

    The holdout is carved off the END of the panel and never seen during folds.
    Everything before it is split into expanding train / forward test pairs.

    RAISES RATHER THAN RETURNING NOTHING. The minimum sizes below exist so a fold
    is never fitted on too little to mean anything -- but silently skipping every
    fold and returning a report whose mean IC is NaN is worse than the problem it
    avoids. A caller reading `mean_ic` gets a number-shaped absence and no reason
    for it, which is precisely how a broken run gets mistaken for a null result.
    The minimums are parameters so a caller with a shorter panel can lower them
    deliberately, in the open, rather than discovering the floor by getting NaN.
    """
    dates = np.array(sorted(x.index.get_level_values("date").unique()))
    n_hold = max(1, int(len(dates) * holdout_frac))
    dev_dates, hold_dates = dates[:-n_hold], dates[-n_hold:]

    rep = ModelReport(label=label)
    # Purge HORIZON sessions between train and test: the last training label
    # overlaps the first test window otherwise, which is a leak the fold boundary
    # alone does not prevent.
    bounds = np.linspace(len(dev_dates) * 0.4, len(dev_dates), n_folds + 1).astype(int)

    for i in range(n_folds):
        tr_end_i, te_end_i = bounds[i], bounds[i + 1]
        if te_end_i - tr_end_i < 5:
            continue
        tr_dates = dev_dates[:max(0, tr_end_i - HORIZON)]
        te_dates = dev_dates[tr_end_i:te_end_i]
        if len(tr_dates) < min_train_dates or len(te_dates) < 5:
            continue

        tr = x.index.get_level_values("date").isin(tr_dates)
        te = x.index.get_level_values("date").isin(te_dates)
        if tr.sum() < min_train_rows or te.sum() < min_test_rows:
            continue

        pred = _fit_predict(x[tr].to_numpy(), y[tr].to_numpy(),
                            x[te].to_numpy(), seed=seed + i)
        ic, top, bot = _score(pred, y[te].to_numpy(),
                              x[te].index.get_level_values("date").to_numpy())
        rep.folds.append(FoldResult(
            fold=i + 1, train_end=str(pd.Timestamp(tr_dates[-1]).date()),
            test_start=str(pd.Timestamp(te_dates[0]).date()),
            test_end=str(pd.Timestamp(te_dates[-1]).date()),
            n_train=int(tr.sum()), n_test=int(te.sum()),
            ic=ic, top_decile_ret=top, bottom_decile_ret=bot, spread=top - bot))

    if not rep.folds:
        raise InsufficientHistory(
            f"{label}: no fold met the minimums ({len(dev_dates)} development "
            f"dates, {len(x):,} rows; need {min_train_dates} train dates, "
            f"{min_train_rows} train rows, {min_test_rows} test rows per fold). "
            f"Lower the minimums explicitly or supply a longer panel.")

    # The holdout, scored once.
    tr_dates = dev_dates[:max(0, len(dev_dates) - HORIZON)]
    tr = x.index.get_level_values("date").isin(tr_dates)
    ho = x.index.get_level_values("date").isin(hold_dates)
    if tr.sum() >= min_train_rows and ho.sum() >= min_test_rows:
        pred = _fit_predict(x[tr].to_numpy(), y[tr].to_numpy(),
                            x[ho].to_numpy(), seed=seed + 99)
        ic, top, bot = _score(pred, y[ho].to_numpy(),
                              x[ho].index.get_level_values("date").to_numpy())
        rep.holdout = FoldResult(
            fold=0, train_end=str(pd.Timestamp(tr_dates[-1]).date()),
            test_start=str(pd.Timestamp(hold_dates[0]).date()),
            test_end=str(pd.Timestamp(hold_dates[-1]).date()),
            n_train=int(tr.sum()), n_test=int(ho.sum()),
            ic=ic, top_decile_ret=top, bottom_decile_ret=bot, spread=top - bot)
    return rep
