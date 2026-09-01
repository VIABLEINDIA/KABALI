"""Swing entry and exit rules, evaluated across the whole cross-section.

The interface is deliberately not `IntradayStrategy`. An intraday rule answers
"is there a setup in this symbol right now"; a cross-sectional rule answers "of
everything investable today, which are the best five", and a per-symbol
`generate()` cannot express that -- the answer for one name depends on every
other name's score on the same date.

`ClenowMomentum` implements Clenow, *Stocks on the Move* (2015), unchanged. Its
parameters live in `config/bot.yaml` under `swing.rules` and its full statement,
including what would falsify it, is in `docs/swing_hypothesis.md`, written
before the first run.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np
import pandas as pd

from kabali.swing import panel as P


@dataclass(frozen=True)
class Ranking:
    """The cross-section on one date, already ordered."""

    date: pd.Timestamp
    scores: pd.Series            # symbol -> score, investable names only, desc
    entries: tuple[str, ...]     # subset that additionally passes the entry filters
    market_ok: bool              # index filter: may we open anything at all

    def rank_of(self, symbol: str) -> int | None:
        """1-based position in the investable ranking, or None if unranked."""
        if symbol not in self.scores.index:
            return None
        return int(self.scores.index.get_loc(symbol)) + 1

    @property
    def size(self) -> int:
        return len(self.scores)


class SwingStrategy(abc.ABC):
    """A cross-sectional rule over a prepared panel."""

    @property
    def name(self) -> str:
        return type(self).__name__

    @abc.abstractmethod
    def rank(self, date: pd.Timestamp) -> Ranking:
        """The investable cross-section as of `date`'s close."""

    @abc.abstractmethod
    def initial_stop(self, symbol: str, date: pd.Timestamp, entry: float) -> float:
        """Stop price for a position entered against `date`'s close."""

    @abc.abstractmethod
    def exit_reason(self, symbol: str, date: pd.Timestamp,
                    ranking: Ranking) -> str | None:
        """Why this holding should be closed, or None to keep it."""

    def __repr__(self) -> str:
        return f"<{self.name}>"


class ClenowMomentum(SwingStrategy):
    """Rank on 90-day exponential-regression slope x R^2; hold the top names.

    Four filters, each answering a different way the raw ranking misleads:

    TREND (close above the 100d SMA) -- a stock can hold a high 90-day slope
    while already rolling over. The slope is a statement about the window that
    just ended, not about today.

    GAP (no move over 15% in 90 days) -- a single overnight jump produces a
    large slope with a good R^2 and is precisely the move that has already
    happened. It is also the one an ATR-scaled stop mis-sizes, because ATR
    measured before the gap understates what the name now does.

    INDEX (NIFTY above its 200d SMA) -- the cross-sectional momentum premium is
    concentrated in rising markets and reverses hard in falling ones. This filter
    is what stops the system buying the strongest names in a bear market.

    LIQUIDITY / PRICE / HISTORY -- the standing floors from `universe:`. A name
    that cannot absorb the order is not investable at any rank.

    The trend and gap filters gate ENTRY and force exit. The index filter gates
    entry only: an open position keeps being managed on its own merits, because
    closing a working position on an index reading is a second, untested bet.
    """

    def __init__(self, panel: P.Panel, index_close: pd.Series, cfg):
        self.panel = panel
        self.cfg = cfg
        rules = cfg.swing.rules
        self.rules = rules

        self.score = P.rolling_slope_r2(panel.close, rules.rank_window)
        self.ma_trend = P.sma(panel.close, rules.trend_ma)
        self.atr = P.atr(panel.high, panel.low, panel.close, rules.atr_window)
        self.turnover = P.rolling_median_turnover(panel.close, panel.volume,
                                                  rules.turnover_window)
        self.gap = P.max_abs_gap(panel.open, panel.close, rules.gap_window)

        u = cfg.universe
        history = panel.close.notna().cumsum()
        # Investable: tradeable at our size, with enough history to be ranked.
        # Note this is NOT filtered by trend or gap -- it is the denominator for
        # the "top 20%" exit test, and a denominator that moved with the filters
        # would make the exit threshold drift with market breadth.
        self.investable = (
            self.score.notna()
            & (history >= rules.min_history_bars)
            & (panel.close >= u.min_price)
            & (panel.close <= u.max_price)
            & (self.turnover >= u.min_median_turnover)
            & self.atr.notna()
            & (self.atr > 0)
        )
        self.entry_ok = (
            self.investable
            & (panel.close > self.ma_trend)
            & (self.gap <= rules.max_gap_pct / 100.0)
        )

        index_close = pd.Series(index_close).astype(float).sort_index()
        index_ma = index_close.rolling(rules.index_ma, min_periods=rules.index_ma).mean()
        self.market_filter = (index_close > index_ma).reindex(panel.dates)
        # A date the index has no bar for is not tradeable: without the filter
        # reading we cannot know whether entries are permitted, and assuming
        # permission is the direction that costs money.
        self.market_filter = self.market_filter.fillna(False).astype(bool)

    # ------------------------------------------------------------------ ranking
    def rank(self, date: pd.Timestamp) -> Ranking:
        date = pd.Timestamp(date)
        if date not in self.panel.dates:
            return Ranking(date, pd.Series(dtype=float), (), False)

        inv = self.investable.loc[date]
        scores = self.score.loc[date][inv].sort_values(ascending=False)
        entries = tuple(s for s in scores.index if bool(self.entry_ok.loc[date, s]))
        return Ranking(date=date, scores=scores, entries=entries,
                       market_ok=bool(self.market_filter.loc[date]))

    # -------------------------------------------------------------------- stops
    def initial_stop(self, symbol: str, date: pd.Timestamp, entry: float) -> float:
        """Entry minus `stop_atr_mult` ATRs, the same unit the size is set by.

        Returning NaN when ATR is unknown is deliberate: the sizer refuses a
        signal it cannot measure, and a defaulted stop distance would size a
        position on a number nobody chose.
        """
        a = self._atr_at(symbol, date)
        if not np.isfinite(a) or a <= 0:
            return float("nan")
        return float(entry - self.rules.stop_atr_mult * a)

    def trail_stop(self, symbol: str, date: pd.Timestamp, close: float,
                   current_stop: float) -> float:
        """Ratchet the stop up behind the close. It never moves down.

        A stop that could loosen is not a stop; it is a way of turning a small
        realised loss into a larger one while believing the risk is unchanged.
        """
        a = self._atr_at(symbol, date)
        if not np.isfinite(a) or a <= 0:
            return current_stop
        candidate = close - self.rules.stop_atr_mult * a
        return float(max(current_stop, candidate))

    def _atr_at(self, symbol: str, date: pd.Timestamp) -> float:
        try:
            return float(self.atr.at[pd.Timestamp(date), symbol])
        except KeyError:
            return float("nan")

    # -------------------------------------------------------------------- exits
    def exit_reason(self, symbol: str, date: pd.Timestamp,
                    ranking: Ranking) -> str | None:
        date = pd.Timestamp(date)
        if symbol not in self.panel.close.columns:
            return "delisted"

        close = self.panel.close.at[date, symbol] if date in self.panel.dates else np.nan
        if not np.isfinite(close):
            return None            # no bar today: hold, do not guess a price

        if not bool(self.investable.at[date, symbol]):
            return "not_investable"
        ma = self.ma_trend.at[date, symbol]
        if np.isfinite(ma) and close < ma:
            return "below_trend_ma"
        gap = self.gap.at[date, symbol]
        if np.isfinite(gap) and gap > self.rules.max_gap_pct / 100.0:
            return "gapped"

        rank = ranking.rank_of(symbol)
        cutoff = max(1, int(np.ceil(ranking.size * self.rules.exit_rank_pct / 100.0)))
        if rank is None or rank > cutoff:
            return "rank_dropped"
        return None
