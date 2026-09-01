"""Momentum burst -- range expansion on a volume surge at a session extreme.

This is the rule that hunts the thing the universe screen selected for: a name
with intraday range and clean travel, at the moment it starts moving. Where the
opening-range rule waits for one specific level, this one fires on the signature
of institutional participation arriving at any point in the session -- a bar that
is both much larger than recent bars AND much heavier than recent bars, printing
at a session extreme.

Requiring range expansion and volume expansion TOGETHER is the whole point.
Volume alone shows up on absorption, where heavy trade produces no movement
because someone is filling into the move. Range alone shows up on thin drifts
through an empty book, which reverse the moment real size appears. The
conjunction is much rarer and much more informative than either.

Valid in trend AND volatile families. Unlike the other rules this one does not
need a calm market -- a burst is a burst -- but in the volatile family the
confidence is discounted, and the risk engine's own volatility scaling shrinks
the position on top of that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from kabali.strategies.base import LONG, SHORT, IntradayStrategy, Signal, SymbolContext
from kabali.strategies.common import (
    clamp_stop, intraday_atr, relative_volume, session_extremes, stop_is_sane,
)


class MomentumBurst(IntradayStrategy):
    families = ("trend", "volatile")
    warmup_bars = 6

    def __init__(self, range_mult: float = 1.8, volume_mult: float = 2.0,
                 stop_atr: float = 1.0, target_r: float = 2.0,
                 extreme_tolerance: float = 0.15):
        self.range_mult = range_mult
        self.volume_mult = volume_mult
        self.stop_atr = stop_atr
        self.target_r = target_r
        self.extreme_tolerance = extreme_tolerance

    def generate(self, ctx: SymbolContext) -> list[Signal]:
        a = intraday_atr(ctx)
        if not np.isfinite(a) or a <= 0:
            return []

        d = ctx.intraday
        last = ctx.last
        hi, lo = float(last["high"]), float(last["low"])
        price = float(last["close"])
        bar_open = float(last["open"])
        bar_range = hi - lo
        if bar_range <= 0:
            return []

        # Range expansion against the recent norm, excluding the latest bar.
        highs = pd.to_numeric(d["high"], errors="coerce")
        lows = pd.to_numeric(d["low"], errors="coerce")
        prior_ranges = (highs - lows).iloc[:-1].tail(20)
        med_range = float(prior_ranges.median()) if len(prior_ranges) else float("nan")
        if not np.isfinite(med_range) or med_range <= 0:
            return []
        range_ratio = bar_range / med_range
        if range_ratio < self.range_mult:
            return []

        rel_vol = relative_volume(ctx)
        if not np.isfinite(rel_vol) or rel_vol < self.volume_mult:
            return []

        sess_high, sess_low = session_extremes(ctx)
        span = max(sess_high - sess_low, 1e-9)
        volatile = getattr(ctx.regime, "family", "") == "volatile"
        out: list[Signal] = []

        # Long: expansion bar closed up, near the session high.
        near_high = (sess_high - price) / span <= self.extreme_tolerance
        if (price > bar_open and near_high and getattr(ctx.regime, "allows_long", True)):
            entry = ctx.round_tick(price)
            stop = ctx.round_tick(clamp_stop(entry, min(lo, entry - self.stop_atr * a), LONG))
            if stop_is_sane(entry, stop):
                target = ctx.round_tick(entry + self.target_r * (entry - stop))
                out.append(Signal(
                    symbol=ctx.symbol, side=LONG, entry=entry, stop=stop, target=target,
                    strategy=self.name, at=ctx.now,
                    confidence=self._confidence(range_ratio, rel_vol, volatile),
                    reasons=(f"bar range {range_ratio:.1f}x the 20-bar median",
                             f"volume {rel_vol:.1f}x the 20-bar median",
                             "closed up at the session high"),
                ))

        near_low = (price - sess_low) / span <= self.extreme_tolerance
        if (price < bar_open and near_low and getattr(ctx.regime, "allows_short", True)):
            entry = ctx.round_tick(price)
            stop = ctx.round_tick(clamp_stop(entry, max(hi, entry + self.stop_atr * a), SHORT))
            if stop_is_sane(entry, stop):
                target = ctx.round_tick(entry - self.target_r * (stop - entry))
                out.append(Signal(
                    symbol=ctx.symbol, side=SHORT, entry=entry, stop=stop, target=target,
                    strategy=self.name, at=ctx.now,
                    confidence=self._confidence(range_ratio, rel_vol, volatile),
                    reasons=(f"bar range {range_ratio:.1f}x the 20-bar median",
                             f"volume {rel_vol:.1f}x the 20-bar median",
                             "closed down at the session low"),
                ))
        return out

    def _confidence(self, range_ratio: float, rel_vol: float, volatile: bool) -> float:
        base = 0.30 + 0.10 * (range_ratio - self.range_mult) + 0.08 * (rel_vol - self.volume_mult)
        if volatile:
            base -= 0.15      # bursts are frequent and less informative in panic
        return float(np.clip(base, 0.05, 0.90))
