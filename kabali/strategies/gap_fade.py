"""Gap fade -- trade the failure of an opening gap to follow through.

A gap is the one intraday setup whose premise is set before the session starts,
and it splits cleanly in two: gaps that keep going, and gaps that fill. This rule
takes only the second, and only after the failure is visible.

The evidence it waits for is the opening drive stalling. A gap up that is going
to run makes its low in the first minutes and never looks back; a gap up that is
going to fill rolls over while the session high stands unchallenged. So the rule
requires price to have traded back through the opening bar's own extreme -- not
merely to have paused -- before it will fade.

The target is the prior close, because that is where the gap is by definition
filled, and it is a level the market itself is aware of. The stop sits beyond the
session extreme made during the drive: if the gap resumes past its own high, the
failure thesis is wrong and there is nothing left to fade.

Declared valid in the range family only. In a strong trend, gaps run, and fading
them is the losing side of the most persistent intraday pattern there is.
"""
from __future__ import annotations

import numpy as np

from kabali.strategies.base import LONG, SHORT, IntradayStrategy, Signal, SymbolContext
from kabali.strategies.common import clamp_stop, intraday_atr, session_extremes, stop_is_sane


class GapFade(IntradayStrategy):
    families = ("range",)
    warmup_bars = 4

    def __init__(self, min_gap_atr: float = 0.75, max_gap_atr: float = 3.0,
                 stop_atr: float = 0.8, min_fill_fraction: float = 0.25):
        self.min_gap_atr = min_gap_atr
        self.max_gap_atr = max_gap_atr
        self.stop_atr = stop_atr
        self.min_fill_fraction = min_fill_fraction

    def generate(self, ctx: SymbolContext) -> list[Signal]:
        if not (np.isfinite(ctx.prev_close) and np.isfinite(ctx.atr_daily)) or ctx.atr_daily <= 0:
            return []
        d = ctx.intraday
        if not len(d):
            return []

        session_open = float(d["open"].iloc[0])
        gap = session_open - ctx.prev_close
        gap_atr = abs(gap) / ctx.atr_daily
        # Too small to be a gap; too large and it is usually news, where the
        # prior close has stopped being a meaningful reference at all.
        if not (self.min_gap_atr <= gap_atr <= self.max_gap_atr):
            return []

        a = intraday_atr(ctx)
        if not np.isfinite(a) or a <= 0:
            return []

        first = d.iloc[0]
        first_high, first_low = float(first["high"]), float(first["low"])
        price = ctx.price
        sess_high, sess_low = session_extremes(ctx)
        total_gap = abs(gap)
        out: list[Signal] = []

        # Gap UP that failed: price has broken back below the opening bar's low,
        # and has already retraced a meaningful part of the gap. Fade it short.
        if gap > 0 and getattr(ctx.regime, "allows_short", True):
            filled = (session_open - price) / total_gap
            if price < first_low and filled >= self.min_fill_fraction:
                entry = ctx.round_tick(price)
                stop = ctx.round_tick(clamp_stop(entry, max(sess_high, entry + self.stop_atr * a),
                                                 SHORT))
                target = ctx.round_tick(ctx.prev_close)
                if stop_is_sane(entry, stop) and target < entry:
                    out.append(Signal(
                        symbol=ctx.symbol, side=SHORT, entry=entry, stop=stop, target=target,
                        strategy=self.name, at=ctx.now,
                        confidence=self._confidence(gap_atr, filled),
                        reasons=(f"gapped up {gap_atr:.1f} daily ATR",
                                 f"broke back below the opening bar low {first_low:.2f}",
                                 f"{filled:.0%} of the gap already retraced",
                                 f"target is prior close {ctx.prev_close:.2f}"),
                    ))

        if gap < 0 and getattr(ctx.regime, "allows_long", True):
            filled = (price - session_open) / total_gap
            if price > first_high and filled >= self.min_fill_fraction:
                entry = ctx.round_tick(price)
                stop = ctx.round_tick(clamp_stop(entry, min(sess_low, entry - self.stop_atr * a),
                                                 LONG))
                target = ctx.round_tick(ctx.prev_close)
                if stop_is_sane(entry, stop) and target > entry:
                    out.append(Signal(
                        symbol=ctx.symbol, side=LONG, entry=entry, stop=stop, target=target,
                        strategy=self.name, at=ctx.now,
                        confidence=self._confidence(gap_atr, filled),
                        reasons=(f"gapped down {gap_atr:.1f} daily ATR",
                                 f"broke back above the opening bar high {first_high:.2f}",
                                 f"{filled:.0%} of the gap already retraced",
                                 f"target is prior close {ctx.prev_close:.2f}"),
                    ))
        return out

    def _confidence(self, gap_atr: float, filled: float) -> float:
        # A modest gap that is already filling is the reliable case; a huge gap
        # barely off its extreme is the one that resumes.
        base = 0.45 + 0.30 * min(filled, 1.0) - 0.10 * max(gap_atr - 1.5, 0.0)
        return float(np.clip(base, 0.05, 0.90))
