"""Opening range breakout.

The oldest intraday rule there is, and the one most often implemented in the
form that loses money. Three guards separate this version from the naive one.

VOLUME CONFIRMATION. A break on below-median volume is a level being brushed,
not taken. Requiring a volume multiple is what stops the rule from buying the
high of the day on a drift-up through the opening range.

GAP EXCLUSION. When the session opens far outside yesterday's range, the opening
range forms around an already-repriced level and the breakout carries none of the
usual information. Those are the days a gap rule should handle, not this one.

REGIME GATE. Declared valid only in the trend family. In chop, an opening range
gets broken in both directions before lunch, and every break is a loss. ULTIMATE
measured this: its intraday venue was negative in every regime bucket with no
regime gate at all.

The stop sits at the opposite side of the opening range, which is the level that
falsifies the idea -- if price returns through the whole range, the breakout was
noise. That is a real invalidation point rather than an arbitrary ATR multiple.
"""
from __future__ import annotations

import numpy as np

from kabali.strategies.base import LONG, SHORT, IntradayStrategy, Signal, SymbolContext
from kabali.strategies.common import (
    clamp_stop, intraday_atr, relative_volume, session_extremes, stop_is_sane,
)


class OpeningRangeBreakout(IntradayStrategy):
    families = ("trend",)
    warmup_bars = 4

    def __init__(self, min_rel_volume: float = 1.3, target_r: float = 2.0,
                 max_gap_atr: float = 1.5, buffer_atr: float = 0.10):
        self.min_rel_volume = min_rel_volume
        self.target_r = target_r
        self.max_gap_atr = max_gap_atr
        self.buffer_atr = buffer_atr

    def generate(self, ctx: SymbolContext) -> list[Signal]:
        rng = ctx.opening_range(ctx.session.opening_range_end)
        if rng is None:
            return []
        or_high, or_low = rng
        if not (np.isfinite(or_high) and np.isfinite(or_low)) or or_high <= or_low:
            return []

        # Only breaks that happen AFTER the range is complete count.
        if ctx.now.time() <= ctx.session.opening_range_end:
            return []

        a = intraday_atr(ctx)
        if not np.isfinite(a) or a <= 0:
            return []

        # A session that opened miles from yesterday is a gap setup, not this one.
        if np.isfinite(ctx.prev_close) and np.isfinite(ctx.atr_daily) and ctx.atr_daily > 0:
            open_px = float(ctx.intraday["open"].iloc[0])
            if abs(open_px - ctx.prev_close) > self.max_gap_atr * ctx.atr_daily:
                return []

        rel_vol = relative_volume(ctx)
        if not np.isfinite(rel_vol) or rel_vol < self.min_rel_volume:
            return []

        price = ctx.price
        buf = self.buffer_atr * a
        high, low = session_extremes(ctx)
        out: list[Signal] = []
        direction = getattr(ctx.regime, "direction", 0)

        # Long: closed above the range high by a buffer, and the regime permits it.
        if price > or_high + buf and getattr(ctx.regime, "allows_long", True):
            entry = ctx.round_tick(price)
            stop = ctx.round_tick(clamp_stop(entry, min(or_low, low), LONG))
            if stop_is_sane(entry, stop):
                target = ctx.round_tick(entry + self.target_r * (entry - stop))
                out.append(Signal(
                    symbol=ctx.symbol, side=LONG, entry=entry, stop=stop, target=target,
                    strategy=self.name, at=ctx.now,
                    confidence=self._confidence(rel_vol, direction, +1),
                    reasons=(f"close {price:.2f} above OR high {or_high:.2f}",
                             f"relative volume {rel_vol:.2f}x",
                             f"stop at OR low {stop:.2f}"),
                ))

        if price < or_low - buf and getattr(ctx.regime, "allows_short", True):
            entry = ctx.round_tick(price)
            stop = ctx.round_tick(clamp_stop(entry, max(or_high, high), SHORT))
            if stop_is_sane(entry, stop):
                target = ctx.round_tick(entry - self.target_r * (stop - entry))
                out.append(Signal(
                    symbol=ctx.symbol, side=SHORT, entry=entry, stop=stop, target=target,
                    strategy=self.name, at=ctx.now,
                    confidence=self._confidence(rel_vol, direction, -1),
                    reasons=(f"close {price:.2f} below OR low {or_low:.2f}",
                             f"relative volume {rel_vol:.2f}x",
                             f"stop at OR high {stop:.2f}"),
                ))
        return out

    def _confidence(self, rel_vol: float, regime_direction: int, side: int) -> float:
        """Volume strength, raised when the break agrees with the market."""
        base = min(0.35 + 0.15 * (rel_vol - self.min_rel_volume), 0.75)
        if regime_direction == side:
            base += 0.15
        elif regime_direction == -side:
            base -= 0.10
        return float(np.clip(base, 0.05, 0.95))
