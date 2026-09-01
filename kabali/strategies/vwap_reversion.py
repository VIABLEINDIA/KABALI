"""VWAP mean reversion -- fade extension away from the session's fair value.

Session VWAP is where the day's volume actually changed hands, which makes it a
better reversion anchor intraday than a moving average: it is weighted by where
participation was, not by where the clock was.

The rule fades price that has stretched several intraday ATRs from VWAP. It is
declared valid only in the RANGE family, and that gate is doing the real work --
in a trend, "extended from VWAP" is the normal state of a stock going up all day,
and fading it is how a mean-reversion rule hands back a month of gains in a week.

Two additional guards:

EXHAUSTION, NOT MOMENTUM. The extension must be stalling -- the latest bar has to
close back toward VWAP relative to its own extreme. Fading a bar that is still
printing new extremes is catching a knife.

NO LATE FADES. Reversion needs time to complete. A fade entered near the entry
cutoff gets closed by the square-off rule at whatever price exists at 15:10,
which converts a mean-reversion edge into a coin flip on the close.
"""
from __future__ import annotations

import numpy as np

from kabali.strategies.base import LONG, SHORT, IntradayStrategy, Signal, SymbolContext
from kabali.strategies.common import clamp_stop, intraday_atr, stop_is_sane


class VWAPReversion(IntradayStrategy):
    families = ("range",)
    warmup_bars = 6

    def __init__(self, entry_atr: float = 2.0, stop_atr: float = 1.2,
                 min_reject_frac: float = 0.30, max_minutes_from_cutoff: int = 45):
        self.entry_atr = entry_atr
        self.stop_atr = stop_atr
        self.min_reject_frac = min_reject_frac
        self.max_minutes_from_cutoff = max_minutes_from_cutoff

    def generate(self, ctx: SymbolContext) -> list[Signal]:
        vwap = ctx.session_vwap()
        a = intraday_atr(ctx)
        if not (np.isfinite(vwap) and np.isfinite(a)) or a <= 0:
            return []

        last = ctx.last
        if last is None:
            return []
        price = float(last["close"])
        hi, lo = float(last["high"]), float(last["low"])
        bar_range = hi - lo
        if bar_range <= 0:
            return []

        extension = (price - vwap) / a
        out: list[Signal] = []

        # Long the downside extension: price stretched below VWAP and the bar
        # closed off its low, so sellers stopped getting filled lower.
        if extension <= -self.entry_atr and getattr(ctx.regime, "allows_long", True):
            reject = (price - lo) / bar_range
            if reject >= self.min_reject_frac:
                entry = ctx.round_tick(price)
                stop = ctx.round_tick(clamp_stop(entry, min(lo, entry - self.stop_atr * a), LONG))
                target = ctx.round_tick(vwap)
                if stop_is_sane(entry, stop) and target > entry:
                    out.append(Signal(
                        symbol=ctx.symbol, side=LONG, entry=entry, stop=stop, target=target,
                        strategy=self.name, at=ctx.now,
                        confidence=self._confidence(abs(extension), reject),
                        reasons=(f"{abs(extension):.1f} ATR below VWAP {vwap:.2f}",
                                 f"bar closed {reject:.0%} off its low",
                                 "target is VWAP"),
                    ))

        if extension >= self.entry_atr and getattr(ctx.regime, "allows_short", True):
            reject = (hi - price) / bar_range
            if reject >= self.min_reject_frac:
                entry = ctx.round_tick(price)
                stop = ctx.round_tick(clamp_stop(entry, max(hi, entry + self.stop_atr * a), SHORT))
                target = ctx.round_tick(vwap)
                if stop_is_sane(entry, stop) and target < entry:
                    out.append(Signal(
                        symbol=ctx.symbol, side=SHORT, entry=entry, stop=stop, target=target,
                        strategy=self.name, at=ctx.now,
                        confidence=self._confidence(abs(extension), reject),
                        reasons=(f"{extension:.1f} ATR above VWAP {vwap:.2f}",
                                 f"bar closed {reject:.0%} off its high",
                                 "target is VWAP"),
                    ))
        return out

    def ready(self, ctx: SymbolContext) -> bool:
        if ctx.bars < self.warmup_bars:
            return False

        # Reversion needs room to complete, and the deadline that matters is the
        # SQUARE-OFF, not the entry cutoff. A fade opened at 14:40 has half an
        # hour to reach VWAP before 15:10 closes it at whatever price exists --
        # which converts a mean-reversion edge into a coin flip on the close.
        # The engine already blocks entries past the cutoff; this is the extra
        # margin the strategy needs on top of that.
        square_off = ctx.session.square_off
        now_min = ctx.now.hour * 60 + ctx.now.minute
        deadline = square_off.hour * 60 + square_off.minute - self.max_minutes_from_cutoff
        return now_min <= deadline

    def _confidence(self, extension: float, reject: float) -> float:
        base = 0.30 + 0.12 * (extension - self.entry_atr) + 0.30 * reject
        return float(np.clip(base, 0.05, 0.90))
