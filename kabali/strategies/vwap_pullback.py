"""VWAP trend pullback -- join an established intraday trend at fair value.

The complement to `VWAPReversion`, and the reason both can coexist: they read the
same distance from VWAP and take opposite actions, so the regime gate is what
decides which question is being asked. Reversion runs in chop and fades stretch;
this runs in a trend and buys the return to value.

The setup wants three things in sequence, which is why it needs the session's
shape rather than only the latest bar:

  1. TREND ESTABLISHED -- price has held one side of VWAP for most of the session
  2. PULLBACK -- price has come back to within a fraction of an ATR of VWAP
  3. RESUMPTION -- the latest confirmed bar turned back in the trend direction

Requiring resumption rather than entering on touch is what keeps this out of the
trade where VWAP simply fails and the trend reverses. The stop goes just beyond
the pullback extreme: if price closes through the level the pullback bottomed at,
the pullback was a reversal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from kabali.strategies.base import LONG, SHORT, IntradayStrategy, Signal, SymbolContext
from kabali.strategies.common import clamp_stop, intraday_atr, stop_is_sane


class VWAPPullback(IntradayStrategy):
    families = ("trend",)
    warmup_bars = 8

    def __init__(self, min_side_frac: float = 0.65, touch_atr: float = 0.5,
                 stop_atr: float = 0.8, target_r: float = 2.0):
        self.min_side_frac = min_side_frac
        self.touch_atr = touch_atr
        self.stop_atr = stop_atr
        self.target_r = target_r

    def generate(self, ctx: SymbolContext) -> list[Signal]:
        vwap = ctx.session_vwap()
        a = intraday_atr(ctx)
        if not (np.isfinite(vwap) and np.isfinite(a)) or a <= 0:
            return []

        d = ctx.intraday
        close = pd.to_numeric(d["close"], errors="coerce")
        above_frac = float((close > vwap).mean())

        last = ctx.last
        price = float(last["close"])
        bar_open = float(last["open"])
        distance = abs(price - vwap) / a

        # The pullback leg: the lowest low (or highest high) of the recent bars
        # is the level whose breach falsifies the trend.
        recent = d.tail(4)
        leg_low = float(pd.to_numeric(recent["low"], errors="coerce").min())
        leg_high = float(pd.to_numeric(recent["high"], errors="coerce").max())

        out: list[Signal] = []

        up_trend = above_frac >= self.min_side_frac
        down_trend = (1.0 - above_frac) >= self.min_side_frac

        if (up_trend and distance <= self.touch_atr and price > bar_open
                and getattr(ctx.regime, "allows_long", True)):
            entry = ctx.round_tick(price)
            stop = ctx.round_tick(clamp_stop(entry, min(leg_low, entry - self.stop_atr * a), LONG))
            if stop_is_sane(entry, stop):
                target = ctx.round_tick(entry + self.target_r * (entry - stop))
                out.append(Signal(
                    symbol=ctx.symbol, side=LONG, entry=entry, stop=stop, target=target,
                    strategy=self.name, at=ctx.now,
                    confidence=self._confidence(above_frac, distance),
                    reasons=(f"price above VWAP for {above_frac:.0%} of session",
                             f"pulled back to {distance:.2f} ATR of VWAP {vwap:.2f}",
                             "latest bar closed up -- resumption"),
                ))

        if (down_trend and distance <= self.touch_atr and price < bar_open
                and getattr(ctx.regime, "allows_short", True)):
            entry = ctx.round_tick(price)
            stop = ctx.round_tick(clamp_stop(entry, max(leg_high, entry + self.stop_atr * a), SHORT))
            if stop_is_sane(entry, stop):
                target = ctx.round_tick(entry - self.target_r * (stop - entry))
                out.append(Signal(
                    symbol=ctx.symbol, side=SHORT, entry=entry, stop=stop, target=target,
                    strategy=self.name, at=ctx.now,
                    confidence=self._confidence(1.0 - above_frac, distance),
                    reasons=(f"price below VWAP for {1 - above_frac:.0%} of session",
                             f"rallied back to {distance:.2f} ATR of VWAP {vwap:.2f}",
                             "latest bar closed down -- resumption"),
                ))
        return out

    def _confidence(self, side_frac: float, distance: float) -> float:
        base = 0.25 + 0.55 * (side_frac - self.min_side_frac) / max(1 - self.min_side_frac, 1e-9)
        base += 0.15 * (1.0 - min(distance / max(self.touch_atr, 1e-9), 1.0))
        return float(np.clip(base, 0.05, 0.90))
