"""Market regime classification from the benchmark index plus breadth.

The bot needs two separate readings and they are genuinely independent:

  DIRECTION  is the market going up, down, or neither -- decides whether the
             strategies are allowed to go long, short, or both.
  FAMILY     is it trending, ranging, or violent -- decides WHICH strategies are
             allowed to run at all, because a breakout rule and a fade rule are
             profitable in opposite conditions.

Collapsing these into one "bullish/bearish" label is the usual mistake. A market
can fall in a clean downtrend (trend family, bearish direction: breakouts work
short) or fall in a volatile panic (volatile family, bearish direction: breakouts
get whipsawed and the correct action is often to stand down). Same direction,
opposite strategy.

ULTIMATE's regime attribution is the reason this module exists at all. On daily
cash equity it found bull/low-vol held 41% of trades and 83% of net P&L, while
bear/high-vol was 4 trades at an average of -Rs 99,707 with a 0% win rate. The
edge, such as it was, lived entirely in one regime. A classifier that keeps the
bot out of the bad one is worth more than another entry rule.

Classification runs on CONFIRMED daily bars only -- the regime for today is
computed from bars up to yesterday's close, so it is knowable before the open.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from kabali.core import atr, ema, rolling_percentile, sma
from kabali.universe.factors import efficiency_ratio

# Absolute floor: below this the index is travelling without arriving, whatever
# its recent history looks like. An efficiency ratio of 0.10 over 50 sessions
# means the index covered ten times its net move -- the top third of a market
# that behaves like that is still chop, and calling it a trend would let breakout
# rules run in the conditions they are worst in.
TREND_EFFICIENCY_FLOOR = 0.10

# The trend test is a PERCENTILE of the index's own trailing efficiency, not an
# absolute number, because an absolute one cannot be calibrated in advance.
#
# The first version used a flat 0.30. Measured against NIFTY's actual
# distribution that turned out to be the 90th percentile -- median 0.126,
# p75 0.202, p90 0.298 -- so before the volatility override only ~10% of
# sessions could qualify, and after it 2.3% did. Three of the five strategies
# require the trend family, so they were structurally almost unreachable: the
# router looked complete and most of the stack could never fire.
#
# A percentile is self-calibrating. "More directional than this index usually
# is" survives a change of index, of window length, or of volatility regime,
# where a hand-picked constant survives none of them.
TREND_EFFICIENCY_PERCENTILE = 65.0
EFFICIENCY_PERCENTILE_WINDOW = 252
# Below this many observations the percentile is not meaningful and the absolute
# floor decides alone.
MIN_PERCENTILE_OBS = 60

# Breadth extremes. Above/below these the direction call is reinforced.
BREADTH_BULL = 0.55
BREADTH_BEAR = 0.45

FAMILIES = ("trend", "range", "volatile")


@dataclass
class RegimeState:
    """The market regime as of one date, with the evidence behind it."""

    as_of: pd.Timestamp
    family: str                 # trend | range | volatile
    direction: int              # +1 bull, -1 bear, 0 neutral
    confidence: float           # 0..1
    close: float = float("nan")
    ema_fast: float = float("nan")
    ema_slow: float = float("nan")
    sma_anchor: float = float("nan")
    atr_pct: float = float("nan")
    atr_percentile: float = float("nan")
    efficiency: float = float("nan")
    efficiency_percentile: float = float("nan")
    breadth: float = float("nan")
    reasons: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        d = {1: "bull", -1: "bear", 0: "neutral"}[self.direction]
        return f"{d}_{self.family}"

    @property
    def allows_long(self) -> bool:
        return self.direction >= 0

    @property
    def allows_short(self) -> bool:
        return self.direction <= 0

    def render(self) -> str:
        lines = [
            f"regime {pd.Timestamp(self.as_of).date()}: {self.label.upper()} "
            f"(confidence {self.confidence:.0%})",
            f"  close {self.close:,.1f} | EMA fast {self.ema_fast:,.1f} "
            f"slow {self.ema_slow:,.1f} | SMA anchor {self.sma_anchor:,.1f}",
            f"  ATR {self.atr_pct * 100:.2f}% (pctile {self.atr_percentile:.0f}) "
            f"| efficiency {self.efficiency:.2f} (p{self.efficiency_percentile:.0f}) "
            f"| breadth {self.breadth:.0%}"
            if np.isfinite(self.breadth) else
            f"  ATR {self.atr_pct * 100:.2f}% (pctile {self.atr_percentile:.0f}) "
            f"| efficiency {self.efficiency:.2f} (p{self.efficiency_percentile:.0f}) "
            f"| breadth n/a",
            f"  long={'yes' if self.allows_long else 'no'} "
            f"short={'yes' if self.allows_short else 'no'}",
        ]
        lines += [f"  - {r}" for r in self.reasons]
        return "\n".join(lines)


def breadth_above_ma(universe_bars: dict[str, pd.DataFrame], length: int = 50,
                     as_of: pd.Timestamp | None = None) -> float:
    """Fraction of sampled names trading above their own `length`-day average.

    Breadth catches the case a single index level hides: a cap-weighted index
    held up by four heavyweights while most of the market is already rolling
    over. That divergence is common right before the volatile regime the
    attribution table says is expensive.
    """
    above = total = 0
    for _, df in universe_bars.items():
        if df is None or df.empty or "close" not in df:
            continue
        d = df
        if as_of is not None and "timestamp" in d.columns:
            d = d[d["timestamp"] <= pd.Timestamp(as_of)]
        close = pd.to_numeric(d["close"], errors="coerce")
        if len(close) < length:
            continue
        ma = sma(close, length)
        last_ma, last_close = ma.iloc[-1], close.iloc[-1]
        if not (np.isfinite(last_ma) and np.isfinite(last_close)):
            continue
        total += 1
        above += int(last_close > last_ma)
    return above / total if total else float("nan")


def classify(bench_bars: pd.DataFrame, cfg,
             universe_bars: dict[str, pd.DataFrame] | None = None,
             as_of: pd.Timestamp | str | None = None) -> RegimeState:
    """Classify the regime from benchmark bars, optionally reinforced by breadth."""
    rg = cfg.regime
    df = bench_bars
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")
        if as_of is not None:
            df = df[df["timestamp"] <= pd.Timestamp(as_of)]
    need = max(rg.trend_anchor, rg.atr_window) + 5
    if len(df) < need:
        raise ValueError(
            f"regime needs >= {need} benchmark bars, got {len(df)}. "
            f"Ingest more {rg.benchmark} history before classifying."
        )

    close = pd.to_numeric(df["close"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    stamp = pd.Timestamp(df["timestamp"].iloc[-1]) if "timestamp" in df.columns else pd.Timestamp(as_of)

    e_fast = float(ema(close, rg.trend_fast).iloc[-1])
    e_slow = float(ema(close, rg.trend_slow).iloc[-1])
    s_anchor = float(sma(close, rg.trend_anchor).iloc[-1])
    last = float(close.iloc[-1])

    atr_series = atr(high, low, close, rg.atr_window)
    atr_pct_series = (atr_series / close.replace(0.0, np.nan))
    atr_pct = float(atr_pct_series.iloc[-1])
    pct_window = min(252, len(atr_pct_series.dropna()))
    atr_pctile = float(rolling_percentile(atr_pct_series, pct_window).iloc[-1] * 100.0) \
        if pct_window >= 30 else float("nan")

    eff = efficiency_ratio(close, window=rg.trend_slow)
    eff_pctile = _efficiency_percentile(close, rg.trend_slow)
    breadth = breadth_above_ma(universe_bars, length=50, as_of=stamp) \
        if universe_bars else float("nan")

    reasons: list[str] = []

    # ---- direction -------------------------------------------------------
    score = 0
    if last > s_anchor:
        score += 1
        reasons.append(f"close above {rg.trend_anchor}d SMA")
    elif last < s_anchor:
        score -= 1
        reasons.append(f"close below {rg.trend_anchor}d SMA")
    if e_fast > e_slow:
        score += 1
        reasons.append(f"EMA{rg.trend_fast} above EMA{rg.trend_slow}")
    elif e_fast < e_slow:
        score -= 1
        reasons.append(f"EMA{rg.trend_fast} below EMA{rg.trend_slow}")
    if np.isfinite(breadth):
        if breadth >= BREADTH_BULL:
            score += 1
            reasons.append(f"breadth {breadth:.0%} of names above 50d MA")
        elif breadth <= BREADTH_BEAR:
            score -= 1
            reasons.append(f"breadth only {breadth:.0%} of names above 50d MA")

    direction = 1 if score >= 2 else -1 if score <= -2 else 0
    if direction == 0:
        reasons.append("trend evidence mixed -- direction neutral, both sides gated")

    # ---- family ----------------------------------------------------------
    # Volatility is checked FIRST and overrides. A violent market can show a
    # clean trend reading and still be the regime where stops get run; ULTIMATE's
    # bear/high-vol bucket was 0% wins.
    if np.isfinite(atr_pctile) and atr_pctile >= rg.high_vol_percentile:
        family = "volatile"
        reasons.append(f"ATR at {atr_pctile:.0f}th percentile -- volatile regime")
    elif not np.isfinite(eff) or eff < TREND_EFFICIENCY_FLOOR:
        family = "range"
        reasons.append(
            f"efficiency {eff:.2f} below the {TREND_EFFICIENCY_FLOOR} floor -- "
            f"index is travelling without arriving")
    elif np.isfinite(eff_pctile) and eff_pctile < TREND_EFFICIENCY_PERCENTILE:
        family = "range"
        reasons.append(
            f"efficiency {eff:.2f} is only its own {eff_pctile:.0f}th percentile "
            f"(needs {TREND_EFFICIENCY_PERCENTILE:.0f}) -- normal chop for this index")
    else:
        family = "trend"
        where = (f", its {eff_pctile:.0f}th percentile" if np.isfinite(eff_pctile) else "")
        reasons.append(f"efficiency {eff:.2f}{where} -- index is travelling directionally")

    # ---- confidence ------------------------------------------------------
    # Agreement among independent readings, not a probability. Efficiency enters
    # as its percentile rather than its level, for the same reason the family
    # test does: 0.25 is unremarkable on one index and exceptional on another.
    eff_term = (eff_pctile / 100.0 if np.isfinite(eff_pctile)
                else (min(eff / 0.6, 1.0) if np.isfinite(eff) else 0.0))
    parts = [
        abs(score) / 3.0,
        min(abs(e_fast - e_slow) / max(abs(e_slow), 1e-9) / 0.02, 1.0),
        eff_term,
    ]
    confidence = float(np.clip(np.mean(parts), 0.0, 1.0))

    return RegimeState(
        as_of=stamp, family=family, direction=direction, confidence=confidence,
        close=last, ema_fast=e_fast, ema_slow=e_slow, sma_anchor=s_anchor,
        atr_pct=atr_pct, atr_percentile=atr_pctile, efficiency=eff,
        efficiency_percentile=eff_pctile, breadth=breadth, reasons=reasons,
    )


def _efficiency_percentile(close: pd.Series, window: int) -> float:
    """Where today's efficiency ratio sits in the index's own recent history.

    Computed over a trailing window of past efficiency readings, so it is
    knowable at the bar it labels. Returns NaN when there is too little history,
    in which case the absolute floor is the only test that applies.
    """
    n = len(close)
    span = min(EFFICIENCY_PERCENTILE_WINDOW, max(n - window - 1, 0))
    if span < MIN_PERCENTILE_OBS:
        return float("nan")
    history = [efficiency_ratio(close.iloc[: n - k], window) for k in range(span, 0, -1)]
    series = pd.Series(history).dropna()
    if len(series) < MIN_PERCENTILE_OBS:
        return float("nan")
    current = efficiency_ratio(close, window)
    if not np.isfinite(current):
        return float("nan")
    return float((series < current).mean() * 100.0)
