"""Cross-sectional ranking that turns ~2,600 candidates into the daily 250.

Two stages, deliberately separated.

GATES come first and are absolute. A name that fails a gate is removed, never
down-weighted, because these are the properties no score should be allowed to
compensate for: we cannot trade an illiquid stock at any composite score, and a
name whose median turnover cannot absorb our order is untradeable however
attractive its momentum looks.

SCORING then ranks the survivors. Every factor is winsorised, z-scored against
the peer set on that day, and combined. Standardising cross-sectionally rather
than against a fixed scale is what makes the score comparable across regimes:
in a quiet market every ATR% is small, and an absolute threshold would empty the
universe on exactly the days it still needs to produce 250 names.

The composite is REGIME-AWARE, which is the part that answers "is the market
bullish or bearish" at the universe level rather than only at the signal level.
Relative strength enters signed in a trending regime (favour leaders to buy in a
bull, laggards to sell in a bear) and unsigned in a ranging one, where what
matters is that a name oscillates with amplitude, not which way it has drifted.

The 250 are a SCAN list, not a portfolio. Risk limits cap concurrent positions
at four; the width exists so the intraday strategies have enough candidates to
find a few clean setups, not so the bot holds 250 names.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Factor -> weight, per regime family. Weights within a family sum to 1.0.
#
# `rs` is relative strength, `trend` is cleanness of travel, `range` is how much
# intraday movement is on offer, `liquidity` is depth, `expansion` is unusual
# recent participation.
REGIME_WEIGHTS: dict[str, dict[str, float]] = {
    # Trending markets: direction is informative, so relative strength carries
    # the most weight and trend cleanness reinforces it.
    "trend": {"rs": 0.32, "trend": 0.22, "range": 0.22, "liquidity": 0.14, "expansion": 0.10},
    # Ranging markets: direction is noise. Amplitude and liquidity are what a
    # mean-reversion rule needs, so range takes the top weight and RS is used
    # unsigned (see `signed_rs` below).
    "range": {"rs": 0.10, "trend": 0.10, "range": 0.42, "liquidity": 0.24, "expansion": 0.14},
    # Volatile markets: survival first. Weight liquidity hard, discount raw
    # range because much of it is gap risk we cannot trade intraday.
    "volatile": {"rs": 0.18, "trend": 0.20, "range": 0.20, "liquidity": 0.32, "expansion": 0.10},
}

DEFAULT_REGIME_FAMILY = "trend"


@dataclass
class GateReport:
    """How many candidates each hard filter removed, in order."""

    started: int = 0
    dropped: dict[str, int] = field(default_factory=dict)
    survived: int = 0

    def render(self) -> str:
        lines = [f"gates: {self.started:,} candidates -> {self.survived:,} eligible"]
        for name, n in self.dropped.items():
            if n:
                lines.append(f"    -{n:>5,}  {name}")
        return "\n".join(lines)


@dataclass
class UniverseSelection:
    """The selected universe plus everything needed to explain it."""

    as_of: pd.Timestamp
    regime_family: str
    regime_direction: int          # +1 bull, -1 bear, 0 neutral
    frame: pd.DataFrame            # full scored+ranked eligible set
    selected: pd.DataFrame         # the top `size` rows
    gates: GateReport

    @property
    def symbols(self) -> list[str]:
        return list(self.selected.index)

    def render(self, top: int = 10) -> str:
        cols = [c for c in ("score", "rank", "close", "atr_pct", "median_turnover",
                            "ret_63", "efficiency_ratio") if c in self.selected.columns]
        head = self.selected[cols].head(top).copy()
        if "median_turnover" in head:
            head["median_turnover"] = (head["median_turnover"] / 1e7).round(2)
            head = head.rename(columns={"median_turnover": "turnover_cr"})
        for c in ("atr_pct", "ret_63", "efficiency_ratio"):
            if c in head:
                head[c] = (head[c] * 100).round(2)
        return "\n".join([
            f"universe {self.as_of.date()} | regime={self.regime_family} "
            f"dir={self.regime_direction:+d} | selected {len(self.selected):,}",
            self.gates.render(),
            "",
            head.round(3).to_string(),
        ])


def winsorised_z(s: pd.Series, limit: float = 3.0) -> pd.Series:
    """Z-score after clipping at `limit` MADs, robust to a few wild rows.

    Median/MAD rather than mean/stdev: one stock up 900% on a merger would
    otherwise inflate the standard deviation enough to compress every genuine
    difference in the rest of the cross-section into noise.
    """
    x = pd.to_numeric(s, errors="coerce")
    med = x.median()
    mad = (x - med).abs().median()
    if not np.isfinite(mad) or mad <= 0:
        std = x.std(ddof=0)
        if not np.isfinite(std) or std <= 0:
            return pd.Series(0.0, index=s.index)
        return ((x - x.mean()) / std).clip(-limit, limit).fillna(0.0)
    scaled = (x - med) / (1.4826 * mad)     # 1.4826 makes MAD a stdev estimate
    return scaled.clip(-limit, limit).fillna(0.0)


def apply_gates(f: pd.DataFrame, cfg, capital: float,
                position_notional: float | None = None) -> tuple[pd.DataFrame, GateReport]:
    """Remove candidates that are untradeable regardless of their score."""
    rep = GateReport(started=len(f))
    if f.empty:
        return f, rep
    df = f
    u = cfg.universe

    def drop(mask: pd.Series, label: str) -> pd.DataFrame:
        nonlocal df
        keep = ~mask.fillna(True)      # a NaN factor fails its own gate
        rep.dropped[label] = int((~keep).sum())
        df = df[keep]
        return df

    drop(~(df["bars"] >= u.min_history_bars), f"history < {u.min_history_bars} bars")
    drop(~df["close"].between(u.min_price, u.max_price),
         f"price outside {u.min_price:.0f}-{u.max_price:.0f}")
    drop(~(df["median_turnover"] >= u.min_median_turnover),
         f"turnover < Rs {u.min_median_turnover / 1e7:.1f} cr/day")
    drop(~(df["zero_volume_frac"] <= 0.05), "zero-volume days > 5%")
    drop(~np.isfinite(df["atr_pct"]) | ~(df["atr_pct"] > 0), "no usable ATR")

    # Capacity: our order must be a small share of what actually trades. This is
    # the gate ULTIMATE found matters most -- a zero-volume check passes names
    # three orders of magnitude too thin, and every fill against them is fiction.
    notional = position_notional if position_notional is not None else capital
    implied_units = notional / df["close"].replace(0.0, np.nan)
    participation_pct = 100.0 * implied_units / df["median_volume"].replace(0.0, np.nan)
    df = df.assign(participation_pct=participation_pct)
    drop(~(df["participation_pct"] <= u.max_participation_pct),
         f"participation > {u.max_participation_pct}% of daily volume")

    rep.survived = len(df)
    return df, rep


def score(f: pd.DataFrame, regime_family: str = DEFAULT_REGIME_FAMILY,
          direction: int = 0) -> pd.DataFrame:
    """Composite score for the eligible set, in the given regime.

    `direction` is +1 bull / -1 bear / 0 neutral and only affects the sign of
    the relative-strength block: in a bear regime the bot sells weakness, so the
    weakest names are the most attractive, not the least.
    """
    if f.empty:
        return f.assign(score=pd.Series(dtype=float))
    w = REGIME_WEIGHTS.get(regime_family, REGIME_WEIGHTS[DEFAULT_REGIME_FAMILY])
    df = f.copy()

    # Blend the return horizons before standardising so the RS block is one
    # factor rather than four correlated ones that would jointly outvote
    # everything else.
    rs_raw = (0.40 * df["ret_63"].fillna(0.0)
              + 0.35 * df["ret_126"].fillna(0.0)
              + 0.25 * df["ret_252_21"].fillna(0.0))
    if direction == 0:
        # Ranging: amplitude matters, drift does not. Use |RS| so a name that
        # moved hard in either direction still ranks above one that never moved.
        rs_raw = rs_raw.abs()
    elif direction < 0:
        rs_raw = -rs_raw

    trend_raw = (0.5 * winsorised_z(df["efficiency_ratio"])
                 + 0.5 * winsorised_z(df["slope_r2"] * (direction if direction else 1)))

    # Range is what pays for the round trip, but the part of it that arrives as
    # an overnight gap is unreachable for a bot that is flat by 15:10. Subtract
    # it rather than rank on raw ATR, which would favour gap-driven names that
    # look volatile and sit still during the session.
    intraday_range = df["range_pct"].fillna(0.0) - 0.5 * df["gap_pct"].fillna(0.0)

    df["z_rs"] = winsorised_z(rs_raw)
    df["z_trend"] = trend_raw
    df["z_range"] = winsorised_z(intraday_range)
    df["z_liquidity"] = winsorised_z(np.log10(df["median_turnover"].clip(lower=1.0)))
    df["z_expansion"] = winsorised_z(df["turnover_expansion"])

    df["score"] = (w["rs"] * df["z_rs"]
                   + w["trend"] * df["z_trend"]
                   + w["range"] * df["z_range"]
                   + w["liquidity"] * df["z_liquidity"]
                   + w["expansion"] * df["z_expansion"])

    df = df.sort_values("score", ascending=False)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def select_universe(factors: pd.DataFrame, cfg, regime_family: str = DEFAULT_REGIME_FAMILY,
                    direction: int = 0, as_of: pd.Timestamp | None = None,
                    position_notional: float | None = None) -> UniverseSelection:
    """Gate, score and take the top `cfg.universe.size` names."""
    as_of = pd.Timestamp(as_of) if as_of is not None else (
        pd.Timestamp(factors["as_of"].max()) if "as_of" in factors.columns and len(factors)
        else pd.Timestamp("NaT")
    )
    eligible, gates = apply_gates(factors, cfg, cfg.capital, position_notional)
    scored = score(eligible, regime_family, direction)
    selected = scored.head(cfg.universe.size)
    return UniverseSelection(
        as_of=as_of,
        regime_family=regime_family,
        regime_direction=direction,
        frame=scored,
        selected=selected,
        gates=gates,
    )
