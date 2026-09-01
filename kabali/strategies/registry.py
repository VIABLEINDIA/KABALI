"""The default strategy set, and the router that picks among them.

Coverage by regime family is deliberate and complete -- every family the
classifier can emit has at least one rule, so the bot is never silently
strategy-less in a regime it can actually be in:

    trend     OpeningRangeBreakout, VWAPPullback, MomentumBurst
    range     VWAPReversion, GapFade
    volatile  MomentumBurst

The volatile family is intentionally thin. ULTIMATE's regime attribution found
bear/high-vol to be four trades at an average of -Rs 99,707 with a 0% win rate,
so the correct default in violent conditions is to do almost nothing rather than
to reach for more rules.

`route` is where signals from different rules on the same symbol are reconciled.
Two rules firing opposite ways on one name is not a hedge, it is a contradiction:
neither is taken. Two rules agreeing is real corroboration and the confidences
combine, because independent rules reaching the same conclusion is more
informative than either alone.
"""
from __future__ import annotations

from collections import defaultdict

from kabali.strategies.base import Signal, StrategyRegistry, SymbolContext
from kabali.strategies.gap_fade import GapFade
from kabali.strategies.momentum_burst import MomentumBurst
from kabali.strategies.opening_range import OpeningRangeBreakout
from kabali.strategies.vwap_pullback import VWAPPullback
from kabali.strategies.vwap_reversion import VWAPReversion

__all__ = ["default_registry", "route", "GapFade", "MomentumBurst",
           "OpeningRangeBreakout", "VWAPPullback", "VWAPReversion"]


def default_registry() -> StrategyRegistry:
    return StrategyRegistry([
        OpeningRangeBreakout(),
        VWAPPullback(),
        MomentumBurst(),
        VWAPReversion(),
        GapFade(),
    ])


def _agree(signals: list[Signal]) -> Signal:
    """Merge same-side signals on one symbol into a single corroborated signal.

    Combination is 1 - prod(1 - c), the probability at least one independent
    rule is right. It rises with agreement but never reaches certainty, which
    matters because these rules are not truly independent -- they read the same
    bars -- so anything that saturated would overstate the evidence.

    The tightest stop wins. When two rules disagree about where the idea is
    falsified, the earlier invalidation is the honest one.
    """
    best = max(signals, key=lambda s: s.confidence)
    if len(signals) == 1:
        return best
    combined = 1.0
    for s in signals:
        combined *= (1.0 - min(max(s.confidence, 0.0), 0.99))
    confidence = min(1.0 - combined, 0.95)

    if best.side == "long":
        stop = max(s.stop for s in signals)
        target = min(s.target for s in signals)
    else:
        stop = min(s.stop for s in signals)
        target = max(s.target for s in signals)

    names = "+".join(sorted({s.strategy for s in signals}))
    reasons = tuple(r for s in signals for r in s.reasons)
    try:
        return Signal(
            symbol=best.symbol, side=best.side, entry=best.entry,
            stop=stop, target=target, strategy=names, at=best.at,
            confidence=confidence,
            reasons=(f"{len(signals)} rules agree",) + reasons,
        )
    except ValueError:
        # The rules priced their entries differently, so the tightest stop of one
        # sits on the wrong side of another's entry and the merge is not a valid
        # signal. In practice every rule enters at the confirmed close, so this is
        # rare -- but silently emitting an invalid combination would be worse than
        # falling back to the single strongest rule, unmerged.
        return best


def route(ctx: SymbolContext, registry: StrategyRegistry) -> list[Signal]:
    """Run every strategy valid in the current regime and reconcile the results."""
    family = getattr(ctx.regime, "family", "trend")
    raw: list[Signal] = []
    for strategy in registry.for_family(family):
        if not strategy.ready(ctx):
            continue
        try:
            raw.extend(strategy.generate(ctx))
        except Exception:                       # noqa: BLE001
            # One malformed rule must not take down the scan for every other
            # symbol. The engine logs these; a rule that raises repeatedly is a
            # bug to fix, not a reason to stop trading the other four.
            continue

    if not raw:
        return []

    by_side: dict[str, list[Signal]] = defaultdict(list)
    for s in raw:
        by_side[s.side].append(s)

    # Contradiction: rules disagree on direction for this symbol. Skip it.
    if len(by_side) > 1:
        return []

    side = next(iter(by_side))
    return [_agree(by_side[side])]
