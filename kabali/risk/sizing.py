"""Position sizing: how many shares, and whether to trade at all.

Sizing is derived from the STOP, never from the price. The quantity that puts
`risk_per_trade` rupees at risk is the only quantity that makes two setups on
different stocks carry the same risk, and it is what lets the daily loss limit
mean something: four positions at Rs 200 of risk is Rs 800, whatever the share
prices happen to be.

    quantity = risk_per_trade / (|entry - stop| + entry * round_trip_cost_rate)

The cost term is not a refinement. Costs leave the account exactly as adverse
price movement does, so a budget that ignores them is not the budget it claims
to be -- see the comment in `size_position`.

That number is then reduced by four independent caps, and the SMALLEST wins.
Each exists for a different failure:

  notional cap      one name cannot become the whole book
  gross cap         total exposure stays inside the leverage we assumed, not
                    the leverage the broker happens to offer today
  participation     our order stays a small share of what actually trades, the
                    gate ULTIMATE found matters most -- thin names pass a
                    zero-volume check and still cannot absorb the order
  affordability     MIS buying power actually available

If the result is below `min_position_notional` the trade is REJECTED rather than
taken small. At Rs 40,000, a Rs 3,000 position pays the same fixed brokerage as a
Rs 20,000 one, so the small version needs several times the move to break even.
Taking it is negative expectancy arithmetic regardless of how good the signal is.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SizingDecision:
    """The sized order, or a refusal with the reason."""

    approved: bool
    quantity: int = 0
    notional: float = 0.0
    risk_amount: float = 0.0      # cost-inclusive: what a stop-out really costs
    price_risk: float = 0.0       # |entry - stop| * quantity, costs excluded
    reason: str = ""
    binding_cap: str = ""
    caps: dict[str, int] = field(default_factory=dict)

    def describe(self) -> str:
        if not self.approved:
            return f"REJECT {self.reason}"
        return (f"qty {self.quantity} | notional Rs {self.notional:,.0f} | "
                f"risk Rs {self.risk_amount:,.0f} | capped by {self.binding_cap}")


def size_position(signal, cfg, book, median_volume: float | None = None,
                  available_margin: float | None = None,
                  risk_limits=None, cost_rate: float | None = None) -> SizingDecision:
    """Size one signal against the risk config and the current book.

    `risk_limits` and `cost_rate` exist for the swing venue, which sizes on the
    same arithmetic but against different numbers: cash rather than MIS
    leverage, and delivery costs, whose STT is 0.1% per side against intraday's
    0.025% on the sell alone. Passing them in keeps ONE sizing implementation.
    A second copy that drifted would be found the way the intraday one was --
    by a paper run settling past a limit it claimed to enforce.

    `risk_limits` must expose the `RiskConfig` surface used below
    (`risk_per_trade`, `position_notional_cap`, `gross_exposure_cap`,
    `min_position_notional`); `SwingRiskConfig` does.
    """
    risk = risk_limits if risk_limits is not None else cfg.risk
    capital = cfg.capital

    risk_per_unit = signal.risk_per_unit
    if not np.isfinite(risk_per_unit) or risk_per_unit <= 0:
        return SizingDecision(False, reason="stop distance is zero or undefined")
    if not np.isfinite(signal.entry) or signal.entry <= 0:
        return SizingDecision(False, reason="entry price is invalid")

    budget = risk.risk_per_trade(capital)

    # Cost is part of the loss, so it must be part of the budget.
    #
    # Sizing on |entry - stop| alone understates what a stopped-out trade
    # actually removes from the account by the round trip's cost, and the error
    # compounds: four positions at Rs 200 of price risk plus ~Rs 21 of costs each
    # lose Rs 884 against an Rs 800 daily limit. The limit then cannot hold, and
    # a paper run flags it as a risk-engine bug -- correctly, because it is one.
    #
    # Per share the loss is (risk_per_unit + entry * cost_rate), so:
    #     quantity = budget / (risk_per_unit + entry * cost_rate)
    cost_rate = _round_trip_rate(cfg) if cost_rate is None else cost_rate
    risk_per_share = risk_per_unit + signal.entry * cost_rate
    base_qty = int(budget // risk_per_share)
    if base_qty < 1:
        return SizingDecision(
            False,
            reason=(f"stop is Rs {risk_per_unit:.2f} wide; with costs even 1 share "
                    f"risks Rs {risk_per_share:.2f}, above the Rs {budget:.0f} "
                    f"per-trade budget"),
        )

    caps: dict[str, int] = {"risk_budget": base_qty}

    # One name cannot dominate the book.
    caps["position_notional"] = int(risk.position_notional_cap(capital) // signal.entry)

    # Remaining room under the gross exposure ceiling.
    remaining_gross = max(risk.gross_exposure_cap(capital) - book.gross_exposure, 0.0)
    caps["gross_exposure"] = int(remaining_gross // signal.entry)

    # Capacity against what the name actually trades.
    if median_volume is not None and np.isfinite(median_volume) and median_volume > 0:
        caps["participation"] = int(
            median_volume * cfg.universe.max_participation_pct / 100.0)

    # Actual buying power, when the broker has told us.
    if available_margin is not None and np.isfinite(available_margin):
        caps["margin"] = int(max(available_margin, 0.0) // signal.entry)

    binding_cap = min(caps, key=lambda k: caps[k])
    quantity = max(min(caps.values()), 0)

    if quantity < 1:
        return SizingDecision(
            False, reason=f"{binding_cap} cap allows 0 shares", binding_cap=binding_cap, caps=caps)

    notional = quantity * signal.entry
    if notional < risk.min_position_notional:
        return SizingDecision(
            False,
            reason=(f"notional Rs {notional:,.0f} below the Rs "
                    f"{risk.min_position_notional:,.0f} floor ({binding_cap} cap binds); "
                    f"fixed costs would dominate"),
            binding_cap=binding_cap, caps=caps,
        )

    return SizingDecision(
        approved=True, quantity=quantity, notional=notional,
        # Cost-inclusive, so `book.open_risk` sums what the positions can really
        # lose and the circuit breaker's remaining-budget check is honest.
        risk_amount=quantity * risk_per_share,
        price_risk=quantity * risk_per_unit,
        binding_cap=binding_cap, caps=caps, reason="ok",
    )


def round_trip_rate(costs) -> float:
    """Round-trip cost as a fraction of notional: statutory model plus slippage.

    Slippage counts because it is a real adverse fill, not a modelling artifact --
    PaperBroker and the live venue both move the price against the order on each
    leg, so it leaves the account exactly as brokerage does.

    Takes a `CostsConfig` rather than the whole bot config because there are now
    two of them: `cfg.costs` is intraday, `cfg.swing.costs` is delivery.
    """
    from kabali.core import get_cost_model
    model = get_cost_model(costs.profile)
    return (model.round_trip_estimate_bps() + 2.0 * costs.slippage_bps) / 10_000.0


def _round_trip_rate(cfg) -> float:
    """The intraday round-trip rate. Kept for callers that pass the bot config."""
    return round_trip_rate(cfg.costs)
