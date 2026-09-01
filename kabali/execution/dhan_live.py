"""Live Dhan order path. Constructing this object is the act of risking money.

Everything about this module is arranged so that reaching a real order requires
passing through the gate, and so that no other route exists.

WHY THIS DOES NOT USE `kabali.core`. The research bridge wraps the SDK in
`ReadOnlyDhan`, which raises `OrderPlacementBlocked` on any method outside a
read-only allowlist. That wrapper is load-bearing and must stay intact, so this
module builds its own `dhanhq` client instead of unwrapping the protected one.
The consequence is the property worth having: nothing that imports the research
core can place an order, no matter what it does with the object it was handed.

CONSTRUCTION IS THE CHECKPOINT. `__init__` demands a `GateVerdict` that actually
passed and re-verifies it rather than trusting the caller's word. There is no
`force=True`, no environment-variable override, and no default that skips the
check -- a caller who wants a live broker has to have already obtained evidence
that the gate opened.

THERE ARE TWO DOORS, AND THE SECOND IS DELIBERATELY TINY. `for_smoke_test` opens
a broker that may place one minimum-size round trip and nothing else. It exists
because the gate asks a question paper trading cannot answer: whether a real
order fills where the model says it will. Market impact, partial fills, queue
position and exchange rejection are invisible to a simulator, and the only
instrument that measures them is a real order.

It is a separate door rather than a relaxation of the first one. The strategy
gate is untouched and still refuses -- a smoke authorization cannot run a
session, size a position, or place a second round trip. Its caps are enforced
here at submit time rather than trusted to the caller, because a bound that
lives only in a script is a bound that the next script forgets. Both doors are
in this module so that the audit surface stays one file.

FAILURES ARE RETURNED, NOT RAISED. A rejected order is a `Fill` with
status='rejected', because a live session that crashed on a single rejection
would leave open positions unmanaged with no square-off. The one exception is
`GateClosed`, which is raised and never caught: if the gate is shut, the correct
behaviour is to not start.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from kabali.execution.broker import BUY, Broker, BrokerState, Fill, OrderIntent
from kabali.execution.gate import GateClosed, GateVerdict

log = logging.getLogger(__name__)

# dhanhq transaction/product/order constants, named locally so a change in the
# SDK's exported names surfaces here rather than as a malformed order payload.
TXN_BUY = "BUY"
TXN_SELL = "SELL"
PRODUCT_INTRADAY = "INTRADAY"
ORDER_MARKET = "MARKET"
ORDER_LIMIT = "LIMIT"
VALIDITY_DAY = "DAY"

ORDER_TAG = "KABALI"
SMOKE_TAG = "KABALI-SMOKE"

#: The operator writes this by hand into state/SMOKE_TEST.json. Distinct from the
#: live-gate phrase so that authorising a one-share probe can never be mistaken
#: for, or copied into, authorisation to trade the strategy.
SMOKE_PHRASE = "I ACCEPT ONE REAL TEST ORDER"


class SmokeRefused(RuntimeError):
    """A smoke-test order was requested without valid, in-bounds authorisation."""


@dataclass(frozen=True)
class SmokeAuthorization:
    """Permission to place one minimum-size round trip, and nothing more.

    Deliberately not a boolean. The caps travel with the permission so that the
    broker can enforce them without consulting anything the caller controls.
    """

    max_quantity: int = 1
    max_notional: float = 5_000.0
    max_orders: int = 2               # one entry, one exit -- no more
    symbol: str | None = None         # when set, the only name it may touch

    @classmethod
    def load(cls, path=None) -> "SmokeAuthorization":
        """Read state/SMOKE_TEST.json, refusing anything short of an exact phrase."""
        from kabali.config import STATE_DIR

        p = Path(path) if path else STATE_DIR / "SMOKE_TEST.json"
        if not p.exists():
            raise SmokeRefused(
                f"no smoke authorisation at {p}. Write it by hand: "
                f'{{"confirm": "{SMOKE_PHRASE}", "symbol": "<TICKER>"}} -- '
                f"this authorises ONE real order of ONE share. It is not the live gate.")
        try:
            blob = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SmokeRefused(f"{p} is not valid JSON: {exc}") from exc
        if str(blob.get("confirm", "")) != SMOKE_PHRASE:
            raise SmokeRefused(f"confirm must be exactly {SMOKE_PHRASE!r}")

        qty = int(blob.get("max_quantity", 1))
        notional = float(blob.get("max_notional", 5_000.0))
        if qty < 1 or qty > 1:
            raise SmokeRefused("max_quantity must be exactly 1; this is a probe, not a trade")
        if notional <= 0 or notional > 20_000.0:
            raise SmokeRefused("max_notional must be within (0, 20000]")
        return cls(max_quantity=qty, max_notional=notional,
                   symbol=(blob.get("symbol") or None))


class DhanBroker(Broker):
    """Places real intraday orders on Dhan. Gated at construction."""

    mode = "live"

    def __init__(self, cfg, verdict: GateVerdict | None, creds=None,
                 smoke: SmokeAuthorization | None = None):
        # Exactly one authorisation. Never both -- they answer different
        # questions, and a caller holding both has confused them.
        if smoke is None:
            if verdict is None or not isinstance(verdict, GateVerdict):
                raise GateClosed("DhanBroker requires a GateVerdict; refusing to construct")
            if not verdict.passed:
                raise GateClosed(
                    "DhanBroker refused: the live gate did not pass.\n" + verdict.render())
        elif verdict is not None:
            raise GateClosed(
                "a smoke authorisation and a gate verdict were both supplied; "
                "these are different permissions and combining them is a mistake")
        self.smoke = smoke
        self._orders_placed = 0

        from dhanhq import DhanContext, dhanhq            # lazy: paper runs need no SDK

        from kabali.core import ultimate_root             # noqa: F401  (path side effect)
        from research.config import load_credentials

        self.cfg = cfg
        self.verdict = verdict
        self.creds = creds or load_credentials()
        self._ctx = DhanContext(self.creds.client_id, self.creds.access_token)
        # Deliberately NOT wrapped in ReadOnlyDhan -- this is the one place in
        # the codebase that is permitted an order surface.
        self.api = dhanhq(self._ctx)
        self.product = cfg.execution.order_product or PRODUCT_INTRADAY
        self.order_type = (cfg.execution.order_type or ORDER_MARKET).upper()
        self.fills: list[Fill] = []
        if smoke is not None:
            log.warning("SMOKE broker constructed: at most %d order(s), %d share, "
                        "Rs %.0f notional%s", smoke.max_orders, smoke.max_quantity,
                        smoke.max_notional,
                        f", {smoke.symbol} only" if smoke.symbol else "")
        else:
            log.warning("LIVE broker constructed: real orders are now possible "
                        "(product=%s type=%s)", self.product, self.order_type)

    @classmethod
    def for_smoke_test(cls, cfg, auth: SmokeAuthorization, creds=None) -> "DhanBroker":
        """A broker permitted one minimum-size round trip and nothing else.

        Does NOT consult the live gate, and cannot substitute for it: the caps
        below make a strategy session impossible regardless of what the caller
        intends. The gate answers "should this system trade?"; this answers
        "does an order arrive where the model says it will?" -- and only a real
        order can answer the second.
        """
        if not isinstance(auth, SmokeAuthorization):
            raise SmokeRefused("for_smoke_test requires a SmokeAuthorization")
        return cls(cfg, verdict=None, creds=creds, smoke=auth)

    def _check_smoke_bounds(self, intent: OrderIntent) -> str | None:
        """Enforce the authorisation's caps. Returns a refusal reason, or None.

        Enforced here rather than in the calling script on purpose. A bound that
        lives in a script is a bound the next script forgets, and the whole point
        of this path is that its worst case is knowable by reading one place.
        """
        a = self.smoke
        if self._orders_placed >= a.max_orders:
            return (f"smoke authorisation permits {a.max_orders} order(s) and "
                    f"{self._orders_placed} have been placed")
        if intent.quantity > a.max_quantity:
            return f"quantity {intent.quantity} exceeds the authorised {a.max_quantity}"
        notional = float(intent.reference_price) * int(intent.quantity)
        if notional > a.max_notional:
            return f"notional Rs {notional:,.0f} exceeds the authorised Rs {a.max_notional:,.0f}"
        if a.symbol and intent.symbol != a.symbol:
            return f"authorised for {a.symbol}, not {intent.symbol}"
        return None

    # ------------------------------------------------------------------ core
    def submit(self, intent: OrderIntent) -> Fill:
        if self.smoke is not None:
            refusal = self._check_smoke_bounds(intent)
            if refusal:
                return self._reject(intent, f"smoke bounds: {refusal}")
        txn = TXN_BUY if intent.side == BUY else TXN_SELL
        # A MARKET order carries price 0; sending a non-zero price with MARKET is
        # a common cause of a silent rejection at the exchange.
        price = 0.0 if self.order_type == ORDER_MARKET else round(intent.reference_price, 2)

        try:
            resp = self.api.place_order(
                security_id=str(intent.security_id),
                exchange_segment="NSE_EQ",
                transaction_type=txn,
                quantity=int(intent.quantity),
                order_type=self.order_type,
                product_type=self.product,
                price=price,
                validity=VALIDITY_DAY,
                tag=SMOKE_TAG if self.smoke is not None else ORDER_TAG,
            )
        except Exception as exc:                          # noqa: BLE001
            return self._reject(intent, f"place_order raised: {type(exc).__name__}: {exc}")
        finally:
            # Counted on ATTEMPT, not on success. An order that errored may still
            # have reached the exchange, and a retry loop that only counted
            # confirmed fills could place many.
            self._orders_placed += 1

        if not isinstance(resp, dict) or resp.get("status") != "success":
            return self._reject(intent, f"broker rejected: {_describe(resp)}")

        data = resp.get("data") or {}
        order_id = str(data.get("orderId") or data.get("order_id") or "")
        if not order_id:
            return self._reject(intent, f"accepted but returned no order id: {_describe(resp)}")

        # A MARKET order that is accepted is not yet a fill. The engine reconciles
        # the true traded price from the order book; until then the reference
        # price is a placeholder and is marked as such.
        fill = Fill(
            symbol=intent.symbol, side=intent.side, quantity=intent.quantity,
            price=float(intent.reference_price), cost=0.0, at=intent.at,
            order_id=order_id, status="filled", slippage=0.0,
            note=f"{intent.reason} (price unconfirmed; reconcile via order book)",
        )
        self.fills.append(fill)
        return fill

    def reconcile(self, fill: Fill) -> Fill:
        """Replace a placeholder price with the actual traded price, if available.

        Live slippage measurement depends entirely on this. Without it the gate's
        slippage-fidelity check would compare the model against itself.
        """
        if not fill.order_id:
            return fill
        try:
            resp = self.api.get_order_by_id(fill.order_id)
        except Exception as exc:                          # noqa: BLE001
            log.warning("reconcile %s failed: %s", fill.order_id, exc)
            return fill
        if not isinstance(resp, dict) or resp.get("status") != "success":
            return fill
        data = resp.get("data") or {}
        if isinstance(data, list):
            data = data[0] if data else {}
        traded = data.get("averageTradedPrice") or data.get("average_traded_price")
        status = str(data.get("orderStatus") or data.get("order_status") or "").upper()
        if traded:
            ref = fill.price
            fill.price = float(traded)
            fill.slippage = round(fill.price - ref if fill.side == BUY else ref - fill.price, 4)
            fill.note = fill.note.replace(" (price unconfirmed; reconcile via order book)", "")
        if status in ("REJECTED", "CANCELLED"):
            fill.status = "rejected"
            fill.note = f"order {status.lower()} at the exchange"
        return fill

    def state(self) -> BrokerState:
        try:
            resp = self.api.get_fund_limits()
        except Exception as exc:                          # noqa: BLE001
            return BrokerState(source=f"unavailable: {exc}", fetched_at=pd.Timestamp.now())
        if not isinstance(resp, dict) or resp.get("status") != "success":
            return BrokerState(source=f"unavailable: {_describe(resp)}",
                               fetched_at=pd.Timestamp.now())
        d = resp.get("data") or {}
        return BrokerState(
            available_margin=_num(d.get("availabelBalance", d.get("availableBalance"))),
            utilised_margin=_num(d.get("utilizedAmount")),
            source="dhan", fetched_at=pd.Timestamp.now(), extra=d,
        )

    # ----------------------------------------------------------------- detail
    def _reject(self, intent: OrderIntent, note: str) -> Fill:
        log.error("order rejected %s %s x%d: %s",
                  intent.side, intent.symbol, intent.quantity, note)
        fill = Fill(symbol=intent.symbol, side=intent.side, quantity=0, price=0.0,
                    cost=0.0, at=intent.at, status="rejected", note=note)
        self.fills.append(fill)
        return fill


def _describe(resp) -> str:
    if not isinstance(resp, dict):
        return str(resp)[:200]
    r = resp.get("remarks")
    if isinstance(r, dict):
        return f"{r.get('error_code')}: {r.get('error_message')}"
    return str(r or resp)[:200]


def _num(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")
