"""Place ONE real share, measure the fill, exit. The order-path probe.

    python scripts/live_smoke.py --dry-run          # everything except the order
    python scripts/live_smoke.py                    # places two real orders

WHAT THIS IS FOR
================
The live gate asks whether the system should trade. This asks a different and
much smaller question: does a real order arrive where the model says it will?

Paper trading cannot answer it. `PaperBroker` fills at the reference price plus
the modelled slippage, so its observed figure is the model played back. Measuring
against the bars -- what `slippage_source: market` records -- is better, because
bars are independent of the model, but bars still cannot see market impact,
partial fills, queue position, or an exchange rejection. The only instrument that
measures those is a real order.

So this buys one share of one liquid name and sells it immediately. It does not
consult a strategy, size a position, or express a view. The cost is the spread
plus two lots of brokerage -- tens of rupees -- and what it buys is the first
honest reading of the live order path.

WHY IT IS NOT THE LIVE GATE
===========================
It runs on `state/SMOKE_TEST.json`, a separate authorisation with its own phrase
and its own caps: one share, one round trip, a notional ceiling. Those caps are
enforced inside `DhanBroker`, not here, so this script could be rewritten
carelessly and still not place a third order. The strategy gate is untouched and
still refuses -- passing this probe does not move it one inch closer to open.

THE POSITION MUST NOT SURVIVE THIS SCRIPT
=========================================
Every exit path closes the position, including exceptions and Ctrl-C. An open
one-share position is a trivial loss and a serious bug: it means the exit path
has a hole, and the exit path is what every stop in the system depends on. The
run therefore ends by re-reading the broker's own position book rather than
trusting that the sell was submitted.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import time as dtime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kabali.config import RUNS_DIR, load_config                       # noqa: E402
from kabali.data.ingest import candidate_instruments                  # noqa: E402
from kabali.core import ensure_core_importable                        # noqa: E402
from kabali.execution.broker import BUY, SELL, OrderIntent            # noqa: E402
from kabali.execution.dhan_live import (                              # noqa: E402
    DhanBroker, SmokeAuthorization, SmokeRefused,
)

log = logging.getLogger("smoke")

#: Well inside the session. Not at the open, where spreads are widest and a probe
#: would measure the auction rather than the market; not near the close, where a
#: failed exit has no time left to retry.
WINDOW_OPEN, WINDOW_SHUT = dtime(9, 30), dtime(14, 30)

#: Seconds to wait before reconciling. A MARKET order that is accepted is not yet
#: filled, and reading the average traded price too early returns nothing.
SETTLE_SECONDS = 3


def _refuse(msg: str) -> int:
    print(f"REFUSED: {msg}")
    return 2


def _quote(inst) -> float | None:
    """Last traded price from the read-only market feed.

    Deliberately not routed through the broker: the price is needed BEFORE the
    order surface is constructed, so that a bounds check can refuse a name that
    is too expensive without ever building something that can trade.
    """
    import requests                                                  # noqa: PLC0415

    from research.config import load_credentials                     # noqa: PLC0415

    c = load_credentials()
    try:
        r = requests.post(
            "https://api.dhan.co/v2/marketfeed/ltp",
            headers={"access-token": c.access_token, "client-id": c.client_id,
                     "Content-Type": "application/json", "Accept": "application/json"},
            json={inst.exchange_segment: [int(inst.security_id)]}, timeout=15)
        if r.status_code != 200:
            log.error("quote failed: %s %s", r.status_code, r.text[:120])
            return None
        seg = (r.json().get("data") or {}).get(inst.exchange_segment) or {}
        return float(seg[str(inst.security_id)]["last_price"])
    except Exception as exc:                                         # noqa: BLE001
        log.error("quote failed: %s", exc)
        return None


def _leg(broker, inst, side, qty, decision_price, note) -> dict:
    intent = OrderIntent(
        symbol=inst.symbol, security_id=inst.security_id, side=side, quantity=qty,
        reference_price=decision_price, reason=f"smoke:{note}",
        at=pd.Timestamp.now(), decision_price=decision_price,
    )
    fill = broker.submit(intent)
    if not fill.ok:
        return {"side": side, "status": "rejected", "note": fill.note,
                "decision": decision_price}
    time.sleep(SETTLE_SECONDS)
    fill = broker.reconcile(fill)
    adverse = (fill.price - decision_price if side == BUY
               else decision_price - fill.price)
    return {"side": side, "status": fill.status, "order_id": fill.order_id,
            "decision": round(decision_price, 2), "filled": round(fill.price, 2),
            "adverse_rs": round(adverse, 4),
            "adverse_bps": round(adverse / decision_price * 10_000.0, 2)
            if decision_price else None,
            "note": fill.note}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default=None, help="override the authorised symbol")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve, quote and check bounds; place nothing")
    ap.add_argument("--ignore-clock", action="store_true",
                    help="skip the market-hours check (dry runs only)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(args.config)
    ensure_core_importable()

    try:
        auth = SmokeAuthorization.load()
    except SmokeRefused as exc:
        return _refuse(str(exc))

    symbol = (args.symbol or auth.symbol or "").upper()
    if not symbol:
        return _refuse("no symbol: set one in state/SMOKE_TEST.json or pass --symbol")
    if auth.symbol and symbol != auth.symbol:
        return _refuse(f"authorised for {auth.symbol}, not {symbol}")

    now = pd.Timestamp.now()
    if not args.ignore_clock and not (WINDOW_OPEN <= now.time() <= WINDOW_SHUT):
        return _refuse(
            f"outside the probe window {WINDOW_OPEN}-{WINDOW_SHUT} (now {now.time()}). "
            f"Not at the open, where the spread measures the auction; not near the "
            f"close, where a failed exit has no time to retry.")
    if args.ignore_clock and not args.dry_run:
        return _refuse("--ignore-clock is for dry runs only")

    pool = {i.symbol: i for i in candidate_instruments(series=cfg.universe.candidate_series)}
    inst = pool.get(symbol)
    if inst is None:
        return _refuse(f"{symbol} does not resolve against the scrip master")

    ltp = _quote(inst)
    if not ltp or ltp <= 0:
        return _refuse(f"no live price for {symbol}")
    notional = ltp * auth.max_quantity
    if notional > auth.max_notional:
        return _refuse(f"{symbol} at Rs {ltp:,.2f} exceeds the authorised "
                       f"Rs {auth.max_notional:,.0f} notional")

    print("=" * 70)
    print(f"LIVE ORDER-PATH PROBE   {symbol}  x{auth.max_quantity}")
    print("=" * 70)
    print(f"  last price        Rs {ltp:,.2f}")
    print(f"  notional at risk  Rs {notional:,.2f}")
    print(f"  authorised        {auth.max_orders} orders, Rs {auth.max_notional:,.0f} cap")
    print("  worst case        the spread plus two lots of brokerage")
    print()

    if args.dry_run:
        print("DRY RUN -- nothing was sent. Bounds check passed.")
        return 0

    broker = DhanBroker.for_smoke_test(cfg, auth)
    legs, entry = [], None
    try:
        entry = _leg(broker, inst, BUY, auth.max_quantity, ltp, "entry")
        legs.append(entry)
        print(f"  BUY  decision Rs {entry['decision']} -> "
              f"{entry.get('filled', 'REJECTED')}  ({entry['status']})")
    finally:
        # The exit runs even if the entry raised. If the entry never filled this
        # is refused by the bounds check and costs nothing; if it did, this is
        # the only thing standing between a probe and an unmanaged position.
        if entry and entry.get("status") == "filled":
            exit_ref = _quote(inst) or entry["filled"]
            ex = _leg(broker, inst, SELL, auth.max_quantity, exit_ref, "exit")
            legs.append(ex)
            print(f"  SELL decision Rs {ex['decision']} -> "
                  f"{ex.get('filled', 'REJECTED')}  ({ex['status']})")

    # Verify flat from the broker's own book, not from our belief about it.
    flat, holding = True, None
    try:
        pos = broker.api.get_positions()
        rows = (pos or {}).get("data") or []
        for r in rows if isinstance(rows, list) else []:
            if str(r.get("tradingSymbol", "")).upper().startswith(symbol) \
                    and int(r.get("netQty", 0) or 0) != 0:
                flat, holding = False, r
    except Exception as exc:                                          # noqa: BLE001
        flat, holding = False, {"error": str(exc)}

    print()
    if flat:
        print("  position verified FLAT")
    else:
        print("  *** POSITION NOT FLAT -- CLOSE IT MANUALLY NOW ***")
        print(f"      {holding}")

    filled = [x for x in legs if x.get("status") == "filled"]
    if filled:
        tot = sum(x["adverse_rs"] for x in filled)
        print()
        print(f"  real slippage vs decision price: Rs {tot:.4f} over {len(filled)} leg(s)")
        for x in filled:
            print(f"    {x['side']:4s} {x['adverse_bps']:+7.2f} bps")
        print(f"  modelled assumption: {cfg.costs.slippage_bps:.1f} bps per side")

    out = RUNS_DIR / f"smoke_{now:%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "at": now.isoformat(timespec="seconds"), "symbol": symbol,
        "quantity": auth.max_quantity, "reference_ltp": ltp,
        "legs": legs, "flat": flat, "holding": holding,
        "modelled_slippage_bps": cfg.costs.slippage_bps,
        "slippage_source": "broker",
    }, indent=2, default=str), encoding="utf-8")
    print(f"\n  wrote {out}")
    print("\nThis probes the order path only. The live gate is unchanged and still closed.")
    return 0 if flat else 1


if __name__ == "__main__":
    raise SystemExit(main())
