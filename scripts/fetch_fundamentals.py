"""Populate the fundamentals cache for the universe.

    python scripts/fetch_fundamentals.py --source json records.json
    python scripts/fetch_fundamentals.py --source api --limit 50
    python scripts/fetch_fundamentals.py --dry-run

TWO SOURCES, BECAUSE ONE OF THEM IS NOT VERIFIED YET
====================================================
`json` reads records from a file you already have. It works today, needs no
credentials, and is the path to use if you can export from Bull AI's web UI or
any other provider.

`api` calls Bull AI's REST endpoint. THE ENDPOINT AND RESPONSE SHAPE IN
`_resolve_api` ARE A BEST GUESS FROM THE MCP SCHEMAS AND HAVE NOT BEEN RUN
AGAINST THE REAL SERVICE. They are isolated in one function, marked, and will
fail loudly rather than quietly returning nothing -- but do not assume they are
right until a live call has proved it. Everything else in this script is
source-agnostic and already exercised.

THE SYMBOL MUST MATCH
=====================
A company search resolves free text and will return a DIFFERENT company rather
than nothing. During manual population "MANINDS Man Industries India" resolved to
Solar Industries: a Rs 183,000 crore explosives maker under a Rs 1,000 crore pipe
manufacturer's symbol, in a well-formed record with nothing marking it wrong.

So a result is accepted only when its NSE code equals the symbol requested.
Everything else is reported as a mismatch and dropped. A missing record shows up
in the coverage report; a wrong one ranks and filters exactly like a right one,
and would be found only by someone who already suspected it.

WRITES ARE CHECKPOINTED
=======================
The store is saved every `--checkpoint` symbols, so a run interrupted at symbol
180 keeps the first 179. A bulk fetch that loses everything on a timeout is a
bulk fetch nobody runs twice.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kabali.config import STATE_DIR, load_config                      # noqa: E402
from kabali.fundamentals.store import (                               # noqa: E402
    Fundamentals, FundamentalsStore, validate_against_price,
)

log = logging.getLogger("fundamentals")

#: Best guess from the MCP schema namespace. VERIFY before trusting.
API_BASE = os.environ.get("BULLAI_API_BASE", "https://api.bull-ai.in/v1")


class Unresolved(RuntimeError):
    """The symbol could not be resolved to a matching company."""


# --------------------------------------------------------------- resolvers

def _record(row: dict, symbol: str, source: str) -> Fundamentals:
    """Map a provider row onto a Fundamentals record, enforcing the symbol match."""
    got = str(row.get("nse_code") or row.get("symbol") or "").upper()
    if got != symbol.upper():
        raise Unresolved(f"resolved to {got or '<none>'} ({row.get('full_name', '?')})")
    cap = row.get("market_cap")
    return Fundamentals(
        symbol=symbol.upper(),
        full_name=str(row.get("full_name") or row.get("name") or ""),
        isin=str(row.get("isin") or ""),
        bse_code=str(row.get("bse_code") or ""),
        market_cap_cr=float(cap) if cap not in (None, "") else None,
        category=str(row.get("category") or ""),
        sector=str(row.get("sector") or ""),
        industry=str(row.get("industry") or ""),
        fetched_at=pd.Timestamp.now().isoformat(timespec="seconds"),
        source=source,
    )


def _resolve_json(symbol: str, blob: dict) -> Fundamentals:
    row = blob.get(symbol.upper())
    if not row:
        raise Unresolved("not present in the supplied file")
    return _record(row, symbol, "file")


def _resolve_api(symbol: str, session, key: str) -> Fundamentals:
    """UNVERIFIED. The endpoint and response shape are inferred, not confirmed.

    Isolated here so that correcting it is a one-function change, and so that a
    wrong guess fails as an exception rather than as an empty cache that looks
    like the universe simply has no fundamentals.
    """
    r = session.get(
        f"{API_BASE}/companies/search",
        headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
        params={"query": symbol, "limit": 3}, timeout=20,
    )
    if r.status_code == 401:
        raise Unresolved("401 unauthorised -- check BULLAI_API_KEY")
    if r.status_code != 200:
        raise Unresolved(f"HTTP {r.status_code}: {r.text[:120]}")
    try:
        rows = (r.json() or {}).get("results") or []
    except ValueError as exc:
        raise Unresolved(f"response was not JSON: {exc}") from exc
    if not rows:
        raise Unresolved("no results")
    # Try every candidate; the exact-symbol rule decides, not the ranking.
    last = None
    for row in rows:
        try:
            return _record(row, symbol, "bull-ai")
        except Unresolved as exc:
            last = exc
    raise last or Unresolved("no candidate matched the symbol")


# -------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["json", "api"], default="api")
    ap.add_argument("path", nargs="?", help="records file, when --source json")
    ap.add_argument("--limit", type=int, default=0, help="stop after N symbols")
    ap.add_argument("--rate", type=float, default=4.0, help="requests per second")
    ap.add_argument("--refresh", action="store_true",
                    help="re-fetch symbols already present")
    ap.add_argument("--checkpoint", type=int, default=25,
                    help="save the store every N symbols")
    ap.add_argument("--dry-run", action="store_true", help="list work, fetch nothing")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_config(args.config)          # validates config before any network call

    upath = STATE_DIR / "universe_latest.json"
    if not upath.exists():
        print(f"no universe at {upath} -- run scripts/build_universe.py")
        return 1
    universe = json.loads(upath.read_text())["universe"]["symbols"]

    store = FundamentalsStore()
    todo = universe if args.refresh else [s for s in universe if not store.get(s)]
    if args.limit:
        todo = todo[: args.limit]

    print(f"universe {len(universe)} | cached {len(store)} | to fetch {len(todo)}")
    if args.dry_run:
        print("DRY RUN -- nothing fetched")
        print("  " + ", ".join(todo[:20]) + (" ..." if len(todo) > 20 else ""))
        return 0
    if not todo:
        print(store.render(universe))
        return 0

    # Resolver setup.
    if args.source == "json":
        if not args.path:
            print("--source json needs a records file")
            return 2
        blob = json.loads(Path(args.path).read_text(encoding="utf-8"))
        blob = {k.upper(): v for k, v in blob.items()}
        resolve = lambda s: _resolve_json(s, blob)                   # noqa: E731
    else:
        key = os.environ.get("BULLAI_API_KEY", "").strip()
        if not key:
            print("BULLAI_API_KEY is not set.\n"
                  "  Set it, or use --source json with an exported records file.\n"
                  "  Note: the REST endpoint in _resolve_api is inferred from the MCP\n"
                  "  schemas and has not been verified against the live service.")
            return 2
        import requests                                              # noqa: PLC0415
        session = requests.Session()
        resolve = lambda s: _resolve_api(s, session, key)            # noqa: E731

    ok, mismatched, failed = [], [], []
    t0 = time.monotonic()
    interval = 1.0 / max(args.rate, 0.1)

    for n, sym in enumerate(todo, 1):
        try:
            store.merge([resolve(sym)])
            ok.append(sym)
        except Unresolved as exc:
            mismatched.append((sym, str(exc)))
        except Exception as exc:                                     # noqa: BLE001
            failed.append((sym, f"{type(exc).__name__}: {exc}"))
        if n % args.checkpoint == 0:
            store.save()
            print(f"  {n}/{len(todo)}  ({(time.monotonic() - t0) / 60:.1f} min, "
                  f"{len(ok)} ok, {len(mismatched)} unmatched)")
        if args.source == "api":
            time.sleep(interval)

    store.save()

    print()
    print("=" * 70)
    print(f"resolved    {len(ok)}")
    print(f"unmatched   {len(mismatched)}   (dropped, not guessed)")
    print(f"errored     {len(failed)}")
    for sym, why in mismatched[:12]:
        print(f"  ~ {sym:14s} {why}")
    for sym, why in failed[:12]:
        print(f"  x {sym:14s} {why}")
    print()
    print(store.render(universe))

    # Cross-check every cap against price. A wrong number is invisible otherwise.
    ucsv = sorted(STATE_DIR.glob("universe_2*.csv"))
    if ucsv:
        prices = pd.read_csv(ucsv[-1], index_col=0)["close"].to_dict()
        flags = validate_against_price(store, prices)
        print(f"\nprice cross-validation: {len(flags)} flag(s)")
        for f in flags:
            print("  !", f)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
