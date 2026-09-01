"""Re-fetch cached series whose recorded coverage overstates what they hold.

    python scripts/repair_cache.py --dry-run
    python scripts/repair_cache.py --rate 2.0

`scripts/diagnose.py` finds them; this repairs them. A series qualifies when its
`fetch_range` claims data materially past the last bar it actually contains --
the signature of a fetch whose chunks partly failed and whose metadata then
declared the requested span covered.

Repair is `load_bars(..., refresh=True)`, which is the only path that can shrink
an overstated claim: the normal path unions ranges so a narrow fetch cannot
destroy a broad cache, and that union would preserve exactly the lie being fixed.

SUCCESS IS NOT "IT RAN". A re-fetch can fail the same way that created the gap,
so each repair is verified against the metadata afterwards and reported as fixed,
partial, or still broken. A repair tool that reports success on a silent failure
is worse than none -- it converts a detectable problem into a believed-solved one.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kabali.config import load_config                                # noqa: E402
from kabali.core import DhanBridge, Instrument, load_bars            # noqa: E402
from kabali.data.ingest import RateLimiter, date_window              # noqa: E402

log = logging.getLogger("repair")

GAP_DAYS = 10


def _survey(bars_dir: Path) -> list[dict]:
    """Every NSE equity cache whose claim exceeds its contents."""
    out = []
    for m in bars_dir.glob("NSE_EQ_*.meta.json"):
        try:
            b = json.loads(m.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if b.get("instrument_type") != "EQUITY":
            continue
        audit, fr = b.get("audit") or {}, b.get("fetch_range") or {}
        if not audit.get("last") or not fr.get("end"):
            continue
        claimed, actual = pd.Timestamp(fr["end"]), pd.Timestamp(audit["last"])
        gap = (claimed.normalize() - actual.normalize()).days
        if gap > GAP_DAYS:
            out.append({"symbol": b.get("symbol", m.stem),
                        "security_id": str(b.get("security_id", "")),
                        "timeframe": b.get("timeframe", "D"),
                        "lot_size": b.get("lot_size", 1),
                        "gap": gap, "ends": str(actual.date()), "meta": m})
    out.sort(key=lambda r: -r["gap"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="list, do not fetch")
    ap.add_argument("--rate", type=float, default=2.0, help="requests per second")
    ap.add_argument("--limit", type=int, default=0, help="repair at most N series")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    cfg = load_config(args.config)
    from research.config import BARS_DIR                             # noqa: PLC0415

    broken = _survey(Path(BARS_DIR))
    if args.limit:
        broken = broken[: args.limit]
    if not broken:
        print("no cached series overstates its coverage")
        return 0

    print(f"{len(broken)} series to repair "
          f"({sum(r['gap'] for r in broken):,} bar-days claimed but missing)")
    for r in broken[:10]:
        print(f"  {r['symbol']:14s} {r['timeframe']:3s} ends {r['ends']}  "
              f"{r['gap']:5d}d short")
    if len(broken) > 10:
        print(f"  ... and {len(broken) - 10} more")
    if args.dry_run:
        return 0

    bridge = DhanBridge()
    limiter = RateLimiter(args.rate)
    start, end = date_window(cfg.universe.history_years)
    fixed, partial, failed = [], [], []
    t0 = time.monotonic()

    for n, r in enumerate(broken, 1):
        inst = Instrument(symbol=r["symbol"], security_id=r["security_id"],
                          exchange_segment="NSE_EQ", instrument_type="EQUITY",
                          lot_size=r["lot_size"])
        limiter.acquire()
        try:
            load_bars(bridge, inst, r["timeframe"], start, end, refresh=True)
        except Exception as exc:                                     # noqa: BLE001
            failed.append((r["symbol"], str(exc)[:70]))
            continue

        # Verify against the metadata the repair just wrote, not against the
        # absence of an exception.
        try:
            b = json.loads(Path(r["meta"]).read_text(encoding="utf-8"))
            claimed = pd.Timestamp(b["fetch_range"]["end"])
            actual = pd.Timestamp(b["audit"]["last"])
            gap = (claimed.normalize() - actual.normalize()).days
        except Exception:                                            # noqa: BLE001
            failed.append((r["symbol"], "metadata unreadable after repair"))
            continue

        if gap <= GAP_DAYS:
            fixed.append((r["symbol"], str(actual.date())))
        else:
            partial.append((r["symbol"], gap, str(actual.date())))

        if n % 10 == 0:
            print(f"  {n}/{len(broken)}  ({(time.monotonic() - t0) / 60:.1f} min)")

    print()
    print("=" * 70)
    print(f"repaired  {len(fixed)}")
    print(f"partial   {len(partial)}   (claim now honest, history still short)")
    print(f"failed    {len(failed)}")
    for sym, gap, ends in partial[:10]:
        print(f"  ~ {sym:14s} still {gap}d short, ends {ends}")
    for sym, why in failed[:10]:
        print(f"  x {sym:14s} {why}")
    print()
    print("Re-run scripts/diagnose.py to confirm.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
