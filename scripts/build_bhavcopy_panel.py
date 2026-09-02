"""Build the survivorship-free bhavcopy panel.

    python scripts/build_bhavcopy_panel.py --start 2006-01-01
    python scripts/build_bhavcopy_panel.py --start 2015-07-01 --end 2015-09-30
    python scripts/build_bhavcopy_panel.py --no-fetch          # cache only

The download is the slow half: roughly 250 trading days a year, one request
each, paced to stay under NSE's rate limiting. Twenty years is about an hour on
a first run and seconds afterwards, because every day-file and every corporate
action window is cached. An interrupted pull is resumed by running it again.

Written for `docs/xsection_hypothesis.md`, which measures that the existing
3.9-year panel cannot resolve the cross-sectional momentum hypothesis at any
sample size and names the longer panel as the precondition for testing it.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kabali.data.bhavcopy import fetch_range, load_cached          # noqa: E402
from kabali.data.corpactions import fetch_actions                  # noqa: E402
from kabali.data.panel import (build_panel, mask_unexplained_jumps,  # noqa: E402
                               suspicious_entities, write_panel)

log = logging.getLogger("build_panel")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2006-01-01")
    ap.add_argument("--end", default=None, help="default: today")
    ap.add_argument("--no-fetch", action="store_true",
                    help="build from the cache without downloading")
    ap.add_argument("--no-mask", action="store_true",
                    help="keep unexplained jumps instead of quarantining them")
    ap.add_argument("--prefix", default="bhav")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    start = pd.Timestamp(args.start).date()
    end = pd.Timestamp(args.end).date() if args.end else date.today()

    if not args.no_fetch:
        print(f"fetching bhavcopy {start} -> {end} (cached days cost nothing)")
        hit, miss = fetch_range(start, end)
        print(f"  {hit:,} trading days, {miss:,} non-trading")

    long = load_cached(start, end)
    print(f"loaded {len(long):,} rows, "
          f"{long['date'].min().date()} -> {long['date'].max().date()}")

    # Reach a quarter beyond the price window: an action just outside it still
    # needs its factor to adjust prices inside it.
    print("fetching corporate actions")
    actions = fetch_actions(start - timedelta(days=95),
                            end + timedelta(days=95))
    print(f"  {len(actions):,} actions, "
          f"{int(actions['factor'].notna().sum()):,} priceable")

    panel = build_panel(long, actions)
    print()
    print(panel.entities.render())
    if panel.confirmation is not None:
        print(panel.confirmation.render())

    if not args.no_mask:
        panel, jumps = mask_unexplained_jumps(panel)
        print(f"quarantined {len(jumps):,} entities with unexplained jumps "
              f"(history before the jump blanked)")
        if len(jumps):
            print(jumps.head(8).to_string(index=False))

    susp = suspicious_entities(long, panel.entities)
    print(f"{len(susp):,} entities contain a gap of 180+ days "
          "(suspensions and possible over-merges)")

    print()
    print(panel.render())
    written = write_panel(panel, prefix=args.prefix)
    for p in written:
        print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
