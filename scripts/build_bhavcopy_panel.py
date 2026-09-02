"""Build the survivorship-free bhavcopy panel.

    python scripts/build_bhavcopy_panel.py --start 2006-01-01
    python scripts/build_bhavcopy_panel.py --start 2015-07-01 --end 2015-09-30
    python scripts/build_bhavcopy_panel.py --no-fetch          # cache only

The download is the slow half: roughly 250 trading days a year, one request
each, paced to stay under NSE's rate limiting. Twenty years is about an hour on
a first run and seconds afterwards, because every day-file and every corporate
action window is cached. An interrupted pull is resumed by running it again.

The build half streams. Holding twenty years as one long frame plus four wide
panels needs ~2GB and would fail on a small machine, so each field is filled,
adjusted, written and released before the next begins -- peak memory is one
panel rather than five. Verified to reproduce the in-memory builder exactly, to
float32 precision, on the 2015 Q3 slice.

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

from kabali.data.bhavcopy import CACHE_DIR, fetch_range                 # noqa: E402
from kabali.data.corpactions import fetch_actions                       # noqa: E402
from kabali.data.panel import (build_panel_streaming,                   # noqa: E402
                               suspicious_from_panel)

log = logging.getLogger("build_panel")


def cached_files(start: date, end: date, cache_dir: Path) -> list[Path]:
    out = []
    for f in sorted(Path(cache_dir).glob("*.parquet")):
        try:
            d = pd.Timestamp(f.stem).date()
        except ValueError:
            continue
        if start <= d <= end:
            out.append(f)
    return out


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

    files = cached_files(start, end, CACHE_DIR)
    if not files:
        print(f"no cached day-files in {CACHE_DIR} for that range")
        return 1
    print(f"{len(files):,} cached day-files, "
          f"{pd.Timestamp(files[0].stem).date()} -> {pd.Timestamp(files[-1].stem).date()}")

    # Reach a quarter beyond the price window: an action just outside it still
    # needs its factor to adjust prices inside it.
    print("fetching corporate actions")
    actions = fetch_actions(start - timedelta(days=95), end + timedelta(days=95))
    print(f"  {len(actions):,} actions, "
          f"{int(actions['factor'].notna().sum()):,} priceable")

    print("building panels (one field at a time)")
    ents, conf, written = build_panel_streaming(
        files, actions, prefix=args.prefix, mask_jumps=not args.no_mask)

    print()
    print(ents.render())
    if conf is not None:
        print(conf.render())
    print(f"quarantined {written.get('quarantined', 0):,} entities with "
          "unexplained jumps (history before the jump blanked)")

    close = pd.read_parquet(written["close"])
    listed = close.notna().sum(axis=1)
    print(f"{suspicious_from_panel(close):,} entities contain a gap of 180+ days "
          "(suspensions and possible over-merges)")
    print()
    print(f"panel {close.shape[0]:,} sessions x {close.shape[1]:,} entities | "
          f"{close.index.min().date()} -> {close.index.max().date()} | "
          f"listed names {listed.iloc[0]:,} at start, {listed.iloc[-1]:,} at end")
    for k, v in written.items():
        if k != "quarantined":
            print(f"  wrote {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
