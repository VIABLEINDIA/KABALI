"""Run Stage 1 of the pre-registered cross-sectional test.

    python scripts/run_stage1.py                    # the bhavcopy panel
    python scripts/run_stage1.py --prefix bhav
    python scripts/run_stage1.py --pilot            # allow a short sample

THE SAMPLE-LENGTH GUARD IS THE POINT. `docs/xsection_hypothesis.md` measures
that a 100x100 spread needs ~22 years to resolve a 5%/yr effect, and commits to
not running the confirmatory test on anything shorter. Without `--pilot` this
script refuses a panel below 15 years rather than producing a number that would
later be quoted as if it settled something. With `--pilot` it runs and stamps
PILOT ONLY on the output.

Nothing here chooses a parameter. Formation, skip, position count and cadence
all come from the pre-registration and the existing momentum module.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kabali.xsection.stage1 import (MIN_YEARS_FOR_VERDICT, N_SIDE,  # noqa: E402
                                    STEP, run_stage1)

STATE = Path("state")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="bhav", help="panel file prefix in state/")
    ap.add_argument("--n-side", type=int, default=N_SIDE)
    ap.add_argument("--step", type=int, default=STEP)
    ap.add_argument("--pilot", action="store_true",
                    help="run on a sample too short to settle the hypothesis")
    ap.add_argument("--save", default=None, help="write the spread series here")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    close_p = STATE / f"{args.prefix}_close.parquet"
    turn_p = STATE / f"{args.prefix}_turnover.parquet"
    for p in (close_p, turn_p):
        if not p.exists():
            print(f"missing {p}\nrun: python scripts/build_bhavcopy_panel.py")
            return 1

    close = pd.read_parquet(close_p)
    turnover = pd.read_parquet(turn_p)
    years = (close.index[-1] - close.index[0]).days / 365.25
    print(f"panel {close.shape[0]:,} sessions x {close.shape[1]:,} entities | "
          f"{close.index.min().date()} -> {close.index.max().date()} "
          f"({years:.1f} years)")

    if years < MIN_YEARS_FOR_VERDICT and not args.pilot:
        print()
        print(f"REFUSING: {years:.1f} years is below the {MIN_YEARS_FOR_VERDICT:.0f} "
              "the power analysis requires.")
        print("docs/xsection_hypothesis.md commits to not running the "
              "confirmatory test on a sample that cannot answer it.")
        print("Pass --pilot to run anyway; the output will be stamped PILOT ONLY "
              "and cannot report PASS.")
        return 2

    print(f"estimators: {args.n_side}x{args.n_side} spread, "
          f"{args.step}-session formation")
    print()
    res = run_stage1(close, turnover, n_side=args.n_side, step=args.step)
    print(res.render())

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        res.spread.daily.to_csv(out)
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
