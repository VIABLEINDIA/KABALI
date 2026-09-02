"""Run Stage 2 of the pre-registered cross-sectional test.

    python scripts/run_stage2.py                     # primary specification
    python scripts/run_stage2.py --spread            # the robustness distribution

Stage 2 runs only because Stage 1 passed. The primary specification -- ten
positions, monthly rebalance, Rs 40,000 -- is the default already in
`XSecConfig`, fixed before this panel existed and used by the first
cross-sectional run. It is the primary for that reason.

`--spread` reports the position-count and capital distribution. It is a
distribution, NOT a menu. `docs/xsection_hypothesis.md` fixed the rule in
advance: if the primary fails and a variant passes, the hypothesis has failed
and the variant is a new hypothesis needing its own sample.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kabali.xsection.stage2 import (PRIMARY_POSITIONS,  # noqa: E402
                                    PRIMARY_REBALANCE, run_stage2)

STATE = Path("state")


def load(prefix: str):
    paths = {k: STATE / f"{prefix}_{k}.parquet"
             for k in ("close", "open", "turnover")}
    for p in paths.values():
        if not p.exists():
            raise SystemExit(f"missing {p}\nrun: python scripts/build_bhavcopy_panel.py")
    return tuple(pd.read_parquet(p).astype("float64") for p in paths.values())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="bhav")
    ap.add_argument("--capital", type=float, default=40_000.0)
    ap.add_argument("--n-positions", type=int, default=PRIMARY_POSITIONS)
    ap.add_argument("--rebalance", type=int, default=PRIMARY_REBALANCE)
    ap.add_argument("--spread", action="store_true",
                    help="report the robustness distribution as well")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    close, open_, turnover = load(args.prefix)
    print(f"panel {close.shape[0]:,} sessions x {close.shape[1]:,} entities")
    print(f"PRIMARY: N={args.n_positions}, rebalance {args.rebalance}d, "
          f"capital Rs {args.capital:,.0f}")
    print()
    res = run_stage2(close, open_, turnover, n_positions=args.n_positions,
                     rebalance_days=args.rebalance, capital=args.capital)
    print(res.render())

    if args.spread:
        print()
        print("robustness distribution -- reported whole, never selected from:")
        print(f"{'capital':>10} {'N':>4} {'CAGR':>8} {'Sharpe':>7} {'alpha':>9} "
              f"{'t':>6} {'costs':>8}  verdict")
        for cap in (args.capital, 1_000_000.0):
            for n in (5, 8, 10, 20):
                r = run_stage2(close, open_, turnover, n_positions=n,
                               rebalance_days=args.rebalance, capital=cap)
                print(f"{cap:>10,.0f} {n:>4} {r.strategy.cagr_pct:>7.2f}% "
                      f"{r.strategy.sharpe:>7.2f} {r.alpha_annual_pct:>+8.2f}% "
                      f"{r.t_alpha:>+6.2f} {r.cost_pct_capital:>7.1f}%  "
                      f"{'PASS' if r.passed() else 'FAIL'}")
        print()
        print("A variant passing while the primary fails is a FAILED hypothesis, "
              "not a result.")
    return 0 if res.passed() else 1


if __name__ == "__main__":
    raise SystemExit(main())
