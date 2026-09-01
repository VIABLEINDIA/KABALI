"""Train and score the neural cross-sectional model, against its controls.

    python scripts/run_neural.py --controls
    python scripts/run_neural.py --out runs/neural

Runs the same pipeline three times: on the real panel, on a synthetic panel with
a momentum effect planted in it, and on pure noise. The real number means nothing
without the other two.

  If the PLANTED panel scores near zero, the model cannot see an edge that is
  definitely there, and any negative on real data is a statement about the model.

  If the NOISE panel scores well, the model manufactures skill from nothing, and
  any positive on real data is a statement about overfitting.

Only when planted is clearly positive and noise is clearly not does the real
number carry information.
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

from kabali.config import STATE_DIR                                   # noqa: E402
from kabali.xsection import controls                                  # noqa: E402
from kabali.xsection.neural import (                                  # noqa: E402
    build_features, walk_forward,
)

log = logging.getLogger("neural")


def _panels():
    close = pd.read_parquet(STATE_DIR / "panel_close.parquet")
    volume = pd.read_parquet(STATE_DIR / "panel_volume.parquet")
    return close, volume


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controls", action="store_true",
                    help="also run the planted-edge and noise panels")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/neural")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    close, volume = _panels()
    print(f"panel {close.shape[0]} sessions x {close.shape[1]} names "
          f"({close.index.min().date()} .. {close.index.max().date()})")
    print()

    reports = {}
    t0 = time.monotonic()

    x, y = build_features(close, volume)
    print(f"real panel: {len(x):,} name-days, {x.shape[1]} features")
    reports["real"] = walk_forward(x, y, "REAL PANEL", n_folds=args.folds, seed=args.seed)
    print(reports["real"].render())
    print()

    if args.controls:
        planted = controls.synthetic_panel(n_symbols=close.shape[1],
                                           n_days=close.shape[0], seed=7,
                                           momentum_strength=0.35)
        pc = planted[0] if isinstance(planted, tuple) else planted
        xp, yp = build_features(pc)
        print(f"planted-edge control: {len(xp):,} name-days")
        reports["planted"] = walk_forward(xp, yp, "PLANTED EDGE (must be positive)",
                                          n_folds=args.folds, seed=args.seed)
        print(reports["planted"].render())
        print()

        noise = controls.synthetic_panel(n_symbols=close.shape[1],
                                         n_days=close.shape[0], seed=13,
                                         momentum_strength=0.0)
        nc = noise[0] if isinstance(noise, tuple) else noise
        xn, yn = build_features(nc)
        print(f"noise control: {len(xn):,} name-days")
        reports["noise"] = walk_forward(xn, yn, "PURE NOISE (must be ~zero)",
                                        n_folds=args.folds, seed=args.seed)
        print(reports["noise"].render())
        print()

    print("=" * 74)
    print("VERDICT")
    print("=" * 74)
    r = reports["real"]
    if "planted" in reports and "noise" in reports:
        p, n = reports["planted"], reports["noise"]
        instrument_ok = p.mean_ic > 0.02
        honest = abs(n.mean_ic) < 0.02
        print(f"  planted edge  mean IC {p.mean_ic:+.4f}  "
              f"{'FOUND' if instrument_ok else 'NOT FOUND -- instrument is blind'}")
        print(f"  pure noise    mean IC {n.mean_ic:+.4f}  "
              f"{'rejected' if honest else 'FITTED -- model manufactures skill'}")
        if not instrument_ok or not honest:
            print()
            print("  The real-panel number below carries no information until the")
            print("  controls pass. Fix the instrument before reading the result.")
    print(f"  REAL PANEL    mean IC {r.mean_ic:+.4f}  t {r.ic_t:+.2f}")
    if r.holdout:
        print(f"  holdout       IC {r.holdout.ic:+.4f}  "
              f"decile spread {r.holdout.spread:+.2%}  (scored once)")
    print()
    print("  An IC near zero means the features do not rank next month's winners.")
    print("  That is a result, not a failure to be tuned away.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps({
        "generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "panel": {"sessions": int(close.shape[0]), "names": int(close.shape[1])},
        "horizon_days": 21,
        "reports": {k: {"mean_ic": v.mean_ic, "ic_t": v.ic_t,
                        "folds": [vars(f) for f in v.folds],
                        "holdout": vars(v.holdout) if v.holdout else None}
                    for k, v in reports.items()},
        "elapsed_min": round((time.monotonic() - t0) / 60, 2),
    }, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {out}/summary.json  ({(time.monotonic() - t0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
