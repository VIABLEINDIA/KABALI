"""Build the daily 250-name intraday universe from five years of Dhan history.

Run:
    python scripts/build_universe.py                 # full build
    python scripts/build_universe.py --limit 200     # smoke test on a subset
    python scripts/build_universe.py --stage1-only   # just refresh liquidity

Two ingest stages, because downloading five years for all ~2,600 EQ names costs
~13,000 requests and most of those names cannot be traded at our size anyway.
Stage 1 fetches a recent window for everything and applies a deliberately LOOSE
turnover prefilter; stage 2 fetches full history only for survivors. The real
gates run afterwards in `selector.apply_gates` against the full-history factors,
so the loose prefilter can only cost recall on borderline names, never admit a
name that should have been rejected.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kabali.config import STATE_DIR, load_config                      # noqa: E402
from kabali.core import DhanBridge, Instrument, load_bars              # noqa: E402
from kabali.data.cache import cached_bars, verify_layout               # noqa: E402
from kabali.data.ingest import (                                      # noqa: E402
    candidate_instruments, date_window, ingest, summarise_liquidity,
)
from kabali.regime.classifier import classify                          # noqa: E402
from kabali.universe.factors import compute_factors, factors_frame     # noqa: E402
from kabali.universe.selector import select_universe                   # noqa: E402

log = logging.getLogger("build_universe")

BENCHMARK = Instrument(symbol="NIFTY", security_id="13",
                       exchange_segment="IDX_I", instrument_type="INDEX")
# Stage-1 prefilter is intentionally slacker than the real gate.
PREFILTER_TURNOVER_FRACTION = 0.5
BREADTH_SAMPLE_CAP = 150


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="cap candidates (smoke test)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--rate", type=float, default=2.5, help="aggregate requests/second")
    ap.add_argument("--stage1-years", type=float, default=1.0)
    ap.add_argument("--stage1-only", action="store_true")
    ap.add_argument("--refresh", action="store_true", help="ignore the bar cache")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    log.info("config: %s", cfg.summary())

    t_start = time.monotonic()
    candidates = candidate_instruments(series=cfg.universe.candidate_series)
    if args.limit:
        candidates = candidates[: args.limit]
    log.info("candidate pool: %d NSE %s-series equities",
             len(candidates), "/".join(cfg.universe.candidate_series))

    # ---------------------------------------------------------- stage 1
    s1_start, s1_end = date_window(args.stage1_years)
    liq_rows: list[dict] = []
    lock = threading.Lock()

    def on_bars(inst, df):
        row = summarise_liquidity(inst, df)
        with lock:
            liq_rows.append(row)

    log.info("stage 1: liquidity screen over %s..%s", s1_start, s1_end)
    rep1 = ingest(candidates, "D", s1_start, s1_end, workers=args.workers,
                  rate_per_second=args.rate, refresh=args.refresh, on_bars=on_bars)
    print(rep1.render())

    liq = pd.DataFrame(liq_rows)
    if liq.empty:
        log.error("stage 1 produced no liquidity rows; aborting")
        return 1
    liq.sort_values("median_turnover", ascending=False).to_csv(
        STATE_DIR / "liquidity_screen.csv", index=False)

    floor = cfg.universe.min_median_turnover * PREFILTER_TURNOVER_FRACTION
    keep = liq[
        (liq["median_turnover"] >= floor)
        & liq["last_close"].between(cfg.universe.min_price, cfg.universe.max_price)
    ]
    log.info("stage 1 prefilter: %d/%d pass turnover >= Rs %.1f cr and price band",
             len(keep), len(liq), floor / 1e7)
    if args.stage1_only:
        log.info("stage1-only requested; stopping after liquidity screen")
        return 0

    survivors = [i for i in candidates if i.symbol in set(keep["symbol"])]

    # ---------------------------------------------------------- stage 2
    full_start, full_end = date_window(cfg.universe.history_years)
    log.info("stage 2: %d survivors, full history %s..%s",
             len(survivors), full_start, full_end)
    rep2 = ingest(survivors, "D", full_start, full_end, workers=args.workers,
                  rate_per_second=args.rate, refresh=args.refresh)
    print(rep2.render())

    # ------------------------------------------------- factors and regime
    if not verify_layout():
        log.error("bar-cache path no longer matches research.datastore; aborting "
                  "rather than building a universe from an empty cache")
        return 1

    bridge = DhanBridge()
    bench = load_bars(bridge, BENCHMARK, "D", full_start, full_end)
    log.info("benchmark %s: %d bars", BENCHMARK.symbol, len(bench))

    # Factors come from the CACHE ONLY -- never `load_bars`, which would re-fetch.
    #
    # This loop used to call `load_bars`, so every symbol whose stage-2 download
    # had failed got re-requested here, one at a time, against an account Dhan
    # was already throttling. A pure-arithmetic step became hours of blocked
    # network calls with no log line explaining it. Missing data must be a
    # counted outcome, not an unbounded wait.
    rows, breadth_sample = [], {}
    uncached = short_history = 0
    for inst in survivors:
        bars = cached_bars(inst, "D", full_start, full_end)
        if bars is None:
            uncached += 1
            continue
        f = compute_factors(inst.symbol, inst.security_id, bars)
        if f is None:
            short_history += 1
            continue
        rows.append(f)
        if len(breadth_sample) < BREADTH_SAMPLE_CAP:
            breadth_sample[inst.symbol] = bars[["timestamp", "close"]].copy()

    log.info("factors computed for %d symbols (%d not cached, %d too little history)",
             len(rows), uncached, short_history)
    if uncached:
        log.warning("%d survivor(s) never downloaded -- re-run to fill them in; "
                    "they are excluded from today's universe", uncached)
    factors = factors_frame(rows)
    if factors.empty:
        log.error("no factor rows; aborting")
        return 1

    regime = classify(bench, cfg, universe_bars=breadth_sample)
    print()
    print(regime.render())
    print()

    sel = select_universe(
        factors, cfg,
        regime_family=regime.family,
        direction=regime.direction,
        as_of=regime.as_of,
        position_notional=cfg.risk.position_notional_cap(cfg.capital),
    )
    print(sel.render(top=15))

    # ------------------------------------------------------------- persist
    stamp = pd.Timestamp(regime.as_of).date().isoformat()
    sel.frame.to_parquet(STATE_DIR / f"universe_scored_{stamp}.parquet")
    sel.selected.to_csv(STATE_DIR / f"universe_{stamp}.csv")
    latest = {
        "as_of": stamp,
        "regime": {
            "label": regime.label, "family": regime.family,
            "direction": regime.direction, "confidence": round(regime.confidence, 4),
            "atr_pct": regime.atr_pct, "atr_percentile": regime.atr_percentile,
            "efficiency": regime.efficiency, "breadth": regime.breadth,
            "reasons": regime.reasons,
        },
        "universe": {
            "size": len(sel.selected),
            "candidates": len(candidates),
            "eligible": sel.gates.survived,
            "symbols": sel.symbols,
        },
        "elapsed_min": round((time.monotonic() - t_start) / 60, 2),
    }
    (STATE_DIR / "universe_latest.json").write_text(json.dumps(latest, indent=2, default=str))
    log.info("wrote universe_latest.json (%d symbols) in %.1f min",
             len(sel.symbols), latest["elapsed_min"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
