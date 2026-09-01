"""Run one forward session -- paper against a live feed, or live if the gate opens.

    python scripts/run_live.py                      # paper, live data, no orders
    python scripts/run_live.py --symbols 60         # narrower scan
    python scripts/run_live.py --live               # requires the gate to PASS

`--live` does not enable live trading. It requests it. The gate is re-evaluated
from `state/paper_record.csv` here and again inside `DhanBroker.__init__`, and
either can refuse. Running without `--live` uses `PaperBroker` against the same
live feed, which is how the paper record that the gate needs gets built.

The session appends its result to `state/paper_record.csv` so the evidence
accumulates across days without any manual bookkeeping.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kabali.config import RUNS_DIR, STATE_DIR, load_config                # noqa: E402
from kabali.core import DhanBridge, Instrument, load_bars                 # noqa: E402
from kabali.data.ingest import RateLimiter, candidate_instruments, date_window  # noqa: E402
from kabali.data.intraday import daily_context                            # noqa: E402
from kabali.engine.live import LiveSession, make_dhan_feed                # noqa: E402
from kabali.execution.gate import GateClosed, LiveGate                    # noqa: E402
from kabali.execution.paper import PaperBroker                            # noqa: E402
from kabali.execution import provenance                                   # noqa: E402
from kabali.regime.classifier import classify                             # noqa: E402
from kabali.strategies.registry import default_registry                   # noqa: E402

log = logging.getLogger("run_live")

BENCHMARK = Instrument(symbol="NIFTY", security_id="13",
                       exchange_segment="IDX_I", instrument_type="INDEX")


def append_record(row: dict, live: bool, cfg) -> None:
    """Append the session to the paper record.

    A LIVE session is written to a separate file. Mixing live results into the
    evidence the gate reads would let a live session's own P&L argue for
    continuing to trade live, which is circular.
    """
    path = STATE_DIR / ("live_record.csv" if live else "paper_record.csv")
    df = pd.DataFrame([row])
    if path.exists():
        prior = pd.read_csv(path)
        prior = prior[prior["trade_date"].astype(str) != str(row["trade_date"])]
        df = pd.concat([prior, df], ignore_index=True)

    if live:
        df.to_csv(path, index=False)
        return
    # A forward paper session extends the gate's evidence, so it must refresh the
    # provenance sidecar too. Writing the rows alone would leave the sidecar
    # describing a shorter record, and the gate would read that disagreement as
    # tampering -- correctly, since an unexplained edit and an un-recorded append
    # are indistinguishable from outside.
    provenance.write(path, df, cfg, source="scripts/run_live.py (forward paper)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", type=int, default=100, help="top-N universe names to scan")
    ap.add_argument("--timeframe", default="5m", choices=["1m", "5m", "15m"])
    ap.add_argument("--poll", type=int, default=0,
                    help="max seconds between feed polls; 0 = wait for the bar boundary")
    ap.add_argument("--rate", type=float, default=2.5)
    ap.add_argument("--live", action="store_true",
                    help="request real orders; the gate may still refuse")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    log.info("config: %s", cfg.summary())

    # ------------------------------------------------------------- the gate
    broker_kind = "paper"
    verdict = None
    if args.live:
        gate = LiveGate(cfg)
        try:
            verdict = gate.require_open(gate.load_record())
        except GateClosed as exc:
            print(exc)
            print()
            print("Refusing to trade live. Run paper sessions until the gate opens.")
            return 2
        broker_kind = "live"
        log.warning("LIVE GATE OPEN -- this session will place real orders")

    # ------------------------------------------------------------- universe
    path = STATE_DIR / "universe_latest.json"
    if not path.exists():
        print(f"no universe at {path} -- run scripts/build_universe.py")
        return 1
    universe = json.loads(path.read_text())["universe"]["symbols"][: args.symbols]

    pool = {i.symbol: i for i in candidate_instruments(series=cfg.universe.candidate_series)}
    instruments = {s: pool[s] for s in universe if s in pool}
    if not instruments:
        print("universe symbols did not resolve against the scrip master")
        return 1

    bridge = DhanBridge()
    d_start, d_end = date_window(cfg.universe.history_years)
    regime = classify(load_bars(bridge, BENCHMARK, "D", d_start, d_end), cfg)
    print(regime.render())
    print()

    active = default_registry().for_family(regime.family)
    if not active:
        log.warning("no strategy is valid in the %s regime -- standing down", regime.family)
        return 0
    log.info("strategies active: %s", ", ".join(s.name for s in active))

    # --------------------------------------------------- per-symbol context
    today = dt.date.today()
    contexts, tradeable = {}, {}
    for sym, inst in instruments.items():
        try:
            daily = load_bars(bridge, inst, "D", d_start, d_end)
        except Exception as exc:                                  # noqa: BLE001
            log.debug("no daily for %s: %s", sym, exc)
            continue
        ctx = daily_context(daily, today)
        if ctx:
            contexts[sym] = ctx
            tradeable[sym] = inst
    log.info("context ready for %d/%d symbols", len(tradeable), len(instruments))
    if not tradeable:
        return 1

    # ------------------------------------------------------------- execute
    if broker_kind == "live":
        from kabali.execution.dhan_live import DhanBroker
        broker = DhanBroker(cfg, verdict)
    else:
        broker = PaperBroker(cfg)
    log.info("broker: %s", broker.describe())

    limiter = RateLimiter(args.rate)
    feed = make_dhan_feed(bridge, tradeable, args.timeframe, limiter=limiter)
    session = LiveSession(cfg, default_registry(), broker, regime,
                          tradeable, contexts, feed, timeframe=args.timeframe)

    result = session.run(trade_date=today, poll_seconds=args.poll)

    print()
    print(result.render())
    row = result.record_row()
    append_record(row, live=(broker_kind == "live"), cfg=cfg)

    out = RUNS_DIR / f"session_{today:%Y%m%d}_{broker_kind}"
    out.mkdir(parents=True, exist_ok=True)
    result.book.trades_frame().to_csv(out / "trades.csv", index=False)
    if hasattr(broker, "fills_frame"):
        broker.fills_frame().to_csv(out / "fills.csv", index=False)
    (out / "session.json").write_text(json.dumps(row, indent=2, default=str))
    log.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
