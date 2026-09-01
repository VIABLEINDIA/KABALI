"""KABALI command line.

    python cli.py health                  connectivity, credentials, gate status
    python cli.py regime                  today's market regime from NIFTY
    python cli.py universe --show 25      show the current 250-name universe
    python cli.py gate                    live-gate status against the paper record
    python cli.py gate --init             write an unarmed gate file template
    python cli.py signals --symbols X,Y   dry-run the strategies on today's bars

Build and replay live in scripts/:
    python scripts/build_universe.py
    python scripts/run_paper.py --days 40 --symbols 120
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kabali.config import STATE_DIR, load_config                      # noqa: E402
from kabali.core import DhanBridge, Instrument, load_bars             # noqa: E402
from kabali.data.ingest import candidate_instruments, date_window     # noqa: E402
from kabali.execution.gate import GateCriteria, LiveGate              # noqa: E402
from kabali.regime.classifier import classify                         # noqa: E402

BENCHMARK = Instrument(symbol="NIFTY", security_id="13",
                       exchange_segment="IDX_I", instrument_type="INDEX")


def cmd_health(args, cfg) -> int:
    print(cfg.summary())
    print()
    bridge = DhanBridge()
    rep = bridge.health_check()
    print(rep.render())
    print()
    gate = LiveGate(cfg)
    verdict = gate.check(gate.load_record())
    print(verdict.render())
    print()
    if cfg.execution.mode == "live" and not verdict.passed:
        print("execution.mode is LIVE but the gate is CLOSED -- the bot will refuse to start.")
    elif cfg.execution.mode == "paper":
        print("execution.mode is PAPER -- no real order can be placed.")
    return 0 if rep.ok else 1


def cmd_regime(args, cfg) -> int:
    bridge = DhanBridge()
    start, end = date_window(cfg.universe.history_years)
    bars = load_bars(bridge, BENCHMARK, "D", start, end)
    state = classify(bars, cfg)
    print(state.render())
    print()
    from kabali.strategies.registry import default_registry
    active = [s.name for s in default_registry().for_family(state.family)]
    print(f"strategies enabled in this regime: {', '.join(active) or 'none'}")
    return 0


def cmd_universe(args, cfg) -> int:
    path = STATE_DIR / "universe_latest.json"
    if not path.exists():
        print(f"no universe yet at {path}\nrun: python scripts/build_universe.py")
        return 1
    data = json.loads(path.read_text())
    r, u = data["regime"], data["universe"]
    print(f"as of {data['as_of']} | regime {r['label']} (confidence {r['confidence']:.0%})")
    print(f"candidates {u['candidates']:,} -> eligible {u['eligible']:,} -> "
          f"selected {u['size']:,}")
    print()
    csv = STATE_DIR / f"universe_{data['as_of']}.csv"
    if csv.exists():
        df = pd.read_csv(csv, index_col=0)
        cols = [c for c in ("rank", "score", "close", "atr_pct", "median_turnover",
                            "ret_63", "participation_pct") if c in df.columns]
        head = df[cols].head(args.show).copy()
        if "median_turnover" in head:
            head["median_turnover"] = (head["median_turnover"] / 1e7).round(1)
            head = head.rename(columns={"median_turnover": "turnover_cr"})
        for c in ("atr_pct", "ret_63"):
            if c in head:
                head[c] = (head[c] * 100).round(2)
        print(head.round(3).to_string())
    else:
        print(", ".join(u["symbols"][: args.show]))
    return 0


def cmd_gate(args, cfg) -> int:
    gate = LiveGate(cfg, GateCriteria())
    if args.init:
        try:
            p = gate.write_template()
        except Exception as exc:                                  # noqa: BLE001
            print(exc)
            return 1
        print(f"wrote unarmed gate template to {p}")
        print("It is NOT armed. Arming requires a paper record that clears every "
              "criterion, plus the confirmation phrase written by hand.")
        return 0
    rec = gate.load_record()
    if rec is None:
        print("no paper record at state/paper_record.csv -- "
              "run scripts/run_paper.py --promote")
    verdict = gate.check(rec)
    print(verdict.render())
    if verdict.evidence:
        print()
        print(json.dumps(verdict.evidence, indent=2, default=str))
    return 0 if verdict.passed else 1


def cmd_signals(args, cfg) -> int:
    """Dry-run the strategy stack on today's bars. Never places an order."""
    from kabali.data.intraday import daily_context, split_sessions
    from kabali.strategies.base import SymbolContext
    from kabali.strategies.registry import default_registry, route

    bridge = DhanBridge()
    d_start, d_end = date_window(cfg.universe.history_years)
    regime = classify(load_bars(bridge, BENCHMARK, "D", d_start, d_end), cfg)
    print(regime.render())
    print()

    if args.symbols:
        wanted = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        path = STATE_DIR / "universe_latest.json"
        if not path.exists():
            print("no universe; pass --symbols or run scripts/build_universe.py")
            return 1
        wanted = json.loads(path.read_text())["universe"]["symbols"][: args.limit]

    pool = {i.symbol: i for i in candidate_instruments(series=cfg.universe.candidate_series)}
    registry = default_registry()
    i_start, i_end = date_window(30 / 365.25)
    found = 0

    for sym in wanted:
        inst = pool.get(sym)
        if inst is None:
            continue
        try:
            intra = load_bars(bridge, inst, "5m", i_start, i_end)
            daily = load_bars(bridge, inst, "D", d_start, d_end)
        except Exception as exc:                                  # noqa: BLE001
            print(f"{sym}: {exc}")
            continue
        sessions = split_sessions(intra, cfg.session)
        if not sessions:
            continue
        day = max(sessions)
        ctx_info = daily_context(daily, day)
        if not ctx_info:
            continue
        frame = sessions[day]
        ctx = SymbolContext(
            symbol=sym, security_id=inst.security_id,
            now=pd.Timestamp(frame["timestamp"].iloc[-1]),
            intraday=frame, daily=ctx_info["daily"], regime=regime,
            session=cfg.session, prev_close=ctx_info["prev_close"],
            atr_daily=ctx_info["atr_daily"], tick_size=inst.tick_size,
        )
        for sig in route(ctx, registry):
            found += 1
            print(f"  [{day}] {sig.describe()}")
            for r in sig.reasons:
                print(f"        - {r}")

    print()
    print(f"{found} signal(s) on the most recent session. Dry run -- nothing was ordered.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(prog="cli.py", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("health", help="connectivity, credentials and gate status")
    sub.add_parser("regime", help="classify today's market regime")

    p_u = sub.add_parser("universe", help="show the current universe")
    p_u.add_argument("--show", type=int, default=20)

    p_g = sub.add_parser("gate", help="live-gate status")
    p_g.add_argument("--init", action="store_true", help="write an unarmed template")

    p_s = sub.add_parser("signals", help="dry-run strategies on the latest session")
    p_s.add_argument("--symbols", default=None, help="comma-separated, else the universe")
    p_s.add_argument("--limit", type=int, default=40)

    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    cfg = load_config(args.config)

    return {
        "health": cmd_health, "regime": cmd_regime, "universe": cmd_universe,
        "gate": cmd_gate, "signals": cmd_signals,
    }[args.command](args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
