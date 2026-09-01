"""Replay historical sessions through the live engine in paper mode.

Run:
    python scripts/run_paper.py --days 40 --symbols 120
    python scripts/run_paper.py --days 60 --symbols 250 --out runs/paper_full
    python scripts/run_paper.py --days 60 --symbols 250 --promote

A run writes only into its own `runs/` directory unless `--promote` is given.
Promotion replaces `state/paper_record.csv`, which is the evidence the live gate
scores, so it is a deliberate act rather than a side effect of any replay: a
two-day smoke test must not be able to overwrite a sixty-four-session record.
Promotion also writes the provenance sidecar the gate verifies.

This is the bot's forward test run backwards: identical engine, identical risk
limits, identical cost model, driven from stored 5-minute bars instead of a live
feed. It produces the paper record the live gate is evaluated against.

Three things are recomputed AS OF each session date, never once up front, because
computing them once would leak the future into every earlier day:

  REGIME     classified from benchmark bars strictly before the session
  UNIVERSE   re-scored from daily factors strictly before the session
  CONTEXT    prev_close / ATR / median volume from bars strictly before it

The result is a genuine walk-forward: on every simulated day the bot knows only
what it could have known that morning. It is still NOT a substitute for forward
paper trading on live fills -- the gate keeps a separate slippage-fidelity check
for exactly that reason -- but it is the difference between a walk-forward and a
backtest that quietly knew the answer.
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

from kabali.config import ROOT, RUNS_DIR, STATE_DIR, load_config           # noqa: E402
from kabali.core import DhanBridge, Instrument, load_bars                  # noqa: E402
from kabali.data.ingest import RateLimiter, candidate_instruments, date_window  # noqa: E402
from kabali.data.intraday import daily_context, split_sessions             # noqa: E402
from kabali.engine.session import SessionEngine                            # noqa: E402
from kabali.execution.gate import GateCriteria, evaluate                   # noqa: E402
from kabali.execution.paper import PaperBroker                             # noqa: E402
from kabali.execution import provenance                                    # noqa: E402
from kabali.regime.classifier import classify                              # noqa: E402
from kabali.strategies.registry import default_registry                    # noqa: E402
from kabali.universe.factors import compute_factors, factors_frame         # noqa: E402
from kabali.universe.selector import select_universe                       # noqa: E402

log = logging.getLogger("run_paper")

BENCHMARK = Instrument(symbol="NIFTY", security_id="13",
                       exchange_segment="IDX_I", instrument_type="INDEX")


def _group_reasons(failed: dict[str, str]) -> dict[str, list[str]]:
    """Bucket failures by their message, so one cause is reported once."""
    out: dict[str, list[str]] = {}
    for sym, reason in failed.items():
        key = reason.split(" Chunk errors:")[0].strip()
        out.setdefault(key, []).append(sym)
    return dict(sorted(out.items(), key=lambda kv: -len(kv[1])))


def _top_reason(failed: dict[str, str]) -> str:
    groups = _group_reasons(failed)
    return next(iter(groups), "unknown") if groups else "unknown"


def load_universe_symbols(limit: int) -> list[str]:
    path = STATE_DIR / "universe_latest.json"
    if not path.exists():
        raise SystemExit(
            f"{path} not found -- run scripts/build_universe.py first")
    data = json.loads(path.read_text())
    syms = data["universe"]["symbols"]
    return syms[:limit] if limit else syms


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=40, help="calendar days of intraday history")
    ap.add_argument("--symbols", type=int, default=120, help="top-N universe names to scan")
    ap.add_argument("--timeframe", default="5m", choices=["1m", "5m", "15m"])
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--rate", type=float, default=2.5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--promote", action="store_true",
                    help="replace state/paper_record.csv -- the evidence the live "
                         "gate scores -- with this run's sessions")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    log.info("config: %s", cfg.summary())

    out_dir = Path(args.out) if args.out else RUNS_DIR / f"paper_{pd.Timestamp.now():%Y%m%d_%H%M}"
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = load_universe_symbols(args.symbols)
    all_inst = {i.symbol: i for i in candidate_instruments(series=cfg.universe.candidate_series)}
    instruments = {s: all_inst[s] for s in symbols if s in all_inst}
    log.info("scanning %d symbols over %d days at %s",
             len(instruments), args.days, args.timeframe)

    bridge = DhanBridge()
    intraday_start, intraday_end = date_window(args.days / 365.25)
    daily_start, daily_end = date_window(cfg.universe.history_years)

    # ------------------------------------------------------- fetch intraday
    #
    # Failures are RETRIED and then REPORTED BY REASON. Both matter, and the
    # first version of this loop did neither: it counted failures and moved on,
    # so a run that lost 53 of 120 symbols to transient rate-limiting logged the
    # same "59 failed" as one where the data genuinely did not exist. The two
    # demand opposite responses -- wait and retry, versus drop the symbol -- and
    # a bare count cannot tell them apart. It silently halved the universe the
    # result was computed on.
    t0 = time.monotonic()
    sessions_by_symbol, daily_by_symbol = {}, {}
    failed: dict[str, str] = {}

    def fetch_batch(items, limiter, pass_name):
        for n, (sym, inst) in enumerate(items, 1):
            try:
                limiter.acquire()
                intra = load_bars(bridge, inst, args.timeframe, intraday_start, intraday_end)
                daily = load_bars(bridge, inst, "D", daily_start, daily_end)
            except Exception as exc:                              # noqa: BLE001
                failed[sym] = f"{type(exc).__name__}: {exc}"
                continue
            sess = split_sessions(intra, cfg.session)
            if not sess:
                failed[sym] = "no complete sessions in the window"
                continue
            sessions_by_symbol[sym] = sess
            daily_by_symbol[sym] = daily
            failed.pop(sym, None)
            if n % 25 == 0:
                log.info("%s %d/%d (%.1f min)", pass_name, n, len(items),
                         (time.monotonic() - t0) / 60)

    limiter = RateLimiter(args.rate)
    fetch_batch(list(instruments.items()), limiter, "intraday")

    if failed:
        retry = [(s, instruments[s]) for s in list(failed) if s in instruments]
        log.warning("retrying %d symbol(s) at half rate after: %s",
                    len(retry), _top_reason(failed))
        fetch_batch(retry, RateLimiter(max(args.rate / 2.0, 0.5)), "retry")

    log.info("intraday ready for %d/%d symbols in %.1f min",
             len(sessions_by_symbol), len(instruments), (time.monotonic() - t0) / 60)
    if failed:
        log.warning("%d symbol(s) still unavailable; reasons:", len(failed))
        for reason, syms in _group_reasons(failed).items():
            log.warning("    %3d  %s  (e.g. %s)", len(syms), reason[:90], ", ".join(syms[:4]))
    if not sessions_by_symbol:
        log.error("no intraday data; aborting")
        return 1
    coverage = len(sessions_by_symbol) / max(len(instruments), 1)
    if coverage < 0.8:
        log.warning("only %.0f%% of the requested universe was scanned -- the result "
                    "below describes that subset, not the full universe", coverage * 100)

    bench_daily = load_bars(bridge, BENCHMARK, "D", daily_start, daily_end)

    trade_dates = sorted({d for s in sessions_by_symbol.values() for d in s})
    log.info("replaying %d trading sessions: %s .. %s",
             len(trade_dates), trade_dates[0], trade_dates[-1])

    registry = default_registry()
    records, all_trades = [], []

    # ------------------------------------------------------------- replay
    for day in trade_dates:
        # Regime as of the morning of `day`: bars strictly before it.
        prior = bench_daily[pd.to_datetime(bench_daily["timestamp"]).dt.date < day]
        if len(prior) < cfg.regime.trend_anchor + 5:
            continue
        try:
            regime = classify(prior, cfg)
        except ValueError as exc:
            log.debug("skip %s: %s", day, exc)
            continue

        # Universe as of the same morning.
        factor_rows = []
        contexts = {}
        for sym, sess in sessions_by_symbol.items():
            if day not in sess:
                continue
            ctx = daily_context(daily_by_symbol[sym], day)
            if not ctx:
                continue
            f = compute_factors(sym, instruments[sym].security_id,
                                daily_by_symbol[sym], as_of=pd.Timestamp(day))
            if f is None:
                continue
            factor_rows.append(f)
            contexts[sym] = ctx

        if not factor_rows:
            continue
        sel = select_universe(
            factors_frame(factor_rows), cfg,
            regime_family=regime.family, direction=regime.direction,
            as_of=pd.Timestamp(day),
            position_notional=cfg.risk.position_notional_cap(cfg.capital),
        )
        chosen = [s for s in sel.symbols if s in sessions_by_symbol
                  and day in sessions_by_symbol[s]]
        if not chosen:
            continue

        broker = PaperBroker(cfg)
        engine = SessionEngine(cfg, registry, broker, regime,
                               instruments={s: instruments[s] for s in chosen},
                               daily_context={s: contexts[s] for s in chosen})
        day_bars = {s: sessions_by_symbol[s][day] for s in chosen}
        res = engine.run(day, day_bars)

        records.append(res.record_row())
        tf = res.book.trades_frame()
        if len(tf):
            tf.insert(0, "trade_date", day)
            all_trades.append(tf)
        log.info("%s | %s | %d names | %d sig | %d entries | net Rs %s",
                 day, regime.label, len(chosen), res.signals_generated,
                 res.entries, f"{res.net_pnl:,.0f}")

    # ------------------------------------------------------------- report
    if not records:
        log.error("no sessions replayed")
        return 1

    rec = pd.DataFrame(records)
    rec.to_csv(out_dir / "paper_sessions.csv", index=False)
    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    if len(trades):
        trades.to_csv(out_dir / "paper_trades.csv", index=False)
    if args.promote:
        provenance.write(STATE_DIR / "paper_record.csv", rec, cfg,
                         source=str(out_dir.relative_to(ROOT)
                                    if out_dir.is_relative_to(ROOT) else out_dir))
        log.info("promoted %d sessions to state/paper_record.csv", len(rec))
    else:
        log.info("not promoted; the live gate still reads the previous record. "
                 "Re-run with --promote to make this run the evidence.")

    print()
    print("=" * 78)
    print(f"PAPER REPLAY  {rec['trade_date'].min()} .. {rec['trade_date'].max()}")
    print("=" * 78)
    n_tr = int(rec["trades"].sum())
    net = float(rec["net_pnl"].sum())
    gross = float(rec["gross_pnl"].sum())
    costs = float(rec["costs"].sum())
    wins = int(rec["wins"].sum())
    print(f"sessions          {len(rec)}")
    print(f"round trips       {n_tr}")
    print(f"win rate          {wins / n_tr * 100:.1f}%" if n_tr else "win rate          n/a")
    print(f"gross P&L         Rs {gross:>12,.2f}")
    print(f"costs             Rs {costs:>12,.2f}"
          + (f"   ({costs / abs(gross) * 100:.0f}% of gross)" if gross else ""))
    print(f"net P&L           Rs {net:>12,.2f}   "
          f"({net / cfg.capital * 100:+.2f}% on Rs {cfg.capital:,.0f})")
    print(f"best session      Rs {rec['net_pnl'].max():>12,.2f}")
    print(f"worst session     Rs {rec['net_pnl'].min():>12,.2f}")
    print(f"halted sessions   {int(rec['halted'].sum())}")
    print(f"signals / entries {int(rec['signals'].sum())} / {int(rec['entries'].sum())}"
          f"  (risk blocked {int(rec['rejected_by_risk'].sum())})")

    if len(trades):
        print()
        print("by strategy:")
        g = trades.groupby("strategy").agg(
            trades=("net_pnl", "size"), net=("net_pnl", "sum"),
            wins=("net_pnl", lambda x: int((x > 0).sum())))
        g["win_rate"] = (g["wins"] / g["trades"] * 100).round(0)
        print(g.sort_values("net", ascending=False).to_string())
        print()
        print("by exit reason:")
        print(trades.groupby("reason")["net_pnl"].agg(["size", "sum"]).to_string())

    print()
    verdict = evaluate(rec, GateCriteria())
    print(verdict.render())

    (out_dir / "summary.json").write_text(json.dumps({
        "sessions": len(rec), "trades": n_tr, "gross_pnl": gross,
        "costs": costs, "net_pnl": net,
        "gate_passed": verdict.passed, "gate_failures": verdict.failures,
    }, indent=2, default=str))
    log.info("wrote %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
