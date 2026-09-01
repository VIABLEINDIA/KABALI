"""Walk the swing hypothesis forward over the cached daily bars.

Run:
    python scripts/run_swing.py --out runs/swing_v1
    python scripts/run_swing.py --capital 400000 --out runs/swing_400k
    python scripts/run_swing.py --weekday-sweep --out runs/swing_weekdays

Reads the local parquet cache and nothing else -- no network call is possible
from this path, which is why a four-year study runs in minutes instead of being
throttled into an afternoon.

WHAT IS AND IS NOT DECIDED HERE. The rules, their parameters, and the tests that
would falsify them are fixed in `docs/swing_hypothesis.md` and
`config/bot.yaml`, both written before the first run. This script executes them
and reports what came out, including the two benchmarks the result has to beat
and the weekday spread that says whether the rebalance day was load-bearing.
Nothing in it selects.

THE UNIVERSE IS POINT-IN-TIME. Eligibility -- price, turnover, history, trend,
gap -- is recomputed from trailing windows at every rebalance close. The one
bias that cannot be removed this way is survivorship: the cache holds names
listed today. It inflates the strategy and both benchmarks together, which is
why strategy-minus-benchmark is the number to read.
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

from kabali.config import RUNS_DIR, load_config                              # noqa: E402
from kabali.core import Instrument, get_cost_model                           # noqa: E402
from kabali.data.cache import cached_bars, verify_layout                     # noqa: E402
from kabali.data.ingest import candidate_instruments                         # noqa: E402
from kabali.swing.benchmark import buy_and_hold, equal_weight                # noqa: E402
from kabali.swing.panel import build_panel                                   # noqa: E402
from kabali.swing.portfolio import SwingEngine                               # noqa: E402
from kabali.swing.strategy import ClenowMomentum                             # noqa: E402

log = logging.getLogger("run_swing")

BENCHMARK = Instrument(symbol="NIFTY", security_id="13",
                       exchange_segment="IDX_I", instrument_type="INDEX")

WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri")


def load_index(cfg) -> pd.Series:
    df = cached_bars(BENCHMARK, "D", min_bars=cfg.swing.rules.index_ma + 5)
    if df is None:
        raise SystemExit(
            "NIFTY daily bars are not cached, and the index filter cannot run without "
            "them. Ingest with: python scripts/build_universe.py"
        )
    s = pd.Series(df["close"].to_numpy(dtype=float),
                  index=pd.DatetimeIndex(pd.to_datetime(df["timestamp"])).normalize())
    return s[~s.index.duplicated(keep="last")].sort_index()


def first_tradeable_date(strategy, dates) -> pd.Timestamp | None:
    """The first session on which the rules could have produced any answer.

    Starting the study before this point would report the warmup as flat
    performance and quietly improve every ratio that divides by time.
    """
    for d in dates:
        if bool(strategy.investable.loc[d].any()) and pd.notna(strategy.market_filter.loc[d]):
            return d
    return None


def summarise(result, benchmarks, cfg, extra: dict) -> dict:
    m = result.metrics()
    eq = result.equity
    out = {
        "strategy": result.strategy,
        "capital": result.initial_capital,
        "first_session": str(result.first_bar.date()),
        "last_session": str(result.last_bar.date()),
        "sessions": int(result.bars_tested),
        "years": round((result.last_bar - result.first_bar).days / 365.25, 2),
        "trades": result.trade_count,
        "gross_pnl": round(result.gross_pnl, 2),
        "costs": round(result.costs, 2),
        "net_pnl": round(result.net_pnl, 2),
        "final_equity": round(float(eq.iloc[-1]), 2) if len(eq) else None,
        "return_pct": round(float(eq.iloc[-1] / result.initial_capital - 1.0) * 100, 2) if len(eq) else None,
        "cagr_pct": None if m.cagr_pct is None else round(m.cagr_pct, 2),
        "max_drawdown_pct": None if m.max_drawdown_pct is None else round(m.max_drawdown_pct, 2),
        "sharpe": None if m.sharpe is None else round(m.sharpe, 2),
        "win_rate_pct": None if m.win_rate is None else round(m.win_rate, 1),
        "profit_factor": None if m.profit_factor is None else round(m.profit_factor, 2),
        "avg_hold_days": None if m.avg_bars_held is None else round(m.avg_bars_held, 1),
        "top_trade_share": None if m.top_trade_share is None else round(m.top_trade_share, 3),
        "exposure_days_pct": round(result.exposure_days / max(result.bars_tested, 1) * 100, 1),
        "cost_drag_pct": None if m.cost_drag_pct is None else round(m.cost_drag_pct, 2),
        "rejections": dict(sorted(result.rejections.items(), key=lambda kv: -kv[1])),
        "benchmarks": {},
    }
    for b in benchmarks:
        out["benchmarks"][b.name] = {
            "return_pct": round(b.return_pct, 2),
            "cagr_pct": None if b.cagr_pct() is None else round(b.cagr_pct(), 2),
            "max_drawdown_pct": round(b.max_drawdown_pct(), 2),
            "net_pnl": round(b.net_pnl, 2),
            "note": b.note,
        }
        out[f"vs_{b.name}_pp"] = round(out["return_pct"] - b.return_pct, 2) \
            if out["return_pct"] is not None else None
    out.update(extra)
    return out


def yearly_table(result) -> pd.DataFrame:
    """Per-calendar-year P&L. One good year is not a result about the rule."""
    tf = result.trades_frame()
    eq = result.equity
    rows = []
    for year, grp in (tf.groupby(tf["closed_at"].dt.year) if len(tf) else []):
        year_eq = eq[eq.index.year == year]
        rows.append({
            "year": int(year),
            "trades": len(grp),
            "gross": round(float(grp["gross_pnl"].sum()), 2),
            "costs": round(float(grp["costs"].sum()), 2),
            "net": round(float(grp["net_pnl"].sum()), 2),
            "win_rate_pct": round(float((grp["net_pnl"] > 0).mean() * 100), 1),
            "equity_end": round(float(year_eq.iloc[-1]), 2) if len(year_eq) else None,
        })
    return pd.DataFrame(rows)


def exit_table(result) -> pd.DataFrame:
    """Which exit closed each trade, and what it was worth."""
    tf = result.trades_frame()
    if not len(tf):
        return pd.DataFrame()
    g = tf.groupby("exit_reason")["net_pnl"]
    return pd.DataFrame({
        "trades": g.size(),
        "net": g.sum().round(2),
        "avg": g.mean().round(2),
        "win_rate_pct": (tf.groupby("exit_reason")["net_pnl"].apply(
            lambda s: (s > 0).mean() * 100).round(1)),
    }).sort_values("trades", ascending=False)


def run_once(cfg, panel, index_close, start, end, weekday: int | None = None):
    if weekday is not None:
        from dataclasses import replace
        cfg = replace(cfg, swing=replace(
            cfg.swing, rules=replace(cfg.swing.rules, rebalance_weekday=weekday)))
    strategy = ClenowMomentum(panel, index_close, cfg)
    engine = SwingEngine(panel, strategy, cfg)
    return strategy, engine.run(start=start, end=end)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capital", type=float, default=None,
                    help="override account capital; a capacity diagnostic, not a knob")
    ap.add_argument("--min-bars", type=int, default=250,
                    help="drop symbols with less cached history than this")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--weekday-sweep", action="store_true",
                    help="rerun on all five rebalance weekdays and report the spread")
    ap.add_argument("--out", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    if args.capital is not None:
        from dataclasses import replace
        cfg = replace(cfg, account=replace(cfg.account, capital=float(args.capital)))
    log.info("config: %s", cfg.summary())
    log.info("%s", cfg.swing.summary(cfg.capital))

    if not verify_layout():
        log.error("bar cache path disagrees with research.datastore; aborting")
        return 1

    out_dir = Path(args.out) if args.out else RUNS_DIR / f"swing_{pd.Timestamp.now():%Y%m%d_%H%M}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ data
    t0 = time.monotonic()
    index_close = load_index(cfg)
    instruments = candidate_instruments(series=cfg.universe.candidate_series)
    log.info("%d EQ instruments in the scrip master; reading cached daily bars",
             len(instruments))
    panel = build_panel(instruments, calendar=index_close.index, min_bars=args.min_bars)
    log.info("panel: %s (%.1fs)", panel.describe(), time.monotonic() - t0)
    if not len(panel.symbols):
        log.error("no cached daily bars; run scripts/build_universe.py first")
        return 1

    # --------------------------------------------------------------- the run
    strategy = ClenowMomentum(panel, index_close, cfg)
    start = pd.Timestamp(args.start) if args.start else first_tradeable_date(strategy, panel.dates)
    if start is None:
        log.error("no session in the panel has an investable name; nothing to test")
        return 1
    end = pd.Timestamp(args.end) if args.end else panel.dates[-1]
    log.info("study window %s -> %s (warmup consumed %d sessions)",
             start.date(), end.date(), int((panel.dates < start).sum()))

    _, result = run_once(cfg, panel, index_close, start, end)
    log.info("run complete in %.1fs: %d trades, net Rs %.0f",
             time.monotonic() - t0, result.trade_count, result.net_pnl)

    # ---------------------------------------------------------- benchmarks
    window = panel.slice(start, end)
    investable_at_start = [s for s in panel.symbols
                           if bool(strategy.investable.loc[start, s])]
    benchmarks = [
        buy_and_hold(index_close[(index_close.index >= start) & (index_close.index <= end)],
                     cfg.capital, cfg.swing.costs, name="nifty_buy_hold",
                     note="index, not an ETF: no expense ratio or tracking error modelled"),
        equal_weight(window, investable_at_start, cfg.capital, cfg.swing.costs),
    ]

    extra = {
        "panel_symbols": len(panel.symbols),
        "investable_at_start": len(investable_at_start),
        "rebalance_weekday": WEEKDAYS[cfg.swing.rules.rebalance_weekday],
        "rules": {
            "rank_window": cfg.swing.rules.rank_window,
            "trend_ma": cfg.swing.rules.trend_ma,
            "index_ma": cfg.swing.rules.index_ma,
            "max_gap_pct": cfg.swing.rules.max_gap_pct,
            "exit_rank_pct": cfg.swing.rules.exit_rank_pct,
            "stop_atr_mult": cfg.swing.rules.stop_atr_mult,
            "max_concurrent_positions": cfg.swing.risk.max_concurrent_positions,
            "per_trade_risk_pct": cfg.swing.risk.per_trade_risk_pct,
        },
        "cost_profile": cfg.swing.costs.profile,
        # Fingerprinted for the same reason ULTIMATE fingerprints every result:
        # a P&L computed under a different cost model is not comparable to this
        # one, and the only way to know is to record which model produced it.
        "cost_fingerprint": get_cost_model(cfg.swing.costs.profile).fingerprint(),
    }

    # -------------------------------------------------------- weekday spread
    if args.weekday_sweep:
        spread = []
        for wd in range(5):
            _, r = run_once(cfg, panel, index_close, start, end, weekday=wd)
            spread.append({
                "weekday": WEEKDAYS[wd], "trades": r.trade_count,
                "net_pnl": round(r.net_pnl, 2),
                "return_pct": round(float(r.equity.iloc[-1] / r.initial_capital - 1) * 100, 2),
            })
            log.info("  %s: %d trades, net Rs %.0f", WEEKDAYS[wd], r.trade_count, r.net_pnl)
        extra["weekday_spread"] = spread
        rets = [s["return_pct"] for s in spread]
        extra["weekday_spread_pp"] = round(max(rets) - min(rets), 2)

    # ------------------------------------------------------------- artefacts
    summary = summarise(result, benchmarks, cfg, extra)
    result.trades_frame().to_csv(out_dir / "swing_trades.csv", index=False)

    curves = pd.DataFrame({"strategy": result.equity})
    for b in benchmarks:
        curves[b.name] = b.equity.reindex(result.equity.index).ffill()
    curves.to_csv(out_dir / "swing_equity.csv")

    yearly = yearly_table(result)
    if len(yearly):
        yearly.to_csv(out_dir / "swing_by_year.csv", index=False)
        summary["by_year"] = yearly.to_dict("records")
    exits = exit_table(result)
    if len(exits):
        exits.to_csv(out_dir / "swing_by_exit.csv")
        summary["by_exit"] = exits.reset_index().to_dict("records")

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # ---------------------------------------------------------------- report
    print()
    print(f"  SWING RESULT  {summary['strategy']}  {summary['first_session']} -> "
          f"{summary['last_session']}  ({summary['years']} years)")
    print(f"  capital Rs {summary['capital']:,.0f} | {summary['trades']} round trips | "
          f"exposure {summary['exposure_days_pct']}% of sessions")
    print()
    print(f"  gross      Rs {summary['gross_pnl']:>12,.0f}")
    print(f"  costs      Rs {summary['costs']:>12,.0f}")
    print(f"  net        Rs {summary['net_pnl']:>12,.0f}   ({summary['return_pct']}%, "
          f"CAGR {summary['cagr_pct']}%)")
    print(f"  max DD     {summary['max_drawdown_pct']}%   Sharpe {summary['sharpe']}   "
          f"win rate {summary['win_rate_pct']}%   PF {summary['profit_factor']}")
    print()
    for name, b in summary["benchmarks"].items():
        print(f"  {name:<22} {b['return_pct']:>8.2f}%   CAGR {b['cagr_pct']}%   "
              f"maxDD {b['max_drawdown_pct']}%   -> strategy is "
              f"{summary[f'vs_{name}_pp']:+.2f} pp")
    print()
    if extra.get("weekday_spread"):
        print(f"  weekday spread: {extra['weekday_spread_pp']} pp across the five "
              f"rebalance days")
        for s in extra["weekday_spread"]:
            print(f"    {s['weekday']}  {s['return_pct']:>8.2f}%  ({s['trades']} trades)")
        print()
    if summary["rejections"]:
        print("  entries refused:", ", ".join(f"{k}={v}" for k, v in summary["rejections"].items()))
    print(f"  written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
