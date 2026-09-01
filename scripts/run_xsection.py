"""Cross-sectional momentum: one pre-committed run, against a matched benchmark.

    python scripts/run_xsection.py                # cached panel, full report
    python scripts/run_xsection.py --rebuild      # re-read parquet, re-screen
    python scripts/run_xsection.py --controls     # planted-edge and noise runs

THE COMPARISON THAT MATTERS is not the strategy's total return. It is the
strategy against an equal-weight buy-and-hold of the *identical* 606 names over
the *identical* window. That benchmark carries exactly the same survivorship
bias, so subtracting it cancels most of the bias rather than arguing about its
size. A momentum portfolio that returns 200% while the names it picks from
returned 190% has demonstrated almost nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kabali.core import ensure_core_importable          # noqa: E402
from kabali.xsection import controls as ctl             # noqa: E402
from kabali.xsection import evaluate as ev              # noqa: E402
from kabali.xsection import momentum as mom             # noqa: E402
from kabali.xsection import portfolio as pf             # noqa: E402
from kabali.xsection.panel import load_panel            # noqa: E402

ensure_core_importable()
from research.metrics import max_drawdown_pct, sharpe_ratio  # noqa: E402

CACHE = Path("state")
STATE_FILES = {"close": "panel_close.parquet", "open": "panel_open.parquet",
               "volume": "panel_volume.parquet"}


# ---------------------------------------------------------------- benchmarking
def equal_weight(close: pd.DataFrame, start) -> pd.Series:
    """Daily-rebalanced equal-weight hold of every name in the panel.

    `fill_method=None` is not a detail. The pandas default pads missing values
    before differencing, which computes a return across a gap the portfolio could
    not have traded through -- the same forward-fill-into-a-decision mistake the
    panel builder refuses to make.
    """
    c = close.loc[start:]
    rets = c.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    return (1.0 + rets.mean(axis=1, skipna=True)).cumprod()


def stats(equity: pd.Series, trades: pd.DataFrame, capital: float) -> dict:
    e = equity.dropna()
    years = max((e.index[-1] - e.index[0]).days / 365.25, 1e-9)
    total = e.iloc[-1] / e.iloc[0] - 1.0
    return {
        "total_return_pct": 100.0 * total,
        "cagr_pct": 100.0 * ((1.0 + total) ** (1.0 / years) - 1.0),
        "sharpe": sharpe_ratio(e),
        "max_drawdown_pct": max_drawdown_pct(e),
        "final_equity": float(e.iloc[-1]),
        "trades": int(len(trades)),
        "costs": float(trades["cost"].sum()) if len(trades) else 0.0,
        "costs_pct_of_capital": (100.0 * float(trades["cost"].sum()) / capital
                                 if len(trades) else 0.0),
        "years": years,
    }


def yearly(equity: pd.Series, bench: pd.Series) -> pd.DataFrame:
    """Calendar-year returns for both, so one good year cannot carry the verdict."""
    rows = []
    for year, grp in equity.groupby(equity.index.year):
        b = bench.reindex(grp.index).dropna()
        if len(grp) < 2 or len(b) < 2:
            continue
        rows.append({
            "year": int(year),
            "strategy_pct": 100.0 * (grp.iloc[-1] / grp.iloc[0] - 1.0),
            "benchmark_pct": 100.0 * (b.iloc[-1] / b.iloc[0] - 1.0),
        })
    df = pd.DataFrame(rows)
    if len(df):
        df["excess_pp"] = df.strategy_pct - df.benchmark_pct
    return df


# --------------------------------------------------------------------- running
def run_one(close, open_, turnover, cfg, label):
    scores = mom.momentum(close)
    mask = mom.eligible(close, turnover)
    res = pf.run(close, open_, scores, mask, cfg)
    start = res.equity.index[0]
    bench = equal_weight(close, start) * cfg.capital
    s = stats(res.equity, res.trades, cfg.capital)
    b = stats(bench, pd.DataFrame(columns=["cost"]), cfg.capital)
    ab = ev.alpha_beta(res.equity, bench)
    return {"label": label, "result": res, "bench": bench,
            "strategy": s, "benchmark": b, "alpha_beta": ab,
            "excess_pp": s["total_return_pct"] - b["total_return_pct"]}


def render(run: dict) -> str:
    s, b = run["strategy"], run["benchmark"]
    ab, res = run["alpha_beta"], run["result"]
    out = [
        "=" * 78,
        f"{run['label']}",
        "=" * 78,
        f"window            {res.equity.index[0].date()} .. {res.equity.index[-1].date()}"
        f"  ({s['years']:.1f}y)",
        f"rebalances        {len(res.holdings)}",
        f"trades            {s['trades']}",
        "",
        f"{'':<18}{'STRATEGY':>14}{'EQUAL-WEIGHT':>16}{'DIFFERENCE':>14}",
        f"{'total return':<18}{s['total_return_pct']:>13.1f}%{b['total_return_pct']:>15.1f}%"
        f"{run['excess_pp']:>13.1f}pp",
        f"{'CAGR':<18}{s['cagr_pct']:>13.1f}%{b['cagr_pct']:>15.1f}%"
        f"{s['cagr_pct'] - b['cagr_pct']:>13.1f}pp",
        f"{'Sharpe':<18}{_f(s['sharpe']):>14}{_f(b['sharpe']):>16}"
        f"{_f((s['sharpe'] or 0) - (b['sharpe'] or 0)):>14}",
        f"{'max drawdown':<18}{_f(s['max_drawdown_pct'])+'%':>14}"
        f"{_f(b['max_drawdown_pct'])+'%':>16}",
        "",
        f"{'beta vs benchmark':<18}{ab.beta:>13.2f} {'':>15}(t vs 1.0 = {ab.t_beta_vs_one:.2f})",
        f"{'alpha':<18}{ab.alpha_annual_pct:>12.1f}%/yr{'':>11}(t = {ab.t_alpha:.2f})",
        f"  -> {ab.verdict()}",
        "",
        f"costs             Rs {s['costs']:,.0f}  ({s['costs_pct_of_capital']:.1f}% of capital"
        f" over {s['years']:.1f}y)",
        f"final equity      Rs {s['final_equity']:,.0f}   (benchmark Rs {b['final_equity']:,.0f})",
    ]
    return "\n".join(out)


def _f(v, nd=2):
    return "n/a" if v is None else f"{v:.{nd}f}"


def load_or_build(rebuild: bool):
    paths = {k: CACHE / v for k, v in STATE_FILES.items()}
    if not rebuild and all(p.exists() for p in paths.values()):
        close = pd.read_parquet(paths["close"])
        open_ = pd.read_parquet(paths["open"])
        volume = pd.read_parquet(paths["volume"])
        return close, open_, close * volume, None
    panel = load_panel()
    panel.close.to_parquet(paths["close"])
    panel.open.to_parquet(paths["open"])
    panel.volume.to_parquet(paths["volume"])
    return panel.close, panel.open, panel.turnover, panel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="re-read the parquet cache")
    ap.add_argument("--controls", action="store_true", help="also run planted-edge and noise panels")
    ap.add_argument("--capital", type=float, default=40_000.0)
    ap.add_argument("--positions", type=int, default=10)
    ap.add_argument("--rebalance-days", type=int, default=21)
    ap.add_argument("--random", type=int, default=0, metavar="N",
                    help="N random-ranking portfolios as an empirical null")
    ap.add_argument("--grid", action="store_true",
                    help="report EVERY position-count/cadence cell, not the best one")
    ap.add_argument("--out", default="runs/xsection")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = pf.XSecConfig(capital=args.capital, n_positions=args.positions,
                        rebalance_days=args.rebalance_days)

    close, open_, turnover, panel = load_or_build(args.rebuild)
    print(f"panel: {close.shape[1]} symbols x {len(close)} sessions  "
          f"{close.index[0].date()} .. {close.index[-1].date()}\n")

    live = run_one(close, open_, turnover, cfg,
                   "CROSS-SECTIONAL MOMENTUM 12-1, monthly, long-only, NSE cash delivery")
    print(render(live))

    yr = yearly(live["result"].equity, live["bench"])
    if len(yr):
        print("\nby calendar year:")
        print(yr.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

    live["result"].trades.to_csv(outdir / "trades.csv", index=False)
    live["result"].equity.to_csv(outdir / "equity.csv")
    summary = {k: live[k] for k in ("strategy", "benchmark", "excess_pp")}
    summary["alpha_beta"] = vars(live["alpha_beta"])
    summary["config"] = vars(cfg)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    if args.random:
        rnd = ev.random_portfolios(close, open_, mom.eligible(close, turnover), cfg,
                                   pf.run, trials=args.random)
        pct = ev.percentile_of(live["strategy"]["total_return_pct"], rnd.total_return_pct)
        print(f"\nRANDOM-RANKING NULL ({args.random} trials, same engine, "
              f"same eligibility, same costs):")
        for q in (5, 25, 50, 75, 95):
            print(f"  p{q:<3} total {np.percentile(rnd.total_return_pct, q):8.1f}%")
        print(f"  strategy {live['strategy']['total_return_pct']:.1f}% sits at the "
              f"{pct:.0f}th percentile of random selection")
        rnd.to_csv(outdir / "random_null.csv", index=False)

    if args.grid:
        print("\nROBUSTNESS -- every cell reported, none selected:")
        print(f"{'N':>4}{'reb':>5}{'total%':>9}{'CAGR%':>8}{'SR':>6}"
              f"{'beta':>7}{'alpha%/yr':>11}{'t':>7}{'trades':>8}")
        for n in (5, 10, 20, 30, 50):
            for reb in (21, 63):
                g = pf.XSecConfig(capital=args.capital, n_positions=n, rebalance_days=reb)
                r = run_one(close, open_, turnover, g, "")
                gs, gab = r["strategy"], r["alpha_beta"]
                print(f"{n:>4}{reb:>5}{gs['total_return_pct']:>9.1f}{gs['cagr_pct']:>8.1f}"
                      f"{_f(gs['sharpe']):>6}{gab.beta:>7.2f}"
                      f"{gab.alpha_annual_pct:>11.1f}{gab.t_alpha:>7.2f}{gs['trades']:>8}")

    if args.controls:
        print("\n" + "=" * 78 + "\nCONTROLS\n" + "=" * 78)
        for label, strength in (("NEGATIVE CONTROL (pure noise, no edge to find)", 0.0),
                                ("POSITIVE CONTROL (planted 12-1 momentum)", 0.10)):
            c, o = ctl.synthetic_panel(momentum_strength=strength)
            t = ctl.synthetic_turnover(c)
            run = run_one(c, o, t, cfg, label)
            print("\n" + render(run))

    print("\nCAVEATS")
    for line in (panel.caveats() if panel else _cached_caveats()):
        print(f"  - {line}")
    print(f"\nwrote {outdir}/")
    return 0


def _cached_caveats():
    from kabali.xsection.panel import Panel
    return Panel(*(pd.DataFrame(),) * 4).caveats()


if __name__ == "__main__":
    raise SystemExit(main())
