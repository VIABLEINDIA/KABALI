"""Diagnose a paper run: where the money went, and what would have to change.

    python scripts/analyze.py runs/paper_20260830_1712

Reports the arithmetic that decides whether a rule can work at all, per strategy:

  PAYOFF RATIO        average win / average loss, as actually realised
  BREAKEVEN WIN RATE  1 / (1 + payoff) -- the hit rate the rule NEEDS
  ACTUAL WIN RATE     the hit rate it GOT

The gap between the last two is the whole verdict, and it separates two very
different diagnoses that a P&L total alone conflates. A rule with a 1.7 designed
reward:risk that realises 1.05 is being bled by exits that are neither stop nor
target -- square-offs closing positions mid-idea. A rule whose payoff holds but
whose win rate is far below breakeven is simply wrong about direction, and no
amount of exit tuning will save it.

DESIGNED VERSUS REALISED reward:risk is reported separately for the same reason.
A strategy that targets 2R and realises 1R has an exit problem; one that targets
2R and realises 2R but wins 25% of the time has a signal problem.

This is a diagnostic, not a search. Every number here is computed from a run that
has already happened -- reading it and then tuning parameters until the result
improves is how 6,036 configurations produced zero robust regions in ULTIMATE.
If a rule fails here, the honest options are to retire it or to test a different
hypothesis on data it has not seen.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def breakeven_win_rate(payoff: float) -> float:
    return 1.0 / (1.0 + payoff) if payoff > 0 else float("nan")


def describe(trades: pd.DataFrame, label: str) -> list[str]:
    if trades.empty:
        return [f"{label}: no trades"]
    wins = trades[trades["net_pnl"] > 0]
    losses = trades[trades["net_pnl"] <= 0]
    if wins.empty or losses.empty:
        return [f"{label}: {len(trades)} trades, all one-sided -- not yet informative"]

    avg_w, avg_l = wins["net_pnl"].mean(), abs(losses["net_pnl"].mean())
    payoff = avg_w / avg_l if avg_l > 0 else float("inf")
    need = breakeven_win_rate(payoff)
    got = len(wins) / len(trades)
    edge = got - need

    out = [
        f"{label}",
        f"    trades {len(trades):>4}   net Rs {trades['net_pnl'].sum():>10,.0f}   "
        f"costs Rs {trades['costs'].sum():>9,.0f}",
        f"    avg win Rs {avg_w:>8,.0f}   avg loss Rs {-avg_l:>9,.0f}   "
        f"payoff {payoff:>5.2f}",
        f"    needs {need * 100:>5.1f}% wins   got {got * 100:>5.1f}%   "
        f"{'EDGE +' if edge > 0 else 'SHORTFALL '}{abs(edge) * 100:.1f}pp",
    ]
    mix = trades.groupby("reason")["net_pnl"].agg(["size", "mean", "sum"])
    for reason, row in mix.iterrows():
        out.append(f"      {reason:<10} {int(row['size']):>4} trades  "
                   f"avg Rs {row['mean']:>8,.0f}  total Rs {row['sum']:>10,.0f}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="a directory written by scripts/run_paper.py")
    args = ap.parse_args()

    run = Path(args.run_dir)
    tpath, spath = run / "paper_trades.csv", run / "paper_sessions.csv"
    if not tpath.exists():
        print(f"no paper_trades.csv in {run}")
        return 1
    trades = pd.read_csv(tpath)
    sessions = pd.read_csv(spath) if spath.exists() else pd.DataFrame()

    print("=" * 74)
    print(f"DIAGNOSIS  {run}")
    print("=" * 74)
    print()
    for line in describe(trades, "ALL STRATEGIES"):
        print(line)

    print()
    print("-" * 74)
    for name, grp in trades.groupby("strategy"):
        print()
        for line in describe(grp, name):
            print(line)

    # Cost drag: the fixed floor is what makes small accounts hard.
    print()
    print("-" * 74)
    notional = trades["quantity"] * trades["entry_price"]
    gross_abs = trades["gross_pnl"].abs().sum()
    print(f"\ncost per round trip   Rs {trades['costs'].mean():.2f} on "
          f"Rs {notional.mean():,.0f} average notional "
          f"({trades['costs'].mean() / notional.mean() * 10_000:.1f} bps)")
    if gross_abs > 0:
        print(f"costs / gross moves   {trades['costs'].sum() / gross_abs * 100:.1f}%")

    # Regime attribution -- where the P&L actually lived.
    if not sessions.empty and "regime" in sessions:
        print()
        print("-" * 74)
        print("\nby regime:")
        g = sessions.groupby("regime").agg(
            sessions=("net_pnl", "size"), net=("net_pnl", "sum"),
            avg=("net_pnl", "mean"), trades=("trades", "sum"))
        print(g.round(1).to_string())
        halts = sessions[sessions.get("halted", False) == True]  # noqa: E712
        if len(halts) and "halt_kind" in halts:
            print()
            print("halts:")
            print(halts["halt_kind"].value_counts().to_string())

    print()
    print("Read the shortfall, not the total. A rule below its breakeven win rate")
    print("does not have a tuning problem, it has a hypothesis problem.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
