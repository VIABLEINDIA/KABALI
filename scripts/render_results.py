"""Regenerate the README's results section from the promoted record.

    python scripts/render_results.py            # print the block
    python scripts/render_results.py --write    # splice it into README.md
    python scripts/render_results.py --check    # exit 1 if the README is stale

WHY THIS EXISTS
===============
The results section was transcribed by hand and went stale three times in one
day -- twice while being actively maintained. Every re-promote moves six figures
in a headline table, four rows of a breakeven table, a regime table and two
sentences of prose, and a number missed anywhere leaves the document disagreeing
with the evidence the gate actually scores.

That is the same failure the provenance sidecar fixed for the CSV: a record and
a claim about it, drifting apart with nothing to notice. Fixing it for the CSV
and leaving the prose hand-maintained solved half a problem.

So the numbers are generated from `state/paper_record.csv`, its provenance
sidecar, and the trades file of the run that produced it. There is exactly one
source, and the README cannot disagree with it.

WHAT IS AND IS NOT GENERATED
============================
The tables and the figures inside the surrounding sentences are generated. The
ARGUMENT is not: "the risk engine works, the strategies do not", the reasoning
about breakeven win rates, and the instruction not to tune are written by a
person and live outside the markers. A generator that rewrote the interpretation
each time the number moved would be a machine for producing whatever conclusion
the latest run implies -- which is the habit this codebase exists to resist.

--check is for CI or a pre-commit hook: it fails when the README no longer
matches the promoted record, rather than waiting for someone to notice.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kabali.config import ROOT, RUNS_DIR, STATE_DIR                   # noqa: E402
from kabali.execution import provenance                               # noqa: E402

BEGIN = "<!-- BEGIN GENERATED RESULTS -- scripts/render_results.py -->"
END = "<!-- END GENERATED RESULTS -->"


def _trades_for(meta: dict) -> pd.DataFrame | None:
    """The trade-level file of the run that produced the promoted record."""
    src = str(meta.get("source", ""))
    if not src:
        return None
    p = (ROOT / src) / "paper_trades.csv"
    if not p.exists():
        p = RUNS_DIR / Path(src).name / "paper_trades.csv"
    return pd.read_csv(p) if p.exists() else None


def _breakeven(g: pd.DataFrame) -> tuple[float, float]:
    """Payoff ratio and the hit rate it implies, for one set of trades."""
    wins = g.loc[g.net_pnl > 0, "net_pnl"]
    losses = g.loc[g.net_pnl < 0, "net_pnl"]
    if not len(wins) or not len(losses):
        return float("nan"), float("nan")
    payoff = wins.mean() / abs(losses.mean())
    return payoff, 100.0 / (1.0 + payoff)


def render() -> str:
    rec_path = STATE_DIR / "paper_record.csv"
    if not rec_path.exists():
        raise SystemExit("no promoted record; run scripts/run_paper.py --promote")
    rec = pd.read_csv(rec_path)
    meta = provenance.read(rec_path) or {}
    trades = _trades_for(meta)

    n = len(rec)
    n_tr = int(rec["trades"].sum())
    gross, costs, net = (float(rec[c].sum()) for c in ("gross_pnl", "costs", "net_pnl"))
    wins = int(rec["wins"].sum())
    worst = float(rec["net_pnl"].min())
    limit = float(rec["loss_limit"].iloc[0]) if "loss_limit" in rec else 800.0
    breaches = int((rec["net_pnl"] < -rec["loss_limit"]).sum()) if "loss_limit" in rec else 0
    halted = int(rec["halted"].sum()) if "halted" in rec else 0
    pf_g = float(rec.loc[rec.net_pnl > 0, "net_pnl"].sum())
    pf_l = abs(float(rec.loc[rec.net_pnl < 0, "net_pnl"].sum()))
    pf = pf_g / pf_l if pf_l else float("inf")

    out = [BEGIN, ""]
    out.append(
        f"Walk-forward paper replay, {n} sessions "
        f"({meta.get('first_date', '?')} → {meta.get('last_date', '?')}), 250 names "
        f"scanned daily, regime and universe recomputed as-of each morning. This is "
        f"the promoted record (`{meta.get('source', '?')}`, code "
        f"`{meta.get('code_fingerprint', '?')}`) — the evidence the gate scores. "
        f"Reproduce with\n`python scripts/run_paper.py --days 60 --symbols 250 --promote`.")
    out += ["", "| | |", "|---|---|",
            f"| Sessions | {n} |",
            f"| Round trips | {n_tr:,} |",
            f"| Win rate | {wins / n_tr * 100:.1f}% |" if n_tr else "| Win rate | n/a |",
            f"| **Gross P&L** | **−₹{abs(gross):,.0f}** |" if gross < 0
            else f"| **Gross P&L** | **₹{gross:,.0f}** |",
            f"| Costs | ₹{costs:,.0f} |",
            f"| **Net P&L** | **{'−' if net < 0 else ''}₹{abs(net):,.0f} "
            f"({net / 40000 * 100:+.1f}% on ₹40,000)** |",
            f"| Profit factor | {pf:.2f} |",
            f"| Worst session | −₹{abs(worst):,.0f} (limit ₹{limit:,.0f} — "
            f"{'held' if abs(worst) <= limit else 'BREACHED'}) |",
            f"| Sessions breaching the daily limit | **{breaches} / {n}** |",
            f"| Halted sessions | {halted} |", ""]

    if trades is not None and len(trades):
        payoff, need = _breakeven(trades)
        got = (trades.net_pnl > 0).mean() * 100
        out.append(f"**It loses before costs.** Gross is "
                   f"{'−' if gross < 0 else ''}₹{abs(gross):,.0f}, so this is not a fee "
                   f"problem with a good signal underneath. Costs then "
                   f"{'more than double' if costs > abs(gross) else 'add to'} the loss — "
                   f"₹{costs:,.0f} over {n_tr:,} round trips is "
                   f"{costs / 40000 * 100:.0f}% of capital in friction across one quarter "
                   f"— but removing them entirely still leaves a losing system.")
        out += ["", "**Breakeven diagnosis — payoff against hit rate:**", "",
                "| Strategy | Trades | Payoff | Needs | Got | Gap |",
                "|---|---|---|---|---|---|"]
        for name, g in sorted(trades.groupby("strategy"),
                              key=lambda kv: -len(kv[1])):
            p, nd = _breakeven(g)
            gt = (g.net_pnl > 0).mean() * 100
            gap = gt - nd
            out.append(f"| {name} | {len(g)} | "
                       + (f"{p:.2f} | {nd:.1f}% | {gt:.1f}% | {gap:+.1f}pp |"
                          if p == p else f"— | — | {gt:.1f}% | sample too small |"))
        out.append(f"| **All strategies** | **{len(trades)}** | **{payoff:.2f}** | "
                   f"**{need:.1f}%** | **{got:.1f}%** | **{got - need:+.1f}pp** |")

        if "reason" in trades:
            out += ["", "**By exit reason:**", "", "| Reason | Trades | Net |",
                    "|---|---|---|"]
            for r, g in sorted(trades.groupby("reason"), key=lambda kv: kv[1].net_pnl.sum()):
                out.append(f"| {r} | {len(g)} | "
                           f"{'−' if g.net_pnl.sum() < 0 else ''}₹{abs(g.net_pnl.sum()):,.0f} |")

    if "regime" in rec:
        out += ["", "**By regime:**", "",
                "| Regime | Sessions | Net | Per session |", "|---|---|---|---|"]
        for r, g in sorted(rec.groupby("regime"), key=lambda kv: kv[1].net_pnl.sum()):
            s = g.net_pnl.sum()
            out.append(f"| {r} | {len(g)} | {'−' if s < 0 else '+'}₹{abs(s):,.0f} | "
                       f"{'−' if s < 0 else '+'}₹{abs(s / len(g)):,.0f} |")

    out += ["", f"<sub>Generated from `state/paper_record.csv` "
                f"(promoted {str(meta.get('written_at', '?'))[:16]}, config "
                f"`{meta.get('config_fingerprint', '?')}`). Do not edit by hand — "
                f"run `scripts/render_results.py --write`.</sub>", "", END]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true", help="splice into README.md")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the README does not match the record")
    args = ap.parse_args()

    block = render()
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")

    if not args.write and not args.check:
        # The block contains rupee signs and arrows; a Windows console defaults to
        # cp1252 and raises on them. Write through a UTF-8 wrapper rather than
        # stripping characters that belong in the document.
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print(block)
        return 0

    if BEGIN not in text or END not in text:
        print(f"markers not found in {readme}. Add these around the results block:\n"
              f"  {BEGIN}\n  {END}")
        return 2

    current = text[text.index(BEGIN):text.index(END) + len(END)]
    if args.check:
        if current.strip() == block.strip():
            print("README results match the promoted record")
            return 0
        print("README results are STALE -- run scripts/render_results.py --write")
        return 1

    readme.write_text(text.replace(current, block), encoding="utf-8")
    print(f"updated {readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
