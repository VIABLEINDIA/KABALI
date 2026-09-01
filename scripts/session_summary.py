"""Summarise a finished forward session, and put it where a person will see it.

    python scripts/session_summary.py                 # print
    python scripts/session_summary.py --notify        # also raise a desktop toast

Run by `forward_session.cmd` once the session ends. Exists because a scheduled
task finishes into an empty room: the result lands in a log nobody has open, and
the interesting part -- whether the risk engine held, whether anything traded at
all -- is four screens up from the end of it.

WHAT IT REPORTS, AND WHY THOSE
==============================
The session's own row, then the two things that are only knowable by comparison:
whether the daily loss limit held, and how the day sits against the promoted
record's per-session average. A single forward session says almost nothing about
the edge -- one day against 66 -- but it says a great deal about whether the
machinery ran correctly on live data, which is the actual point of running it.

An empty session is reported as empty rather than as success. On a holiday the
feed returns no bars, no session is produced, and everything exits zero; calling
that "ran fine" would make a non-trading day indistinguishable from a working one.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kabali.config import RUNS_DIR, STATE_DIR                         # noqa: E402


def build(today: dt.date | None = None) -> tuple[str, bool]:
    """Returns (summary text, whether a session actually traded)."""
    today = today or dt.date.today()
    rec_path = STATE_DIR / "paper_record.csv"
    lines = [f"KABALI forward session  {today}", "=" * 46]

    if not rec_path.exists():
        return "\n".join(lines + ["no paper record found"]), False

    rec = pd.read_csv(rec_path)
    rec["trade_date"] = rec["trade_date"].astype(str)
    row = rec[rec["trade_date"] == str(today)]

    if row.empty:
        lines += [
            "NO SESSION RECORDED for today.",
            "",
            "Expected on a market holiday: the feed returns no bars, no session",
            "is produced, and the process exits zero. Not the same as a session",
            "that ran and found nothing -- check the log to tell them apart.",
            f"  log: runs/forward_{today:%Y%m%d}.log",
        ]
        return "\n".join(lines), False

    r = row.iloc[-1]
    prior = rec[rec["trade_date"] != str(today)]
    avg = float(prior["net_pnl"].mean()) if len(prior) else float("nan")
    limit = float(r.get("loss_limit", 800.0))
    net = float(r["net_pnl"])
    breached = net < -limit

    lines += [
        f"  regime          {r.get('regime', '?')}",
        f"  universe        {int(r.get('universe_size', 0))} names",
        f"  signals/entries {int(r.get('signals', 0))} / {int(r.get('entries', 0))}"
        f"   (risk blocked {int(r.get('rejected_by_risk', 0))})",
        f"  round trips     {int(r.get('trades', 0))}",
        f"  net P&L         Rs {net:,.2f}",
        f"  daily limit     Rs {limit:,.0f}  -> "
        + ("*** BREACHED -- RISK ENGINE BUG ***" if breached else "held"),
    ]
    if r.get("halted"):
        lines.append(f"  HALTED          {r.get('halt_reason', '')}")
    if avg == avg:
        lines += ["",
                  f"  against the promoted record: {len(prior)} prior sessions "
                  f"averaged Rs {avg:,.0f}/session.",
                  "  One session is not evidence about the edge. It is evidence the",
                  "  machinery ran on live data, which is what this run is for."]
    lines += ["", f"  log: runs/forward_{today:%Y%m%d}.log",
              "  gate: python cli.py gate"]
    return "\n".join(lines), True


def notify(title: str, body: str) -> None:
    """Best-effort desktop toast. Never fails the run."""
    try:
        import subprocess
        ps = (
            "[reflection.assembly]::LoadWithPartialName('System.Windows.Forms')>$null;"
            "$n=New-Object System.Windows.Forms.NotifyIcon;"
            "$n.Icon=[System.Drawing.SystemIcons]::Information;"
            "$n.BalloonTipTitle=$env:KT;$n.BalloonTipText=$env:KB;"
            "$n.Visible=$true;$n.ShowBalloonTip(20000);Start-Sleep -Seconds 12;"
            "$n.Dispose()"
        )
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], timeout=40,
                       env={**__import__("os").environ, "KT": title, "KB": body[:250]},
                       capture_output=True)
    except Exception:                                                 # noqa: BLE001
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--notify", action="store_true", help="raise a desktop toast too")
    ap.add_argument("--date", default=None)
    args = ap.parse_args()

    day = pd.Timestamp(args.date).date() if args.date else dt.date.today()
    text, traded = build(day)

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(text)

    out = RUNS_DIR / "LATEST_SESSION.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"\nwrote {out}")

    if args.notify:
        head = text.splitlines()
        one = next((x.strip() for x in head if "net P&L" in x), "session finished")
        notify("KABALI forward session", f"{one}. See runs/LATEST_SESSION.txt")
    return 0 if traded else 0


if __name__ == "__main__":
    raise SystemExit(main())
