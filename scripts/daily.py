"""One command for one trading day: refresh, classify, decide, trade, record.

    python scripts/daily.py                  # paper session on live data
    python scripts/daily.py --live           # requests live; the gate may refuse
    python scripts/daily.py --skip-universe  # reuse this morning's universe

Intended to be triggered once each trading morning, before 09:15. Windows Task
Scheduler:

    schtasks /create /tn KABALI /tr "python D:\\KABALI\\scripts\\daily.py" ^
             /sc weekly /d MON,TUE,WED,THU,FRI /st 08:45

The universe refresh is cheap after the first run -- `research.datastore` serves
every bar it already holds from the parquet cache, so a warm day fetches only the
sessions since the last run rather than five years again.

STAGES ARE ORDERED SO A FAILURE IS SAFE. The universe and regime are computed
BEFORE the broker is constructed. If anything upstream fails the process exits
having placed no orders, which is the correct outcome for a bot that cannot see
the market properly.

The bot is a holiday-detector by accident, not by design: on a non-trading day
the feed returns no bars, no session is produced, and it exits having done
nothing. That is fine, but it means an empty result is not evidence of anything.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

log = logging.getLogger("daily")


def run(cmd: list[str], label: str) -> int:
    log.info("%s: %s", label, " ".join(cmd))
    proc = subprocess.run(cmd, cwd=ROOT)
    if proc.returncode != 0:
        log.error("%s failed with exit code %d", label, proc.returncode)
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", type=int, default=100, help="top-N names to scan live")
    ap.add_argument("--live", action="store_true", help="request live orders")
    ap.add_argument("--skip-universe", action="store_true")
    ap.add_argument("--poll", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    today = dt.date.today()
    if today.weekday() >= 5:
        log.info("%s is a weekend -- nothing to do", today)
        return 0

    py = sys.executable

    if not args.skip_universe:
        rc = run([py, "scripts/build_universe.py"], "universe refresh")
        if rc != 0:
            log.error("aborting: the universe is stale and no session should run on it")
            return rc

    cmd = [py, "scripts/run_live.py", "--symbols", str(args.symbols),
           "--poll", str(args.poll)]
    if args.live:
        cmd.append("--live")
    rc = run(cmd, "session")

    # Always report where the gate stands afterwards, whether or not we traded.
    run([py, "cli.py", "gate"], "gate status")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
