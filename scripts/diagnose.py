"""Operational self-diagnosis: what is wrong with the machine, not with the edge.

    python scripts/diagnose.py
    python scripts/diagnose.py --json runs/diagnosis.json
    python scripts/diagnose.py --universe-only

This reads the bar cache's audit sidecars, the run logs, the universe and the
promoted paper record, and reports what is degrading the bot's ability to see the
market: illiquid names being traded, stale or missing history, sessions lost to
network failures, symbols the vendor keeps refusing.

WHAT THIS DELIBERATELY DOES NOT DO
==================================
It does not look at P&L, and it does not suggest a parameter. A tool that read
its own trading results and proposed adjustments would be automating the exact
search that produced zero robust regions from 6,036 configurations -- with a
larger space and less discipline. Every finding here is falsifiable against
something outside the strategy: a bar that is missing, a fetch that failed, a
name whose volume cannot support the fills the backtest assumed.

That line matters more than it looks. "The bot should learn from its logs" is
two different requests wearing one sentence. Learning that RELIANCE's cache is
three weeks stale is free. Learning that VWAPReversion "should" use a 2.4 ATR
entry because that fits the last 66 sessions is how a losing system acquires a
convincing backtest.

SEVERITY IS ABOUT TRADING, NOT TIDINESS
=======================================
A warning on a name the bot will never trade is noise. The same warning on a name
in today's universe is a live problem, because the fills the sizer assumes are
being computed from that data. Findings are therefore ranked by whether they
touch the traded universe first, and by count second.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kabali.config import RUNS_DIR, STATE_DIR, load_config            # noqa: E402
from kabali.core import ensure_core_importable                       # noqa: E402
from kabali.execution import provenance                              # noqa: E402
from kabali.fundamentals.store import FundamentalsStore              # noqa: E402

log = logging.getLogger("diagnose")

CRITICAL, WARN, INFO = "CRITICAL", "WARNING", "INFO"
_ORDER = {CRITICAL: 0, WARN: 1, INFO: 2}

#: The venue this bot actually trades. Anything else in the shared ULTIMATE cache
#: -- futures, options, indices, BSE lines -- belongs to another venue and its
#: staleness says nothing about this one.
TRADED_SEGMENTS = {"NSE_EQ"}
TRADED_INSTRUMENTS = {"EQUITY"}

#: A cache whose data ends this far short of the range it records as fetched is
#: not merely stale -- it is lying about what it holds.
COVERAGE_GAP_DAYS = 10

#: Cache older than this is suspect for a bot that trades today.
STALE_CACHE_DAYS = 5
#: A universe is re-scored every morning; one this old was not.
STALE_UNIVERSE_DAYS = 3


@dataclass
class Finding:
    severity: str
    area: str
    headline: str
    detail: str = ""
    remedy: str = ""
    subjects: list[str] = field(default_factory=list)   # affected symbols, capped
    count: int = 0

    def render(self) -> str:
        tag = {CRITICAL: "[!!]", WARN: "[! ]", INFO: "[  ]"}[self.severity]
        out = [f"{tag} {self.area}: {self.headline}"]
        if self.detail:
            out += [f"       {line}" for line in self.detail.splitlines()]
        if self.subjects:
            shown = ", ".join(self.subjects[:12])
            more = f" (+{self.count - len(self.subjects[:12])} more)" if self.count > 12 else ""
            out.append(f"       affected: {shown}{more}")
        if self.remedy:
            out.append(f"       -> {self.remedy}")
        return "\n".join(out)


# ------------------------------------------------------------------ helpers

def _universe() -> tuple[set[str], pd.Timestamp | None]:
    p = STATE_DIR / "universe_latest.json"
    if not p.exists():
        return set(), None
    blob = json.loads(p.read_text())
    return set(blob["universe"]["symbols"]), pd.Timestamp(blob["as_of"])


def _classify_warning(text: str) -> str:
    """Bucket an audit warning into a kind, so one cause is reported once."""
    t = text.lower()
    if "zero volume" in t:
        return "zero_volume"
    if "gaps longer than" in t:
        return "history_gaps"
    if "incoherent" in t:
        return "repaired_bars"
    if "duplicate" in t:
        return "duplicate_bars"
    return "other"


def _range_had_sessions(line: str) -> bool:
    """Could the failed range have contained a trading session at all?

    A vendor refusal on a range with no weekdays is the vendor being right, not
    a symptom. Calendar chunking used to leave a weekend-only tail, and counting
    those refusals reported 35 healthy symbols as repeatedly failing -- with a
    remedy of dropping them from the universe, which would have deleted 14% of it.

    Weekends only; exchange holidays would need a calendar this does not have.
    Erring toward reporting a failure is right here: a false positive costs a
    look, a false negative hides a symbol that genuinely stopped resolving.
    """
    m = re.search(r"(\d{4}-\d{2}-\d{2})\.\.(\d{4}-\d{2}-\d{2})", line)
    if not m:
        return True
    try:
        return bool(len(pd.bdate_range(m.group(1), m.group(2))))
    except (ValueError, TypeError):
        return True


def _normalise_log_line(line: str) -> str:
    """Collapse a log line to its cause, dropping ids, addresses and dates."""
    s = re.sub(r"0x[0-9A-Fa-f]+", "<addr>", line)
    s = re.sub(r"\d{4}-\d{2}-\d{2}", "<date>", s)
    s = re.sub(r"\d+", "N", s)
    return s.strip()[:160]


# ------------------------------------------------------------------ checks

def check_cache_audits(universe: set[str], bars_dir: Path) -> list[Finding]:
    """Sweep every cached series' audit sidecar.

    The audits already exist -- `research.datastore` writes one on every fetch --
    but nothing has ever read them back in aggregate. A warning is written once,
    at 3am during an ingest, and then never seen again.
    """
    metas = sorted(bars_dir.glob("*.meta.json"))
    if not metas:
        return [Finding(WARN, "cache", "no bar cache found",
                        detail=f"looked in {bars_dir}",
                        remedy="run scripts/build_universe.py")]

    by_kind: dict[str, list[str]] = defaultdict(list)
    hard: list[str] = []
    stale: list[tuple[str, int]] = []
    now = pd.Timestamp.now()
    scanned = 0

    for m in metas:
        try:
            blob = json.loads(m.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        sym, tf = blob.get("symbol", m.stem), blob.get("timeframe", "?")
        # Segment AND instrument type, not just the ticker. "TCS" names both the
        # NSE equity and a BSE futures contract, and an expired future's cache is
        # legitimately 977 days stale -- reporting that as a stale equity sent the
        # first version of this chasing a problem that did not exist.
        if (sym not in universe
                or blob.get("segment") not in TRADED_SEGMENTS
                or blob.get("instrument_type") not in TRADED_INSTRUMENTS):
            continue
        scanned += 1
        audit = blob.get("audit") or {}

        for w in audit.get("warnings", []):
            by_kind[_classify_warning(w)].append(f"{sym}/{tf}")
        if audit.get("hard_failures"):
            hard.append(f"{sym}/{tf}")

        last = audit.get("last")
        if tf == "D" and last:
            age = (now.normalize() - pd.Timestamp(last).normalize()).days
            if age > STALE_CACHE_DAYS:
                stale.append((f"{sym}/{tf}", age))

    out: list[Finding] = []
    if not scanned:
        return [Finding(WARN, "cache", "no cached series matches the current universe",
                        remedy="run scripts/build_universe.py, then run_paper.py")]

    if hard:
        out.append(Finding(
            CRITICAL, "data", f"{len(hard)} traded series failed their integrity audit",
            detail="These did not merely warn -- the audit refused them. Any result "
                   "computed on them is unreliable.",
            remedy="re-fetch with refresh=True, or drop the name from the universe",
            subjects=sorted(hard), count=len(hard)))

    liq = by_kind.get("zero_volume", [])
    if liq:
        out.append(Finding(
            CRITICAL if len(set(liq)) > scanned * 0.05 else WARN, "data",
            f"{len(set(liq))} traded series carry substantial zero-volume history",
            detail="The backtest fills these at the modelled price. A bar with no "
                   "volume had no counterparty, so the fill it implies did not exist. "
                   "This inflates paper results on exactly the illiquid names the "
                   "universe screens were meant to exclude.",
            remedy="raise universe.min_median_turnover, or tighten "
                   "universe.max_participation_pct",
            subjects=sorted(set(liq)), count=len(set(liq))))

    gaps = by_kind.get("history_gaps", [])
    if gaps:
        out.append(Finding(
            WARN, "data", f"{len(set(gaps))} traded series have long history gaps",
            detail="Gaps break indicator warm-up silently: an ATR computed across a "
                   "three-week hole is not the volatility the sizer thinks it is.",
            remedy="inspect with research.datastore.audit_bars; re-fetch the affected "
                   "ranges",
            subjects=sorted(set(gaps)), count=len(set(gaps))))

    rep = by_kind.get("repaired_bars", [])
    if rep:
        out.append(Finding(
            INFO, "data", f"{len(set(rep))} traded series needed bar repair",
            detail="Incoherent bars were widened to contain open/close -- usually "
                   "settlement prints. Under 1% is tolerated by design.",
            subjects=sorted(set(rep)), count=len(set(rep))))

    if stale:
        stale.sort(key=lambda kv: -kv[1])
        worst = stale[0][1]
        out.append(Finding(
            CRITICAL if worst > 14 else WARN, "cache",
            f"{len(stale)} traded series have daily bars older than {STALE_CACHE_DAYS}d",
            detail=f"Worst is {stale[0][0]} at {worst}d. The regime and universe are "
                   f"recomputed each morning from these; stale bars mean yesterday's "
                   f"market classified as today's.",
            remedy="run scripts/build_universe.py to refresh, or daily.py which does it",
            subjects=[f"{s}({a}d)" for s, a in stale], count=len(stale)))

    out.append(Finding(INFO, "cache",
                       f"{scanned} cached series audited across the traded universe"))
    return out


def check_coverage_gaps(bars_dir: Path, universe: set[str]) -> list[Finding]:
    """Caches that claim a range they do not hold.

    `fetch_bars` stitches chunks and returns whatever came back: if some chunks
    fail it logs a warning and hands over the partial series. `load_bars` then
    records `fetch_range` as the range that was REQUESTED. Coverage is judged
    against that record -- deliberately, so a post-listing instrument is not
    re-downloaded forever -- which means a partial fetch cements its own gap.
    The missing years are never asked for again.

    The two halves are individually reasonable and lethal together, and nothing
    reported it because each looked like it worked: the fetch logged a warning
    nobody re-read, and the cache thereafter served instantly.
    """
    rows = []
    for m in bars_dir.glob("*.meta.json"):
        try:
            blob = json.loads(m.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if (blob.get("segment") not in TRADED_SEGMENTS
                or blob.get("instrument_type") not in TRADED_INSTRUMENTS):
            continue
        audit, fr = blob.get("audit") or {}, blob.get("fetch_range") or {}
        if not audit.get("last") or not fr.get("end"):
            continue
        claimed, actual = pd.Timestamp(fr["end"]), pd.Timestamp(audit["last"])
        gap = (claimed.normalize() - actual.normalize()).days
        if gap > COVERAGE_GAP_DAYS:
            rows.append((blob.get("symbol", m.stem), gap, str(actual.date())))

    if not rows:
        return [Finding(INFO, "coverage", "every cached series holds the range it claims")]

    rows.sort(key=lambda r: -r[1])
    traded = [r for r in rows if r[0] in universe]
    total = sum(r[1] for r in rows)
    return [Finding(
        CRITICAL, "coverage",
        f"{len(rows)} cached series claim coverage they do not hold"
        + (f", {len(traded)} of them in today's universe" if traded else ""),
        detail=f"Worst is {rows[0][0]}, whose data ends {rows[0][2]} while the cache "
               f"records it as fetched through the present -- {rows[0][1]} days short. "
               f"{total:,} bar-days are missing in total. "
               f"These will never re-fetch: coverage is judged against the recorded "
               f"request, so a fetch that partially failed cements its own gap. Any "
               f"factor computed on these names is computed on history that stops.",
        remedy="re-fetch with refresh=True, and fix fetch_range so a partial fetch "
               "does not record the full requested span",
        subjects=[f"{s}({g}d)" for s, g, _ in (traded or rows)],
        count=len(traded or rows))]


def check_fundamentals(universe: set[str]) -> list[Finding]:
    """Coverage and freshness of the fundamentals cache.

    This is the one input the bot cannot refresh on its own -- it comes from an
    agent-session connector, not from anything a 09:00 process can call. So its
    age is an operational fact, not an implementation detail, and it belongs in
    the same report as a stale bar cache.
    """
    store = FundamentalsStore()
    if not len(store):
        return [Finding(INFO, "fundamentals", "no fundamentals cached",
                        detail="Valuation data is optional; nothing depends on it yet.",
                        remedy="populate state/fundamentals.json from an agent session")]
    c = store.coverage(sorted(universe))
    sev = INFO if c["covered"] >= len(universe) * 0.9 else WARN
    return [Finding(
        sev, "fundamentals",
        f"{c['covered']}/{c['requested']} universe names have fundamentals",
        detail=f"{c['fresh']} fresh, {c['stale']} stale, {c['missing']} missing; "
               f"median age {c['median_age_days']}d. This cache cannot self-refresh: "
               f"the connector that fills it is not reachable from a scheduled run.",
        remedy="" if sev == INFO else "top up from an agent session before relying on it")]


def check_run_logs(runs_dir: Path, limit_files: int = 40) -> list[Finding]:
    """Group failures across run logs by cause, not by occurrence."""
    logs = sorted(runs_dir.glob("*.log"), key=lambda p: -p.stat().st_mtime)[:limit_files]
    if not logs:
        return [Finding(INFO, "logs", "no run logs found")]

    causes: Counter = Counter()
    per_symbol: Counter = Counter()
    for f in logs:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            if " ERROR " not in line and " WARNING " not in line:
                continue
            causes[_normalise_log_line(line)] += 1
            m = re.search(r"(?:WARNING|ERROR)\s+([A-Z][A-Z0-9&\-]{1,15})\s+(?:D|\d+m)\b", line)
            if m and _range_had_sessions(line):
                per_symbol[m.group(1)] += 1

    out: list[Finding] = []
    net = sum(n for c, n in causes.items()
              if "getaddrinfo" in c or "Max retries exceeded" in c
              or "Connection aborted" in c)
    if net:
        out.append(Finding(
            CRITICAL if net > 200 else WARN, "network",
            f"{net:,} request failures from lost connectivity",
            detail="DNS or socket failures, not vendor rejections. A run that loses "
                   "symbols this way reports the same shortfall as one where the data "
                   "genuinely does not exist -- and the two demand opposite responses.",
            remedy="re-run the affected ingest; if it recurs, lower --rate or --workers"))

    vendor = Counter()
    for c, n in causes.items():
        m = re.search(r"DH-N", c)
        if m and "getaddrinfo" not in c:
            vendor[c] += n
    if vendor:
        total = sum(vendor.values())
        out.append(Finding(
            WARN, "vendor", f"{total} chunk failures rejected by Dhan",
            detail="Vendor-side refusals (DH-9xx). Distinct from network loss: retrying "
                   "immediately will not help if the instrument genuinely has no data "
                   "for the range.",
            remedy="check whether the affected names list after their listing date"))

    if per_symbol:
        repeat = [s for s, n in per_symbol.most_common() if n >= 3]
        if repeat:
            out.append(Finding(
                WARN, "symbols", f"{len(repeat)} symbols failed on 3+ separate occasions",
                detail="A name that fails repeatedly across runs is not transient -- but "
                       "verify before acting. Refusals on ranges with no trading days are "
                       "already excluded; confirm the rest actually fail a live fetch "
                       "before concluding anything about the symbol.",
                remedy="probe each with a live daily_bars call over a recent window; "
                       "only names that fail THAT are candidates for removal",
                subjects=repeat, count=len(repeat)))

    out.append(Finding(INFO, "logs", f"{len(logs)} run log(s) scanned",
                       detail=f"{sum(causes.values()):,} warning/error lines, "
                              f"{len(causes)} distinct causes"))
    return out


def check_universe(as_of: pd.Timestamp | None, size: int) -> list[Finding]:
    if as_of is None:
        return [Finding(CRITICAL, "universe", "no universe has ever been built",
                        remedy="run scripts/build_universe.py")]
    age = (pd.Timestamp.now().normalize() - as_of.normalize()).days
    if age > STALE_UNIVERSE_DAYS:
        return [Finding(
            WARN, "universe", f"universe is {age} days old ({as_of.date()})",
            detail="It is meant to be re-scored every morning. A stale universe trades "
                   "yesterday's liquidity ranking, and names that have since fallen "
                   "below the turnover floor stay eligible.",
            remedy="run scripts/daily.py, which refreshes before the open")]
    return [Finding(INFO, "universe",
                    f"{size} names, scored {as_of.date()} ({age}d old)")]


def check_record(cfg) -> list[Finding]:
    """Health of the promoted evidence -- provenance and halts, never P&L."""
    path = STATE_DIR / "paper_record.csv"
    if not path.exists():
        return [Finding(WARN, "evidence", "no promoted paper record",
                        remedy="run scripts/run_paper.py --promote")]
    rec = pd.read_csv(path)
    out: list[Finding] = []

    ok, detail = provenance.verify(path, rec, cfg)
    out.append(Finding(INFO if ok else CRITICAL, "evidence",
                       "provenance verified" if ok else "provenance does not verify",
                       detail=detail,
                       remedy="" if ok else "re-run scripts/run_paper.py --promote"))

    if "slippage_source" in rec:
        srcs = set(rec["slippage_source"].astype(str))
        if srcs == {"modelled"}:
            out.append(Finding(
                WARN, "evidence", "slippage has never been measured independently",
                detail="Observed slippage is the model played back; the fidelity check "
                       "cannot fail.",
                remedy="re-run the replay so decision-to-fill is measured from bars"))

    if "halt_kind" in rec:
        halts = rec.loc[rec["halt_kind"].notna() & (rec["halt_kind"] != ""), "halt_kind"]
        kinds = Counter(halts.astype(str))
        kinds.pop("square_off", None)      # the normal end of every session
        if kinds:
            worst = ", ".join(f"{k} x{n}" for k, n in kinds.most_common())
            frac = sum(kinds.values()) / len(rec)
            out.append(Finding(
                WARN if frac > 0.20 else INFO, "risk",
                f"{sum(kinds.values())} of {len(rec)} sessions ended on a circuit breaker",
                detail=f"{worst}. These are the engine working, not failing -- but a high "
                       f"rate means the rules are firing into conditions the regime "
                       f"classifier is not excluding.",
                remedy="" if frac <= 0.20 else "inspect the regime label on halted sessions"))
    return out


# -------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=None, metavar="PATH", help="also write findings as JSON")
    ap.add_argument("--universe-only", action="store_true",
                    help="skip the log sweep; cache and universe checks only")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    cfg = load_config(args.config)
    ensure_core_importable()
    from research.config import BARS_DIR                             # noqa: PLC0415

    universe, as_of = _universe()
    findings: list[Finding] = []
    findings += check_universe(as_of, len(universe))
    findings += check_cache_audits(universe, Path(BARS_DIR))
    findings += check_coverage_gaps(Path(BARS_DIR), universe)
    findings += check_fundamentals(universe)
    if not args.universe_only:
        findings += check_run_logs(RUNS_DIR)
    findings += check_record(cfg)

    findings.sort(key=lambda f: (_ORDER[f.severity], -f.count))

    print("=" * 78)
    print(f"KABALI OPERATIONAL DIAGNOSIS   {pd.Timestamp.now():%Y-%m-%d %H:%M}")
    print("=" * 78)
    print()
    for f in findings:
        print(f.render())
        print()

    counts = Counter(f.severity for f in findings)
    print("-" * 78)
    print(f"{counts[CRITICAL]} critical, {counts[WARN]} warning, {counts[INFO]} informational")
    print("Nothing here touches strategy parameters. Data and plumbing only.")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(
            {"generated_at": pd.Timestamp.now().isoformat(timespec="seconds"),
             "findings": [asdict(f) for f in findings]}, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")

    return 1 if counts[CRITICAL] else 0


if __name__ == "__main__":
    raise SystemExit(main())
