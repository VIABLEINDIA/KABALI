"""Provenance for the evidence the live gate reads.

The gate re-scores `state/paper_record.csv` on every run, which makes that file
the most consequential artifact in the repo: it is the only thing standing
between a config edit and real orders. Until now it carried no record of what
produced it, and `run_paper.py` overwrote it unconditionally. Three failures
followed from that, all of them silent:

  WHICH RUN?   `runs/` holds paper_final, paper_v3, paper_full2, four smoke
               runs. Nothing in the record says which one it came from, and the
               directory names actively mislead -- `paper_final` sounds
               authoritative but `paper_v3` ran fourteen minutes later and is
               what the gate was actually scoring.

  WHICH CODE?  `paper_final` was replayed one minute before the daily-loss-limit
               fix landed in `risk/circuit.py`, so its record described a risk
               engine that no longer exists. A record produced by different risk
               or strategy code is not evidence about the system you are about
               to run; it is evidence about a system you deleted.

  WHOSE RUN?   A two-day smoke test wrote the same path as a sixty-four-session
               replay. Evidence could be replaced by something that was never
               meant to be evidence, and the gate would score it without
               noticing the substitution.

So promotion is now explicit (`run_paper.py --promote`) and every promotion
writes a sidecar recording the fingerprints. The gate compares them against the
installed code and the loaded config and refuses a mismatch.

FINGERPRINTS ARE ADVISORY ABOUT INTENT, NOT SECURITY. Anyone who can edit the
record can edit the sidecar. Like the confirmation phrase in the gate file, this
is friction placed where an irreversible decision is made -- it stops an
accident, not an operator who has decided to lie to themselves.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from kabali.config import ROOT

#: Modules whose behaviour determines what a paper session records. A change in
#: any of them means an existing record describes a different system. Sizing,
#: circuit breakers, fills and the strategies themselves qualify; reporting and
#: CLI plumbing do not, because they cannot change a trade.
FINGERPRINTED = (
    "kabali/risk",
    "kabali/strategies",
    "kabali/regime",
    "kabali/universe",
    "kabali/engine/session.py",
    "kabali/execution/paper.py",
)


def _digest(pairs: list[tuple[str, bytes]]) -> str:
    """Stable 12-hex digest over (name, content) pairs, sorted by name."""
    h = hashlib.sha256()
    for name, blob in sorted(pairs):
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(blob)
        h.update(b"\0")
    return h.hexdigest()[:12]


def code_fingerprint(root: Path | None = None) -> str:
    """Digest of the trading logic that produces a paper record.

    Line endings are normalised and `__pycache__` skipped so the same checkout
    fingerprints identically on Windows and Linux, and so a stale .pyc cannot
    move the digest.
    """
    base = Path(root or ROOT)
    pairs: list[tuple[str, bytes]] = []
    for entry in FINGERPRINTED:
        p = base / entry
        files = sorted(p.rglob("*.py")) if p.is_dir() else ([p] if p.exists() else [])
        for f in files:
            if "__pycache__" in f.parts:
                continue
            rel = f.relative_to(base).as_posix()
            pairs.append((rel, f.read_bytes().replace(b"\r\n", b"\n")))
    return _digest(pairs)


def config_fingerprint(cfg) -> str:
    """Digest of the config fields that shape a paper session.

    `execution` is excluded on purpose: paper and live differ by exactly that
    section, and a record must stay valid evidence when the operator flips the
    mode -- that flip is the decision the gate exists to judge. `swing` is
    excluded because it drives a different venue that never writes this record.
    """
    payload = {
        name: asdict(getattr(cfg, name))
        for name in ("account", "risk", "session", "universe", "regime", "costs")
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return _digest([("config", blob)])


def meta_path(record: Path) -> Path:
    return Path(record).with_suffix(".meta.json")


def describe(sessions: pd.DataFrame) -> dict:
    """The summary a sidecar carries so tampering with the CSV is detectable."""
    dates = pd.to_datetime(sessions["trade_date"])
    return {
        "sessions": int(len(sessions)),
        "trades": int(sessions["trades"].sum()) if "trades" in sessions else 0,
        "net_pnl": round(float(sessions["net_pnl"].sum()), 2)
        if "net_pnl" in sessions else 0.0,
        "first_date": str(dates.min().date()),
        "last_date": str(dates.max().date()),
    }


def write(record: Path, sessions: pd.DataFrame, cfg, source: str) -> Path:
    """Write the record and its provenance sidecar together.

    Both or neither: a record without a sidecar fails the gate, so writing the
    CSV alone would close the gate rather than leave it wrongly open. The
    ordering still matters -- sidecar last, so an interrupted write leaves the
    old sidecar disagreeing with the new record and the gate reports tampering
    rather than accepting an unverified file.
    """
    record = Path(record)
    record.parent.mkdir(parents=True, exist_ok=True)
    sessions.to_csv(record, index=False)
    payload = {
        "source": source,
        "written_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "code_fingerprint": code_fingerprint(),
        "config_fingerprint": config_fingerprint(cfg),
        "config_source": str(getattr(cfg, "source", "")),
        **describe(sessions),
    }
    meta_path(record).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return record


def read(record: Path) -> dict | None:
    p = meta_path(record)
    if not p.exists():
        return None
    try:
        blob = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return blob if isinstance(blob, dict) else None


def verify(record: Path, sessions: pd.DataFrame, cfg) -> tuple[bool, str]:
    """Check a record against its sidecar, the installed code and the config.

    Returns (ok, detail). Every failure names the remedy, because the only
    correct response to any of them is to re-run the replay and promote it --
    never to edit the sidecar until it agrees.
    """
    meta = read(record)
    if meta is None:
        return False, (f"no provenance sidecar at {meta_path(record).name} -- "
                       f"re-run `scripts/run_paper.py --promote`")

    actual = describe(sessions)
    drift = [k for k in ("sessions", "trades", "first_date", "last_date")
             if meta.get(k) != actual[k]]
    if drift:
        return False, (f"record does not match its sidecar ({', '.join(drift)} differ) "
                       f"-- it was edited or partially written after promotion")

    have_code = code_fingerprint()
    if meta.get("code_fingerprint") != have_code:
        return False, (f"evidence was produced by different trading code "
                       f"({meta.get('code_fingerprint')}, installed {have_code}) -- "
                       f"re-run the replay against the current code")

    have_cfg = config_fingerprint(cfg)
    if meta.get("config_fingerprint") != have_cfg:
        return False, (f"evidence was produced under a different config "
                       f"({meta.get('config_fingerprint')}, loaded {have_cfg}) -- "
                       f"re-run the replay under the current config")

    return True, (f"from {meta.get('source', 'unknown')} at "
                  f"{str(meta.get('written_at', ''))[:16]}, code {have_code}, "
                  f"config {have_cfg}")
