"""Direct, offline reads of the bar cache.

`research.datastore.load_bars` is the right tool while ingesting: when the cache
does not cover the window it asks Dhan for the rest. That is exactly wrong for
the stages that come AFTER ingest.

The universe build learned this the expensive way. Its factor loop called
`load_bars` for every survivor, so the symbols whose download had failed were
silently re-requested one at a time -- against an account Dhan was already
throttling. A step that is pure arithmetic over local parquet turned into hours
of blocked network calls, with no log line to say why, because from the outside
a slow fetch and a slow computation look identical.

So anything downstream of ingest reads through here instead. `cached_bars`
touches the filesystem and nothing else: a symbol that is not on disk returns
None immediately and the caller skips it. Missing data becomes a countable,
reportable outcome rather than an unbounded wait.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from kabali.core import ultimate_root

log = logging.getLogger(__name__)

# Mirrors research.config.BARS_DIR and research.datastore._cache_path. Asserted
# against the real thing in `verify_layout` so a change upstream is caught here
# rather than as a universe that mysteriously shrinks to zero.
BARS_DIRNAME = ("data", "bars")


def bars_dir() -> Path:
    return ultimate_root().joinpath(*BARS_DIRNAME)


def cache_path(security_id: str, segment: str, timeframe: str) -> Path:
    safe_tf = str(timeframe).replace("/", "_")
    return bars_dir() / f"{segment}_{security_id}_{safe_tf}.parquet"


def verify_layout() -> bool:
    """Confirm this module and `research.datastore` agree on the cache path."""
    from research.datastore import _cache_path
    mine = cache_path("2885", "NSE_EQ", "D")
    theirs = Path(_cache_path("2885", "NSE_EQ", "D"))
    if mine != theirs:
        log.error("cache path drift: kabali=%s research=%s", mine, theirs)
        return False
    return True


def cached_bars(instrument, timeframe: str = "D",
                start: str | None = None, end: str | None = None,
                min_bars: int = 1) -> pd.DataFrame | None:
    """Read one symbol's cached bars, or None. Never touches the network.

    `start`/`end` slice what is on disk; they do NOT cause a fetch of anything
    missing. A caller that needs guaranteed coverage must ingest first and check
    the ingest report.
    """
    path = cache_path(instrument.security_id, instrument.exchange_segment, timeframe)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as exc:                              # noqa: BLE001
        log.warning("unreadable cache %s: %s", path.name, exc)
        return None
    if df is None or df.empty or "timestamp" not in df.columns:
        return None

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if start is not None:
        df = df[df["timestamp"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["timestamp"] <= pd.Timestamp(end) + pd.Timedelta(1, unit="D")]
    if len(df) < min_bars:
        return None
    return df.sort_values("timestamp").reset_index(drop=True)


def load_many(instruments, timeframe: str = "D", start: str | None = None,
              end: str | None = None, min_bars: int = 1) -> tuple[dict, list[str]]:
    """Read many symbols from cache. Returns (frames, missing_symbols)."""
    frames, missing = {}, []
    for inst in instruments:
        df = cached_bars(inst, timeframe, start, end, min_bars)
        if df is None:
            missing.append(inst.symbol)
        else:
            frames[inst.symbol] = df
    return frames, missing


def coverage(instruments, timeframe: str = "D") -> pd.DataFrame:
    """Per-symbol cache inventory: rows and date span, without fetching."""
    rows = []
    for inst in instruments:
        df = cached_bars(inst, timeframe)
        rows.append({
            "symbol": inst.symbol,
            "security_id": inst.security_id,
            "cached": df is not None,
            "bars": 0 if df is None else len(df),
            "first": None if df is None else df["timestamp"].min(),
            "last": None if df is None else df["timestamp"].max(),
        })
    return pd.DataFrame(rows)
