"""NSE bhavcopy: the survivorship-free daily panel.

WHY THIS EXISTS. The Dhan panel reaches back to 2021 and holds only names
listed *today*. Both facts are fatal to a cross-sectional test:
`docs/xsection_hypothesis.md` measures that even a beta-neutral 100x100 spread
needs ~22 years to resolve a 5%/yr effect, and the missing delisted names bias
the benchmark more than the strategy. Bhavcopy fixes both at once -- it is an
as-of-the-date snapshot, so a company that vanished in 2009 is present in every
file printed before it vanished and absent afterwards, which is exactly what a
point-in-time universe means.

TWO FORMATS, ONE CUTOVER. NSE replaced the legacy per-day CSV with the UDiFF
schema in mid-2024 and stopped publishing legacy some weeks later. Both are
read here and normalised to one frame. Both are always tried; `UDIFF_FROM` only
decides which is asked *first*, because asking UDiFF for a 2006 date is a
guaranteed 404 and a throttle sleep, and across eighteen pre-UDiFF years that
doomed request was most of the backfill's runtime.

THE KEY IS ISIN, NOT SYMBOL -- EXCEPT BEFORE 2011, WHEN THERE IS NO ISIN. Over
twenty years symbols get reused and renamed, so a ticker string is not a
company. ISIN is stable across renames and is the panel's key. But the legacy
file only gained an ISIN column between 2010 and 2012: every file before that
carries SYMBOL and nothing else. Those rows get a `SYM:` stand-in key, and
`panel.py` bridges the eras through the shared symbol, so a company trading in
2006 and 2015 resolves to one entity across the changeover.

THREE SCHEMAS, NOT TWO. Pre-2011 legacy (no ISIN), post-2011 legacy (ISIN and
TOTALTRADES added), and UDiFF from mid-2024. All three normalise to one frame.

RATE LIMITS ARE REAL. archives.nseindia.com returns 403 with an empty body to
unprimed clients and to clients that ask too fast. That 403 is indistinguishable
from "file does not exist" unless you know to prime a session against the site
first and pace the requests -- an early probe here read the 403s on pre-2019
dates as missing archives, and concluded twenty years of history was
unavailable. It is available. The session below primes and paces.
"""
from __future__ import annotations

import io
import logging
import time
import zipfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

CACHE_DIR = Path("data/bhavcopy")

#: Stand-in key for the pre-2011 legacy files, which carry no ISIN column.
SYNTH_PREFIX = "SYM:"

#: First date to try UDiFF before legacy. The two formats overlapped for some
#: weeks in mid-2024, so this only picks which is asked first, never which is
#: accepted -- both are always tried.
UDIFF_FROM = date(2024, 6, 1)

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
_HEADERS = {"User-Agent": _UA, "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/all-reports"}

#: Normalised column set every reader returns, in this order.
COLUMNS = ["date", "isin", "symbol", "series", "open", "high", "low",
           "close", "prev_close", "volume", "turnover"]

#: Politeness delay between live downloads, seconds. Below ~0.4 the archive
#: starts returning empty 403s; the cache means this is paid once per day-file.
THROTTLE = 0.5


def _legacy_url(d: date) -> str:
    return (f"https://archives.nseindia.com/content/historical/EQUITIES/"
            f"{d:%Y}/{d.strftime('%b').upper()}/"
            f"cm{d.strftime('%d%b%Y').upper()}bhav.csv.zip")


def _udiff_url(d: date) -> str:
    return (f"https://nsearchives.nseindia.com/content/cm/"
            f"BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.csv.zip")


def make_session() -> requests.Session:
    """A session primed against the site, as the archive host requires."""
    s = requests.Session()
    s.headers.update(_HEADERS)
    try:
        s.get("https://www.nseindia.com/all-reports", timeout=20)
    except requests.RequestException as exc:            # noqa: BLE001
        log.debug("priming request failed, continuing: %s", exc)
    return s


# --------------------------------------------------------------- normalising
def _from_legacy(raw: pd.DataFrame, d: date) -> pd.DataFrame:
    # ISIN and TOTALTRADES were added to the legacy file between 2010 and 2012;
    # earlier years carry neither. Rather than drop a decade, synthesise a key
    # from the symbol. The entity graph in `panel.py` then bridges the eras --
    # a pre-2011 SYM: key and the real ISIN that appears later share a symbol,
    # so they resolve to one company. The cost is that in the pre-ISIN era a
    # ticker reused by two firms cannot be told apart; MAX_LINK_GAP_DAYS is the
    # only guard there, which is why `--start` before 2011 is a coarser panel.
    symbol = raw["SYMBOL"].astype("string").str.strip()
    isin = (raw["ISIN"].astype("string").str.strip() if "ISIN" in raw.columns
            else pd.Series(pd.NA, index=raw.index, dtype="string"))
    isin = isin.fillna(SYNTH_PREFIX + symbol).replace("", pd.NA) \
               .fillna(SYNTH_PREFIX + symbol)
    out = pd.DataFrame({
        "date": pd.Timestamp(d),
        "isin": isin,
        "symbol": symbol,
        "series": raw["SERIES"].astype("string").str.strip(),
        "open": raw["OPEN"], "high": raw["HIGH"], "low": raw["LOW"],
        "close": raw["CLOSE"], "prev_close": raw["PREVCLOSE"],
        "volume": raw["TOTTRDQTY"], "turnover": raw["TOTTRDVAL"],
    })
    return out


def _from_udiff(raw: pd.DataFrame, d: date) -> pd.DataFrame:
    # UDiFF carries the whole cash segment; equities are FinInstrmTp == STK.
    if "FinInstrmTp" in raw.columns:
        raw = raw[raw["FinInstrmTp"].astype("string").str.strip() == "STK"]
    out = pd.DataFrame({
        "date": pd.Timestamp(d),
        "isin": raw["ISIN"].astype("string").str.strip(),
        "symbol": raw["TckrSymb"].astype("string").str.strip(),
        "series": raw["SctySrs"].astype("string").str.strip(),
        "open": raw["OpnPric"], "high": raw["HghPric"], "low": raw["LwPric"],
        "close": raw["ClsPric"], "prev_close": raw["PrvsClsgPric"],
        "volume": raw["TtlTradgVol"], "turnover": raw["TtlTrfVal"],
    })
    return out


def _read_zip_csv(payload: bytes) -> pd.DataFrame:
    z = zipfile.ZipFile(io.BytesIO(payload))
    return pd.read_csv(z.open(z.namelist()[0]), low_memory=False)


def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["isin", "close"])
    # Real ISINs are 12 characters; SYM: keys stand in for the pre-2011 era.
    df = df[(df["isin"].str.len() == 12)
            | df["isin"].str.startswith(SYNTH_PREFIX)]
    for c in ("open", "high", "low", "close", "prev_close", "volume", "turnover"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[df["close"] > 0]
    return df[COLUMNS].reset_index(drop=True)


# ------------------------------------------------------------------ fetching
@dataclass
class DayResult:
    """One calendar day. `frame` is None on a non-trading day."""
    day: date
    frame: pd.DataFrame | None
    source: str            # "cache" | "udiff" | "legacy" | "holiday"


def fetch_day(d: date, session: requests.Session | None = None,
              cache_dir: Path = CACHE_DIR, refresh: bool = False) -> DayResult:
    """One day's equities, normalised and cached.

    A 404 from both formats means a non-trading day, and is cached as such --
    the alternative is re-requesting every weekend and holiday on every rebuild,
    which over twenty years is thousands of pointless round trips.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pq = cache_dir / f"{d:%Y%m%d}.parquet"
    empty = cache_dir / f"{d:%Y%m%d}.holiday"

    if not refresh:
        if pq.exists():
            return DayResult(d, pd.read_parquet(pq), "cache")
        if empty.exists():
            return DayResult(d, None, "cache")

    s = session or make_session()
    # A day is only a non-trading day if the archive says the file is ABSENT.
    # A file that downloads but will not parse is a bug in this reader, and
    # caching it as a holiday would bake that bug in permanently -- the day is
    # never retried, and a decade quietly disappears from the panel. So the two
    # outcomes are tracked apart and only genuine absence is cached.
    found_but_unreadable = False
    # Try the format that era actually used first. Both are still attempted, so
    # ordering cannot change the result -- but it halves the wall clock of a
    # long backfill. Asking UDiFF for a 2006 date is a guaranteed 404 followed
    # by a throttle sleep, and over eighteen pre-UDiFF years that doomed request
    # was most of the runtime (3.0 hours measured, against 1.6 with this order).
    readers = [("udiff", _udiff_url(d), _from_udiff),
               ("legacy", _legacy_url(d), _from_legacy)]
    if d < UDIFF_FROM:
        readers.reverse()
    for name, url, parse in readers:
        try:
            r = s.get(url, timeout=30)
        except requests.RequestException as exc:        # noqa: BLE001
            log.warning("%s %s: %s", d, name, exc)
            found_but_unreadable = True                 # unknown, so do not cache
            continue
        time.sleep(THROTTLE)
        if r.status_code != 200 or not r.content:
            continue
        try:
            frame = _tidy(parse(_read_zip_csv(r.content), d))
        except Exception as exc:                        # noqa: BLE001
            log.error("%s %s: downloaded but unreadable (%s)", d, name, exc)
            found_but_unreadable = True
            continue
        if frame.empty:
            log.error("%s %s: parsed to zero rows", d, name)
            found_but_unreadable = True
            continue
        frame.to_parquet(pq, index=False)
        return DayResult(d, frame, name)

    if found_but_unreadable:
        return DayResult(d, None, "error")
    empty.touch()
    return DayResult(d, None, "holiday")


def trading_days(start: date, end: date) -> list[date]:
    """Weekdays in range. Holidays are discovered by fetching, not predicted."""
    days, d = [], start
    while d <= end:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def fetch_range(start: date, end: date, cache_dir: Path = CACHE_DIR,
                progress_every: int = 100) -> tuple[int, int]:
    """Populate the cache across a date range. Returns (trading, non-trading).

    Resumable: anything already cached costs no request, so an interrupted pull
    is restarted by running it again.
    """
    s = make_session()
    days = trading_days(start, end)
    hit = miss = err = 0
    for i, d in enumerate(days, 1):
        res = fetch_day(d, s, cache_dir)
        if res.source == "error":
            err += 1
        elif res.frame is None:
            miss += 1
        else:
            hit += 1
        if progress_every and i % progress_every == 0:
            log.info("bhavcopy %s (%d/%d) %d trading, %d non-trading, %d error",
                     d, i, len(days), hit, miss, err)
    if err:
        log.error("%d day(s) downloaded but could not be read -- NOT cached, "
                  "so they are retried on the next run", err)
    return hit, miss


def load_cached(start: date | None = None, end: date | None = None,
                cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    """Every cached day in range, concatenated long. No network."""
    cache_dir = Path(cache_dir)
    files = sorted(cache_dir.glob("*.parquet"))
    keep = []
    for f in files:
        d = pd.Timestamp(f.stem).date()
        if start and d < start:
            continue
        if end and d > end:
            continue
        keep.append(f)
    if not keep:
        raise FileNotFoundError(f"no cached bhavcopy in {cache_dir} for that range")
    return pd.concat((pd.read_parquet(f) for f in keep), ignore_index=True)
