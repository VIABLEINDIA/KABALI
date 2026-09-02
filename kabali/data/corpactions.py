"""Corporate actions, and the price factors that make bhavcopy continuous.

BHAVCOPY IS UNADJUSTED. Measured, not assumed: across 90,525 consecutive-day
pairs in Q3 2015, `prev_close` equalled the raw prior close 99.98% of the time,
including on every confirmed split -- AEGISCHEM 814.60 -> 80.60, BEL 3343 ->
1094, KOTAKBANK and FEDERALBNK both halving in July 2015. The field is a copy of
yesterday's close, not an adjusted one. Left alone, a 1:10 split enters a
momentum ranking as a -90% return and the name is bought as a "loser" or sold as
a "winner" for a reason that never happened.

The residual 0.02% of mismatches were names resuming after 29-64 day trading
gaps, not corporate actions. Gaps, not adjustments.

WHY PARSE *AND* VERIFY. NSE's action feed carries a free-text `subject` whose
vocabulary is genuinely irregular -- "Bonus 1:1", "Bonus 3:4", "Fv Split-Rs.10/-
To Rs.2/-", combined actions in one string ("Spl-Rs10 To Rs2/Bonus-1:2"), and
some values truncated mid-number by an upstream field width. A parser over that
will be wrong sometimes and cannot know when. So the parsed factor is treated as
a *hypothesis* and confirmed against the price jump actually observed on the
ex-date. A factor that both the text and the tape agree on is applied; one they
disagree on is refused and reported. The confirmation rate is a data-quality
number the panel builder prints rather than hides.

The opposite design -- inferring factors from price jumps alone -- is rejected
on purpose: a genuine 50% crash and a 1:2 split are the same number, and there
were 62 one-day drops beyond -30% in a single quarter.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

CACHE = Path("data/corpactions")
_API = ("https://www.nseindia.com/api/corporates-corporateActions"
        "?index=equities&from_date={a}&to_date={b}")
_PRIME = "https://www.nseindia.com/companies-listing/corporate-filings-actions"

#: A parsed factor is applied only if the observed ex-date price ratio agrees
#: within this relative tolerance. Wide enough to absorb a real move on top of
#: the action, tight enough that a 1:2 and a 1:10 can never be confused.
TOLERANCE = 0.25

_BONUS = re.compile(r"bonus[^0-9]{0,8}(\d+)\s*:\s*(\d+)", re.I)

#: Split detection is two-stage on purpose. The vocabulary is too irregular for
#: one pattern -- "Fv Split Rs.10/- To Re.1/-", "Split-Rs.10tors.2", "Spl-Rs10
#: To Rs2" and "Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs
#: 1/- Per Share" all mean the same thing. So: confirm the string is about a
#: split at all, then find the "A ... to ... B" face-value pair anywhere in it.
_IS_SPLIT = re.compile(r"split|sub-?division|\bspl", re.I)
_FV_PAIR = re.compile(
    r"(?:rs\.?|re\.?)\s*(\d+(?:\.\d+)?)"      # from face value
    r"[^0-9]{0,24}?"                             # "/- Per Share ", "tors.", "-"
    r"to\s*(?:rs\.?|re\.?)?\s*(\d+(?:\.\d+)?)",  # to face value
    re.I)
#: Same pair without the currency prefix ("Spl-Rs10 To Rs2" covered above;
#: this catches "Split 10 to 1").
_FV_BARE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:/-)?\s*to\s*(\d+(?:\.\d+)?)", re.I)


def parse_factor(subject: str) -> tuple[float | None, str]:
    """Price multiplier implied by an action string, and what was recognised.

    Returns (factor, kind). A factor of 0.5 means prices before the ex-date must
    be halved to line up with prices after it. `None` means nothing priceable
    was recognised -- dividends, AGMs and interest payments land here by design.

    Indian convention: "Bonus a:b" is a new shares for every b held, so the
    share count goes b -> a+b and the price factor is b/(a+b). A face-value
    split from A to B multiplies the price by B/A. Both can appear at once.
    """
    if not subject:
        return None, ""
    factor, kinds = 1.0, []
    m = _BONUS.search(subject)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        if a > 0 and b > 0:
            factor *= b / (a + b)
            kinds.append(f"bonus {m.group(1)}:{m.group(2)}")
    if _IS_SPLIT.search(subject):
        m = _FV_PAIR.search(subject) or _FV_BARE.search(subject)
        if m:
            old, new = float(m.group(1)), float(m.group(2))
            if old > 0 and new > 0 and new < old:
                factor *= new / old
                kinds.append(f"split {m.group(1)}->{m.group(2)}")
    if not kinds:
        return None, ""
    return factor, " + ".join(kinds)


# ------------------------------------------------------------------ fetching
def fetch_actions(start: date, end: date, session=None,
                  cache_dir: Path = CACHE, refresh: bool = False) -> pd.DataFrame:
    """Every corporate action in range, in 90-day windows (the API's practical
    limit), cached per window."""
    from kabali.data.bhavcopy import make_session

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    s = session
    frames = []
    a = start
    while a <= end:
        b = min(a + timedelta(days=89), end)
        pq = cache_dir / f"ca_{a:%Y%m%d}_{b:%Y%m%d}.parquet"
        if pq.exists() and not refresh:
            frames.append(pd.read_parquet(pq))
        else:
            if s is None:
                s = make_session()
                s.headers["Accept"] = "application/json"
                try:
                    s.get(_PRIME, timeout=20)
                except Exception:                        # noqa: BLE001
                    pass
                time.sleep(1.0)
            url = _API.format(a=f"{a:%d-%m-%Y}", b=f"{b:%d-%m-%Y}")
            try:
                r = s.get(url, timeout=40)
                rows = r.json() if r.status_code == 200 and r.content else []
                if isinstance(rows, dict):
                    rows = rows.get("data", [])
            except Exception as exc:                     # noqa: BLE001
                log.warning("corporate actions %s..%s: %s", a, b, exc)
                rows = []
            df = pd.DataFrame(rows)
            if not df.empty:
                df = df[[c for c in ("isin", "symbol", "series", "exDate", "subject")
                         if c in df.columns]]
            df.to_parquet(pq, index=False)
            frames.append(df)
            time.sleep(1.2)
        a = b + timedelta(days=1)

    out = pd.concat([f for f in frames if not f.empty], ignore_index=True) \
        if any(not f.empty for f in frames) else pd.DataFrame()
    if out.empty:
        return pd.DataFrame(columns=["isin", "symbol", "ex_date", "subject",
                                     "factor", "kind"])
    out = out.rename(columns={"exDate": "ex_date"})
    out["ex_date"] = pd.to_datetime(out["ex_date"], format="%d-%b-%Y",
                                    errors="coerce")
    out = out.dropna(subset=["ex_date", "isin"])
    parsed = out["subject"].fillna("").map(parse_factor)
    out["factor"] = [p[0] for p in parsed]
    out["kind"] = [p[1] for p in parsed]
    return out.drop_duplicates(subset=["isin", "ex_date", "subject"]) \
              .reset_index(drop=True)


# ----------------------------------------------------------------- verifying
@dataclass
class Confirmation:
    """Which parsed factors the tape agrees with.

    `out_of_range` is kept apart from `rejected` on purpose. An action whose
    ex-date has no bar in the panel -- because it falls outside the window, or
    the name was not listed then -- was never testable, and folding it into the
    refusal count makes the quality metric depend on how far past the panel edge
    the action feed was queried. The number that means something is the rate
    among actions that *could* be checked.
    """
    confirmed: pd.DataFrame
    rejected: pd.DataFrame
    out_of_range: pd.DataFrame | None = None

    def rate(self) -> float:
        n = len(self.confirmed) + len(self.rejected)
        return float(len(self.confirmed)) / n if n else float("nan")

    def render(self) -> str:
        n = len(self.confirmed) + len(self.rejected)
        oor = 0 if self.out_of_range is None else len(self.out_of_range)
        tail = f", {oor} outside the panel" if oor else ""
        return (f"corporate actions: {len(self.confirmed)}/{n} price-confirmed "
                f"({self.rate():.1%}), {len(self.rejected)} contradicted{tail}")


def confirm(actions: pd.DataFrame, long_prices: pd.DataFrame,
            tolerance: float = TOLERANCE, key: str = "isin") -> Confirmation:
    """Keep only factors the observed ex-date price ratio agrees with.

    `key` is the column both frames are joined on. The panel builder passes the
    resolved *entity* rather than "isin", because NSE files an action under the
    ISIN the action retires -- joining on the raw ISIN drops precisely the
    splits that matter most.

    `long_prices` is the tidy bhavcopy frame: date / <key> / close. The observed
    ratio is close(ex_date) / close(previous listed session), which for a clean
    split lands on the parsed factor. Anything the tape contradicts is refused
    rather than applied -- an unverified factor is more dangerous than none,
    because it silently manufactures a return.
    """
    acts = actions.dropna(subset=["factor"]).copy()
    if acts.empty:
        empty = acts.assign(observed=pd.Series(dtype=float))
        return Confirmation(empty, empty, empty)

    px = long_prices[["date", key, "close"]].dropna().sort_values([key, "date"])
    px["prev"] = px.groupby(key)["close"].shift(1)
    px["observed"] = px["close"] / px["prev"]

    m = acts.merge(px[["date", key, "observed"]],
                   left_on=[key, "ex_date"], right_on=[key, "date"],
                   how="left").drop(columns=["date"])
    testable = m["observed"].notna()
    ok = testable & ((m["observed"] - m["factor"]).abs() <= tolerance * m["factor"])
    return Confirmation(m[ok].reset_index(drop=True),
                        m[testable & ~ok].reset_index(drop=True),
                        m[~testable].reset_index(drop=True))


def adjustment_series(confirmed: pd.DataFrame, dates: pd.DatetimeIndex,
                      isins: pd.Index, key: str = "isin") -> pd.DataFrame:
    """Cumulative back-adjustment multiplier, one column per ISIN.

    A price on date t is multiplied by the product of every confirmed factor
    whose ex-date is strictly after t, which makes the series continuous through
    the action while leaving the most recent prices at their traded values.
    """
    adj = pd.DataFrame(1.0, index=dates, columns=isins)
    if confirmed.empty:
        return adj
    for isin, grp in confirmed.groupby(key):
        if isin not in adj.columns:
            continue
        col = pd.Series(1.0, index=dates)
        for _, row in grp.iterrows():
            col.loc[dates < row["ex_date"]] *= float(row["factor"])
        adj[isin] = col
    return adj
