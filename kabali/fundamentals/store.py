"""The fundamentals cache: read, merge, and be honest about its age.

One JSON file at `state/fundamentals.json`, keyed by NSE symbol. Small enough to
read whole, human-readable enough to inspect when a number looks wrong, and
append-only in practice so a partial refresh never destroys what it did not
cover.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from kabali.config import STATE_DIR

STORE_PATH = STATE_DIR / "fundamentals.json"

#: Company facts move slowly, but market cap moves with price every day. Past
#: this, a record is reported as stale rather than quietly used.
STALE_AFTER_DAYS = 30


@dataclass(frozen=True)
class Fundamentals:
    """What is known about one listed company, and when it was known."""

    symbol: str
    full_name: str = ""
    isin: str = ""
    bse_code: str = ""
    market_cap_cr: float | None = None      # rupees crore, as reported
    category: str = ""
    sector: str = ""
    industry: str = ""
    fetched_at: str = ""
    source: str = "bull-ai"

    @property
    def age_days(self) -> float | None:
        if not self.fetched_at:
            return None
        try:
            return (pd.Timestamp.now() - pd.Timestamp(self.fetched_at)).total_seconds() / 86400
        except (ValueError, TypeError):
            return None

    @property
    def stale(self) -> bool:
        age = self.age_days
        return age is None or age > STALE_AFTER_DAYS


class FundamentalsStore:
    """Load, merge and query the cache. Never fetches -- see the package docstring."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path or STORE_PATH)
        self._records: dict[str, Fundamentals] = {}
        self.load()

    # ------------------------------------------------------------------ io
    def load(self) -> "FundamentalsStore":
        if not self.path.exists():
            self._records = {}
            return self
        try:
            blob = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._records = {}
            return self
        self._records = {
            k: Fundamentals(**{**v, "symbol": k})
            for k, v in (blob.get("companies") or {}).items()
            if isinstance(v, dict)
        }
        return self

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "written_at": pd.Timestamp.now().isoformat(timespec="seconds"),
            "count": len(self._records),
            "companies": {k: {kk: vv for kk, vv in asdict(v).items() if kk != "symbol"}
                          for k, v in sorted(self._records.items())},
        }
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return self.path

    def merge(self, records) -> int:
        """Add or replace records. Returns how many were written.

        Replace rather than update-in-place: a partial record from a failed fetch
        should not inherit fields from an older one and present as complete.
        """
        n = 0
        for r in records:
            if not r.symbol:
                continue
            self._records[r.symbol] = r
            n += 1
        return n

    # --------------------------------------------------------------- query
    def get(self, symbol: str) -> Fundamentals | None:
        return self._records.get(symbol.upper())

    def __len__(self) -> int:
        return len(self._records)

    def coverage(self, symbols) -> dict:
        """How much of a universe this store actually covers, and how fresh."""
        syms = [s.upper() for s in symbols]
        have = [s for s in syms if s in self._records]
        fresh = [s for s in have if not self._records[s].stale]
        ages = [self._records[s].age_days for s in have
                if self._records[s].age_days is not None]
        return {
            "requested": len(syms),
            "covered": len(have),
            "fresh": len(fresh),
            "stale": len(have) - len(fresh),
            "missing": len(syms) - len(have),
            "median_age_days": round(float(pd.Series(ages).median()), 1) if ages else None,
        }

    def frame(self, symbols=None) -> pd.DataFrame:
        """Records as a frame, indexed by symbol, for joining onto factors."""
        recs = [self._records[s.upper()] for s in symbols
                if s.upper() in self._records] if symbols is not None \
            else list(self._records.values())
        if not recs:
            return pd.DataFrame(columns=["market_cap_cr", "sector", "category",
                                         "stale", "age_days"]).rename_axis("symbol")
        df = pd.DataFrame([asdict(r) for r in recs]).set_index("symbol")
        df["age_days"] = [r.age_days for r in recs]
        df["stale"] = [r.stale for r in recs]
        return df

    def render(self, symbols=None) -> str:
        c = self.coverage(symbols if symbols is not None else list(self._records))
        age = f"{c['median_age_days']}d" if c["median_age_days"] is not None else "n/a"
        return (f"fundamentals: {c['covered']}/{c['requested']} covered "
                f"({c['fresh']} fresh, {c['stale']} stale, {c['missing']} missing), "
                f"median age {age}")


def validate_against_price(store: "FundamentalsStore", prices: dict) -> list[str]:
    """Cross-check market cap against price by implying the share count.

    Market cap arrives as a single number with no way to tell a correct one from
    a wrong one by inspection -- and a wrong one ranks, filters and sorts exactly
    like a right one. But market cap and price are not independent: their ratio
    is shares outstanding, which for a listed Indian company sits in a knowable
    band. A name whose implied count falls far outside it is reporting a market
    cap that does not belong to the security being traded.

    This catches the failure that matters -- a units error or a stale corporate
    action -- without needing a second data source to compare against.
    """
    LO, HI = 0.05e7, 5_000e7          # 5 lakh to 500 crore shares
    problems = []
    for sym, rec in store._records.items():
        px = prices.get(sym)
        if not px or not rec.market_cap_cr:
            continue
        implied = rec.market_cap_cr * 1e7 / px
        if not (LO <= implied <= HI):
            problems.append(
                f"{sym}: market cap Rs {rec.market_cap_cr:,.0f}cr at Rs {px:,.2f} "
                f"implies {implied / 1e7:,.1f} crore shares -- outside the plausible band")
    return problems
