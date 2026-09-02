"""Assembling twenty years of bhavcopy into an adjusted, survivorship-free panel.

THE IDENTITY PROBLEM. Neither available key is stable over twenty years, and
they break in different places:

  * **Symbol** changes on a rename. AEGISLOG became AEGISCHEM.
  * **ISIN** changes on a face-value split. FEDERALBNK is INE171A01011 before
    its July 2015 bonus and INE171A01029 after; AEGIS appears in bhavcopy under
    INE208C01017 and INE208C01025, switching at the split.

Keying on ISIN therefore severs a company's history at precisely the corporate
actions this module exists to adjust for -- the twelve-month momentum window
would see a name with three months of history and rank it as ineligible, or
worse, rank the stub. Keying on symbol merges genuinely different companies that
reused a ticker.

The fix is to use neither alone. Every (symbol, ISIN) pair ever observed is an
edge in a bipartite graph; a company is a connected component of that graph. A
rename links two symbols through their shared ISIN, a split links two ISINs
through their shared symbol, and a chain of both resolves to one entity.

WHY THAT IS SAFE, AND WHERE IT IS NOT. The failure mode is over-merging: if a
dead ticker is reassigned to an unrelated company years later, the component
chains two firms into one. `suspicious_entities` reports components whose
segments are separated by long gaps so the count is visible rather than assumed
away, and `MAX_LINK_GAP_DAYS` refuses the link outright when the two uses of a
symbol are far enough apart to be different companies.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

OUT_DIR = Path("state")

#: A symbol seen again after this long is treated as a different company rather
#: than a continuation. Two years is comfortably longer than any suspension this
#: panel needs to survive and far shorter than a typical ticker reassignment.
MAX_LINK_GAP_DAYS = 730


class _Union:
    """Union-find over arbitrary hashable labels."""

    def __init__(self) -> None:
        self.parent: dict[object, object] = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


@dataclass
class Entities:
    """Resolved company identities across symbol renames and ISIN changes."""
    isin_to_entity: dict[str, str]
    display: dict[str, str] = field(default_factory=dict)   # entity -> symbol
    merged: int = 0                                         # multi-ISIN entities
    symbol_to_entity: dict[str, str] = field(default_factory=dict)

    def entity_for(self, isin: str | None, symbol: str | None) -> str | None:
        """Resolve an external record to an entity, ISIN first then symbol.

        The fallback is not redundancy, it is required. NSE's corporate-action
        feed cites the ISIN that was retired *by* the action -- FEDERALBNK's
        July 2015 bonus is filed under INE171A01011 while every bhavcopy row
        already carries INE171A01029 -- so an ISIN-only join silently drops
        exactly the splits that most need adjusting.
        """
        if isin and isin in self.isin_to_entity:
            return self.isin_to_entity[isin]
        if symbol and symbol in self.symbol_to_entity:
            return self.symbol_to_entity[symbol]
        return None

    def render(self) -> str:
        n = len(set(self.isin_to_entity.values()))
        return (f"{len(self.isin_to_entity):,} ISINs -> {n:,} entities "
                f"({self.merged:,} span more than one ISIN)")


def resolve_entities(long: pd.DataFrame,
                     max_gap_days: int = MAX_LINK_GAP_DAYS) -> Entities:
    """Connected components over the (symbol, ISIN) graph.

    A symbol whose two uses are separated by more than `max_gap_days` is split
    into distinct nodes before the graph is built, so a reassigned ticker cannot
    chain two unrelated companies together.
    """
    obs = (long.groupby(["symbol", "isin"], observed=True)["date"]
           .agg(["min", "max"]).reset_index())

    # Split a symbol into eras where its uses are far apart in time.
    sym_era: dict[tuple[str, str], str] = {}
    for sym, grp in obs.sort_values("min").groupby("symbol", observed=True):
        era, last_end = 0, None
        for _, r in grp.iterrows():
            if last_end is not None and (r["min"] - last_end).days > max_gap_days:
                era += 1
            sym_era[(sym, r["isin"])] = f"{sym}#{era}"
            last_end = max(last_end, r["max"]) if last_end is not None else r["max"]

    uf = _Union()
    for _, r in obs.iterrows():
        uf.union(("I", r["isin"]), ("S", sym_era[(r["symbol"], r["isin"])]))

    isin_to_entity, groups = {}, {}
    for _, r in obs.iterrows():
        root = uf.find(("I", r["isin"]))
        key = f"{root[1]}" if root[0] == "I" else f"{root[1]}"
        isin_to_entity[r["isin"]] = key
        groups.setdefault(key, set()).add(r["isin"])

    # Display name: the symbol most recently seen for the entity.
    last = (long.sort_values("date").groupby("isin", observed=True)["symbol"].last())
    latest = (long.groupby("isin", observed=True)["date"].max())
    display: dict[str, str] = {}
    best: dict[str, pd.Timestamp] = {}
    for isin, ent in isin_to_entity.items():
        d = latest.get(isin)
        if d is not None and (ent not in best or d > best[ent]):
            best[ent], display[ent] = d, last.get(isin, isin)

    sym_to_entity: dict[str, str] = {}
    sym_seen: dict[str, pd.Timestamp] = {}
    for _, r in obs.iterrows():
        ent = isin_to_entity[r["isin"]]
        prev = sym_seen.get(r["symbol"])
        if prev is None or r["max"] > prev:          # most recent use wins
            sym_seen[r["symbol"]], sym_to_entity[r["symbol"]] = r["max"], ent

    merged = sum(1 for v in groups.values() if len(v) > 1)
    return Entities(isin_to_entity, display, merged, sym_to_entity)


def suspicious_entities(long: pd.DataFrame, ents: Entities,
                        gap_days: int = 180) -> pd.DataFrame:
    """Entities whose observed sessions contain a gap this long.

    Not necessarily wrong -- suspensions happen -- but this is where an
    over-merge would show up, so the number is reported rather than trusted.
    """
    df = long[["date", "isin"]].copy()
    df["entity"] = df["isin"].map(ents.isin_to_entity)
    rows = []
    for ent, grp in df.groupby("entity", observed=True):
        d = grp["date"].drop_duplicates().sort_values()
        if len(d) < 2:
            continue
        g = d.diff().dt.days.max()
        if g and g >= gap_days:
            rows.append({"entity": ent, "max_gap_days": int(g),
                         "sessions": len(d), "symbol": ents.display.get(ent, "")})
    return pd.DataFrame(rows).sort_values("max_gap_days", ascending=False) \
        if rows else pd.DataFrame(columns=["entity", "max_gap_days", "sessions", "symbol"])


# ------------------------------------------------------------------- panels
@dataclass
class Panel:
    close: pd.DataFrame
    open: pd.DataFrame
    volume: pd.DataFrame
    turnover: pd.DataFrame
    entities: Entities
    confirmation: object | None = None

    def render(self) -> str:
        c = self.close
        listed = c.notna().sum(axis=1)
        return (f"panel {c.shape[0]:,} sessions x {c.shape[1]:,} entities | "
                f"{c.index.min().date()} -> {c.index.max().date()} | "
                f"listed names {listed.iloc[0]:,} at start, {listed.iloc[-1]:,} at end")


def build_panel(long: pd.DataFrame, actions: pd.DataFrame | None = None,
                series: tuple[str, ...] = ("EQ",)) -> Panel:
    """Wide, split-adjusted panels keyed by resolved entity.

    Adjustment is applied to prices only. Volume is deliberately left raw:
    turnover (price x quantity) is invariant to a split and is what the
    liquidity screen actually uses, so scaling quantity would corrupt the one
    field that needs no correction.
    """
    from kabali.data.corpactions import adjustment_series, confirm

    df = long[long["series"].isin(series)].copy()
    df["date"] = pd.to_datetime(df["date"])
    ents = resolve_entities(df)
    df["entity"] = df["isin"].map(ents.isin_to_entity)

    # One row per entity per day; a same-day ISIN changeover keeps the last.
    df = df.sort_values(["date", "entity", "isin"])
    grouped = df.groupby(["date", "entity"], observed=True).last().reset_index()

    def wide(col: str) -> pd.DataFrame:
        return grouped.pivot(index="date", columns="entity", values=col).sort_index()

    close, open_ = wide("close"), wide("open")
    volume, turnover = wide("volume"), wide("turnover")

    conf = None
    if actions is not None and not actions.empty:
        # Resolve actions to entities BEFORE confirming, so that an action filed
        # under a retired ISIN still finds its price series.
        acts = actions.copy()
        syms = acts["symbol"] if "symbol" in acts.columns else pd.Series(
            [None] * len(acts), index=acts.index)
        acts["entity"] = [ents.entity_for(i, sy) for i, sy in zip(acts["isin"], syms)]
        acts = acts.dropna(subset=["entity"])
        conf = confirm(acts, df[["date", "entity", "close"]], key="entity")
        cc = conf.confirmed.copy()
        if not cc.empty:
            adj = adjustment_series(cc, close.index, close.columns, key="entity")
            close, open_ = close * adj, open_ * adj

    return Panel(close, open_, volume, turnover, ents, conf)


def write_panel(panel: Panel, out_dir: Path = OUT_DIR, prefix: str = "bhav") -> list[Path]:
    """Persist to `state/<prefix>_{close,open,volume,turnover}.parquet`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, frame in (("close", panel.close), ("open", panel.open),
                        ("volume", panel.volume), ("turnover", panel.turnover)):
        p = out_dir / f"{prefix}_{name}.parquet"
        frame.to_parquet(p)
        written.append(p)
    return written


#: A single-session move this extreme is, in practice, never a real price move:
#: it is a corporate action, or a data error. -0.55 sits below the -50% that a
#: 1:2 action produces and above the tick-floor noise of sub-rupee penny names
#: (BIRLACOT oscillating 0.10 -> 0.05 is -50% and genuinely traded).
JUMP_DOWN, JUMP_UP = -0.55, 1.50


def mask_unexplained_jumps(panel: "Panel", jump_down: float = JUMP_DOWN,
                           jump_up: float = JUMP_UP) -> tuple["Panel", pd.DataFrame]:
    """Blank an entity's history before any jump no confirmed action explains.

    NSE's action feed has genuine holes -- ITDCEM's 1:10 split of 2015-08-21 is
    absent from it entirely, so no parser can recover the factor and the series
    still steps by -90%. The alternative to this function is inferring a factor
    from the jump itself, which cannot distinguish a split from a company losing
    most of its value in a session.

    So the jump is not repaired, it is quarantined: prices *before* it are set to
    NaN, which makes the name a short-history listing rather than a name with a
    fabricated -90% return in its formation window. It becomes rankable again
    once enough post-jump history accumulates. Conservative in the one direction
    that matters -- it removes names from the test rather than inventing returns
    for them.
    """
    r = panel.close.pct_change(fill_method=None)
    hits = (r < jump_down) | (r > jump_up)
    rows = []
    close, open_ = panel.close.copy(), panel.open.copy()
    for ent in hits.columns[hits.any()]:
        when = r.index[hits[ent].fillna(False)]
        last = when.max()
        close.loc[close.index < last, ent] = pd.NA
        open_.loc[open_.index < last, ent] = pd.NA
        rows.append({"entity": ent, "symbol": panel.entities.display.get(ent, ent),
                     "last_jump": last, "jumps": int(hits[ent].sum()),
                     "return": float(r.loc[last, ent])})
    masked = Panel(close, open_, panel.volume, panel.turnover,
                   panel.entities, panel.confirmation)
    report = pd.DataFrame(rows).sort_values("return") if rows else pd.DataFrame(
        columns=["entity", "symbol", "last_jump", "jumps", "return"])
    return masked, report
