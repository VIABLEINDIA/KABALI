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

import numpy as np
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
    return resolve_from_observations(obs, max_gap_days)


def resolve_from_observations(obs: pd.DataFrame,
                              max_gap_days: int = MAX_LINK_GAP_DAYS) -> Entities:
    """Identity resolution from a pre-aggregated (symbol, isin, min, max) frame.

    Split out so the streaming builder can resolve identity from a cheap
    first pass over the cache without ever holding the full long frame, which
    at twenty years is ~10M rows and larger than this machine's free memory.
    """

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

    # Display name: the symbol most recently seen for the entity. Taken from
    # `obs` rather than the raw rows so this works for the streaming builder,
    # which never materialises them.
    newest = obs.sort_values("max").drop_duplicates("isin", keep="last")
    last = dict(zip(newest["isin"], newest["symbol"]))
    latest = dict(zip(newest["isin"], newest["max"]))
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
#: it is a corporate action, or a data error.
#:
#: These were -0.55 and 1.50, chosen to sit below the -50% a 1:2 action produces
#: so that sub-rupee names oscillating 0.10 -> 0.05 would not be quarantined.
#: That reasoning was wrong in a way the one-quarter test could not show. Those
#: penny names are already excluded by the Rs 10 price floor in the eligibility
#: screen, so sparing them bought nothing -- while the same gap let every
#: unadjusted 1:2 split through, which is the exact shape that manufactures
#: momentum. On the twenty-year panel it left 15 such moves on names that pass
#: the screen: MASTEK 834 -> 380 and DABUR 73.65 -> 36.02 among them, each a
#: real 2006 bonus that NSE's feed does not carry.
#:
#: 1.50 had the mirror hole: an unadjusted consolidation shows as a doubling,
#: and +100% slipped under it.
#:
#: The cost of tightening is 0.9% of entities, and a handful of genuine one-day
#: crashes lose the history before them. That is the conservative direction --
#: it removes names from the sample rather than inventing returns for them.
JUMP_DOWN, JUMP_UP = -0.42, 0.90


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


# --------------------------------------------------------- streaming builder
def scan_observations(files: list[Path], series: tuple[str, ...] = ("EQ",)
                      ) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    """First pass: (symbol, isin) date ranges, without holding the rows.

    Twenty years of bhavcopy is ~10M rows; concatenated with string columns it
    does not fit in this machine's free memory alongside the wide panels. Only
    the four identity columns are read, and only min/max per pair is kept, so
    the pass costs a few hundred kilobytes regardless of history length.
    """
    span: dict[tuple[str, str], list] = {}
    dates: list[pd.Timestamp] = []
    for f in files:
        df = pd.read_parquet(f, columns=["date", "isin", "symbol", "series"])
        df = df[df["series"].isin(series)]
        if df.empty:
            continue
        d = pd.Timestamp(df["date"].iloc[0])
        dates.append(d)
        for sym, isin in set(zip(df["symbol"], df["isin"])):
            k = (sym, isin)
            cur = span.get(k)
            if cur is None:
                span[k] = [d, d]
            else:
                if d < cur[0]:
                    cur[0] = d
                if d > cur[1]:
                    cur[1] = d
    obs = pd.DataFrame(
        [(s, i, lo, hi) for (s, i), (lo, hi) in span.items()],
        columns=["symbol", "isin", "min", "max"])
    return obs, sorted(dates)


def build_panel_streaming(files: list[Path], actions: pd.DataFrame | None = None,
                          series: tuple[str, ...] = ("EQ",),
                          out_dir: Path = OUT_DIR, prefix: str = "bhav",
                          dtype: str = "float32", mask_jumps: bool = True,
                          max_hold_gb: float = 0.5
                          ) -> tuple[Entities, object, dict]:
    """Build and write each field separately, holding one at a time.

    `build_panel` materialises four wide frames plus an adjustment matrix at
    once; over twenty years that is ~2GB and this machine has 0.6GB free. Here
    each field is filled into a single array, adjusted, written and released
    before the next begins, so peak memory is one panel rather than five.

    float32 is deliberate. It carries ~7 significant digits, which resolves
    paise on every Indian equity price below Rs 100,000 and costs ~1e-7 relative
    error on a return -- far below the precision any of this is used at, and it
    halves the footprint.
    """
    from kabali.data.corpactions import confirm

    obs, dates = scan_observations(files, series)
    ents = resolve_from_observations(obs)
    obs["entity"] = obs["isin"].map(ents.isin_to_entity)
    entities = sorted(set(obs["entity"].dropna()))
    col = {e: i for i, e in enumerate(entities)}
    row = {d: i for i, d in enumerate(dates)}
    index = pd.DatetimeIndex(dates)

    # Confirming actions needs only close, so it runs inside the close pass and
    # the factors are reused for open. Nothing else is adjusted.
    conf, factors, cutoffs = None, {}, {}
    written: dict = {}
    fields = ("close", "open", "volume", "turnover")

    # One array per field is len(dates) x len(entities) x 4 bytes. Twenty years
    # of NSE resolves to ~3,800 entities, so that is ~80MB each and all four fit
    # comfortably -- at which point re-reading all 5,100 day-files once per
    # field is pure waste, and the four passes are what pushed this past the
    # runner's ten-minute ceiling. Hold them together when they fit, fall back
    # to one at a time when they do not.
    est_gb = len(dates) * len(entities) * 4 * len(fields) / 1e9
    single_pass = est_gb <= max_hold_gb
    log.info("panel %d x %d, %.2f GB for all fields -> %s",
             len(dates), len(entities), est_gb,
             "single pass" if single_pass else "one field at a time")

    arrays: dict = {}
    if single_pass:
        for fld in fields:
            arrays[fld] = np.full((len(dates), len(entities)), np.nan, dtype=dtype)
        for f in files:
            df = pd.read_parquet(f, columns=["date", "isin", "series", *fields])
            df = df[df["series"].isin(series)]
            if df.empty:
                continue
            i = row.get(pd.Timestamp(df["date"].iloc[0]))
            if i is None:
                continue
            e = df["isin"].map(ents.isin_to_entity)
            keep = e.notna()
            if not keep.any():
                continue
            j = e[keep].map(col).to_numpy(dtype="int64")
            sub = df.loc[keep]
            for fld in fields:
                arrays[fld][i, j] = sub[fld].to_numpy(dtype=dtype)

    for fld in fields:
        if single_pass:
            arr = arrays.pop(fld)
        else:
            arr = np.full((len(dates), len(entities)), np.nan, dtype=dtype)
            for f in files:
                df = pd.read_parquet(f, columns=["date", "isin", "series", fld])
                df = df[df["series"].isin(series)]
                if df.empty:
                    continue
                i = row.get(pd.Timestamp(df["date"].iloc[0]))
                if i is None:
                    continue
                e = df["isin"].map(ents.isin_to_entity)
                keep = e.notna()
                if not keep.any():
                    continue
                j = e[keep].map(col).to_numpy(dtype="int64")
                arr[i, j] = df.loc[keep, fld].to_numpy(dtype=dtype)

        frame = pd.DataFrame(arr, index=index, columns=entities)
        del arr

        if fld == "close":
            if actions is not None and not actions.empty:
                conf = _confirm_streaming(actions, frame, ents, confirm)
                factors = _factor_steps(conf.confirmed, index)
            _apply_factors(frame, factors, col)
            # Quarantine is decided here, on adjusted close, and the same
            # cutoffs are reused for open -- recomputing them per field could
            # blank different rows in the two panels and desynchronise them.
            cutoffs = _jump_cutoffs(frame) if mask_jumps else {}
            _apply_cutoffs(frame, cutoffs)
        elif fld == "open":
            _apply_factors(frame, factors, col)
            _apply_cutoffs(frame, cutoffs)

        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        path = p / f"{prefix}_{fld}.parquet"
        frame.to_parquet(path)
        written[fld] = path
        del frame

    written["quarantined"] = len(cutoffs)
    return ents, conf, written


def _confirm_streaming(actions, close, ents, confirm):
    """Confirm actions against a wide close frame, entity-keyed.

    Only the (entity, ex-date) cells the actions actually reference are read.
    Stacking the panel to long form would be the obvious way to reuse `confirm`
    unchanged, and it is what an earlier version did -- but over twenty years
    that is ~12M rows carrying a string key, more than a gigabyte, to answer a
    few thousand lookups. It fits comfortably on a one-quarter test panel and
    not at all on the real one, which is the shape of bug this whole builder
    exists to avoid.

    "Previous close" means the previous *listed* session, not the previous row:
    a name that did not trade yesterday must compare against the last day it
    did, or a trading gap is misread as a price move.
    """
    acts = actions.copy()
    syms = acts["symbol"] if "symbol" in acts.columns else pd.Series(
        [None] * len(acts), index=acts.index)
    acts["entity"] = [ents.entity_for(i, s) for i, s in zip(acts["isin"], syms)]
    acts = acts.dropna(subset=["entity"])
    empty = pd.DataFrame(columns=["date", "entity", "close"])
    if acts.empty:
        return confirm(acts, empty, key="entity")

    row = {d: i for i, d in enumerate(close.index)}
    col = {c: j for j, c in enumerate(close.columns)}
    values = close.to_numpy()

    rows = []
    for ent, ex in set(zip(acts["entity"], acts["ex_date"])):
        i, j = row.get(pd.Timestamp(ex)), col.get(ent)
        if i is None or j is None or i == 0:
            continue
        cur = values[i, j]
        if not np.isfinite(cur):
            continue
        prev_col = values[:i, j]
        finite = np.flatnonzero(np.isfinite(prev_col))
        if not len(finite):
            continue
        k = finite[-1]
        rows.append((close.index[k], ent, float(prev_col[k])))
        rows.append((close.index[i], ent, float(cur)))

    sparse = pd.DataFrame(rows, columns=["date", "entity", "close"]) if rows else empty
    return confirm(acts, sparse, key="entity")


def _factor_steps(confirmed: pd.DataFrame, dates: pd.DatetimeIndex) -> dict:
    """entity -> list of (row index of ex-date, factor).

    Kept as steps rather than a dense (dates x entities) matrix: over twenty
    years that matrix is another full panel of memory, and only a few thousand
    entities have any action at all.
    """
    out: dict[str, list] = {}
    if confirmed is None or confirmed.empty:
        return out
    pos = {d: i for i, d in enumerate(dates)}
    for _, r in confirmed.iterrows():
        i = pos.get(pd.Timestamp(r["ex_date"]))
        if i is None:
            continue
        out.setdefault(r["entity"], []).append((i, float(r["factor"])))
    return out


def _apply_factors(frame: pd.DataFrame, factors: dict, col: dict) -> None:
    """Back-adjust in place: prices before an ex-date scale by its factor."""
    for ent, steps in factors.items():
        j = col.get(ent)
        if j is None:
            continue
        values = frame.iloc[:, j].to_numpy(copy=True)
        for i, f in sorted(steps):
            values[:i] *= f
        frame.iloc[:, j] = values


def _jump_cutoffs(close: pd.DataFrame) -> dict:
    """entity -> row index before which history is unexplained, so blanked."""
    r = close.pct_change(fill_method=None)
    hits = (r < JUMP_DOWN) | (r > JUMP_UP)
    out = {}
    cols = hits.columns[hits.any().to_numpy()]
    for ent in cols:
        rows = np.flatnonzero(hits[ent].fillna(False).to_numpy())
        if len(rows):
            out[ent] = int(rows.max())
    return out


def _apply_cutoffs(frame: pd.DataFrame, cutoffs: dict) -> None:
    """Blank everything before each cutoff, in place."""
    if not cutoffs:
        return
    pos = {c: i for i, c in enumerate(frame.columns)}
    for ent, i in cutoffs.items():
        j = pos.get(ent)
        if j is not None:
            frame.iloc[:i, j] = np.nan


def suspicious_from_panel(close: pd.DataFrame, gap_days: int = 180) -> int:
    """How many entities contain a listing gap this long.

    Computed from the panel's own presence pattern so it needs no second pass
    over the cache. Suspensions produce these legitimately; an over-merged
    ticker also would, which is why the count is reported rather than trusted.
    """
    idx = close.index
    n = 0
    present = close.notna().to_numpy()
    days = idx.to_numpy().astype("datetime64[D]").astype(int)
    for j in range(present.shape[1]):
        rows = np.flatnonzero(present[:, j])
        if len(rows) < 2:
            continue
        if np.diff(days[rows]).max() >= gap_days:
            n += 1
    return n
