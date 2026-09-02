"""Tests for the bhavcopy panel: the parts that fail silently.

Every string in `test_parse_factor` is a real NSE `subject` value observed in
the feed between 2006 and 2026, kept verbatim -- irregular spacing, truncation
and all. The parser regressed once already during development, when an escape
in the pattern was written as a literal backspace byte and the split branch
stopped matching entirely. Nothing raised; the panel simply came out with
unadjusted splits in it. That is the failure mode these tests exist to catch,
so they assert on values rather than on the code running.
"""
from __future__ import annotations

import pandas as pd
import pytest

from kabali.data.corpactions import confirm, parse_factor
from kabali.data.panel import resolve_entities


@pytest.mark.parametrize("subject, expected", [
    ("Bonus 1:1", 0.5),
    ("Bonus 4:1", 0.2),
    ("Bonus 3:4", 4 / 7),
    ("Bonus - 1:5/Rights - 1:2", 5 / 6),
    ("Agm/Div - 120%/Bonus 1:1", 0.5),
    ("Annual General Meeting/ Dividend - Rs 29.50/- Per Share And Bonus 1:1", 0.5),
    ("Fv Split-Rs.10/-To Rs.2/-", 0.2),
    ("Fv Split Rs.4/- To Re.1/-", 0.25),
    ("Fv Split Rs.10/- To Rs.2/", 0.2),              # truncated upstream
    ("Split-Rs.10tors.2/Div-60%", 0.2),              # no spaces at all
    ("Face Value Split From Rs 10 To Rs 1", 0.1),
    (" Face Value Split (Sub-Division) - From Rs 10/- Per Share To Rs 1/- Per Share", 0.1),
    ("Spl-Rs10 To Rs2/Bonus-1:2", 0.2 * (2 / 3)),    # both, multiplied
])
def test_parse_factor(subject, expected):
    factor, kind = parse_factor(subject)
    assert factor == pytest.approx(expected, rel=1e-6), kind


@pytest.mark.parametrize("subject", [
    "Dividend - Re 0.40 Per Sh",
    "Annual General Meeting",
    "Interest Payment",
    "Buy Back Of Shares",
    "",
])
def test_non_price_actions_yield_no_factor(subject):
    """Dividends and meetings must not produce a price factor."""
    assert parse_factor(subject)[0] is None


def test_confirm_refuses_a_factor_the_tape_contradicts():
    """A parsed factor with no matching price jump is refused, not applied."""
    actions = pd.DataFrame({
        "isin": ["X"], "symbol": ["FOO"],
        "ex_date": [pd.Timestamp("2020-01-02")],
        "subject": ["Bonus 1:1"], "factor": [0.5], "kind": ["bonus 1:1"],
    })
    # The price does not halve on the ex-date, so the action is contradicted.
    prices = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        "isin": ["X", "X"], "close": [100.0, 99.0],
    })
    c = confirm(actions, prices)
    assert len(c.confirmed) == 0
    assert len(c.rejected) == 1


def test_confirm_accepts_a_factor_the_tape_agrees_with():
    actions = pd.DataFrame({
        "isin": ["X"], "symbol": ["FOO"],
        "ex_date": [pd.Timestamp("2020-01-02")],
        "subject": ["Bonus 1:1"], "factor": [0.5], "kind": ["bonus 1:1"],
    })
    prices = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        "isin": ["X", "X"], "close": [100.0, 50.5],
    })
    c = confirm(actions, prices)
    assert len(c.confirmed) == 1
    assert len(c.rejected) == 0


def test_untestable_action_is_out_of_range_not_refused():
    """An ex-date with no bar was never checkable and must not count as a
    refusal -- otherwise the quality metric depends on how wide the action
    feed was queried."""
    actions = pd.DataFrame({
        "isin": ["X"], "symbol": ["FOO"],
        "ex_date": [pd.Timestamp("2021-06-01")],
        "subject": ["Bonus 1:1"], "factor": [0.5], "kind": ["bonus 1:1"],
    })
    prices = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
        "isin": ["X", "X"], "close": [100.0, 99.0],
    })
    c = confirm(actions, prices)
    assert len(c.rejected) == 0
    assert c.out_of_range is not None and len(c.out_of_range) == 1


def _obs(rows):
    return pd.DataFrame(rows, columns=["date", "isin", "symbol"]).assign(
        date=lambda d: pd.to_datetime(d["date"]))


def test_entity_survives_an_isin_change():
    """A split changes the ISIN; the company is still one company.

    FEDERALBNK is INE171A01011 before its 2015 bonus and INE171A01029 after.
    Keyed on ISIN alone the history splits in two, and a 12-month momentum
    window would see a three-month-old listing.
    """
    long = _obs([("2015-07-01", "INE171A01011", "FEDERALBNK"),
                 ("2015-07-09", "INE171A01029", "FEDERALBNK")])
    ents = resolve_entities(long)
    assert len(set(ents.isin_to_entity.values())) == 1


def test_entity_survives_a_symbol_rename():
    """AEGISLOG became AEGISCHEM under one ISIN."""
    long = _obs([("2015-07-01", "INE208C01017", "AEGISLOG"),
                 ("2016-07-01", "INE208C01017", "AEGISCHEM")])
    ents = resolve_entities(long)
    assert len(set(ents.isin_to_entity.values())) == 1


def test_reused_ticker_is_not_merged_across_a_long_gap():
    """A ticker reassigned years later is a different company, not a
    continuation -- this is the over-merge the gap rule exists to refuse."""
    long = _obs([("2006-01-02", "INEOLD0000A1", "ZZZ"),
                 ("2020-01-02", "INENEW0000B2", "ZZZ")])
    ents = resolve_entities(long)
    assert len(set(ents.isin_to_entity.values())) == 2


def _write_day(tmp, d, rows):
    """One synthetic cached day-file in the normalised bhavcopy shape."""
    df = pd.DataFrame(rows, columns=["isin", "symbol", "close", "open",
                                     "volume", "turnover"])
    df.insert(0, "date", pd.Timestamp(d))
    df.insert(4, "series", "EQ")
    df.insert(5, "high", df["close"])
    df.insert(6, "low", df["close"])
    df.insert(7, "prev_close", df["close"])
    p = tmp / f"{pd.Timestamp(d):%Y%m%d}.parquet"
    df.to_parquet(p, index=False)
    return p


def test_streaming_builder_matches_the_in_memory_one(tmp_path):
    """The two builders must agree.

    The streaming path exists only because the in-memory one needs ~2GB over
    twenty years. It is the one that actually runs, so if the two ever diverge
    the cheap builder that every test uses stops describing the real output.
    """
    import numpy as np

    from kabali.data.panel import build_panel, build_panel_streaming

    days = pd.bdate_range("2020-01-01", periods=6)
    files = []
    for k, d in enumerate(days):
        files.append(_write_day(tmp_path, d, [
            ("INE000000001", "AAA", 100.0 + k, 99.0 + k, 1000 + k, 5e5 + k),
            ("INE000000002", "BBB", 50.0 - k, 49.0 - k, 2000 + k, 6e5 + k),
        ]))

    long = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    ref = build_panel(long)
    ents, _, written = build_panel_streaming(files, out_dir=tmp_path,
                                             prefix="s", mask_jumps=False)

    assert set(ents.isin_to_entity.values()) == set(ref.entities.isin_to_entity.values())
    got = pd.read_parquet(written["close"])
    cols = sorted(ref.close.columns)
    assert sorted(got.columns) == cols
    assert np.allclose(ref.close[cols].to_numpy(),
                       got[cols].to_numpy(), rtol=1e-6, equal_nan=True)


def test_streaming_builder_quarantines_an_unexplained_jump(tmp_path):
    """A 1:10-shaped step with no corporate action blanks the prior history."""
    from kabali.data.panel import build_panel_streaming

    days = pd.bdate_range("2020-01-01", periods=5)
    prices = [100.0, 101.0, 9.9, 10.0, 10.1]        # -90% on day 3, unexplained
    files = [_write_day(tmp_path, d, [("INE000000001", "AAA", p, p, 10, 1e5)])
             for d, p in zip(days, prices)]

    _, _, written = build_panel_streaming(files, out_dir=tmp_path, prefix="q")
    close = pd.read_parquet(written["close"])
    col = close.columns[0]
    assert close[col].iloc[:2].isna().all(), "pre-jump history must be blanked"
    assert close[col].iloc[2:].notna().all(), "post-jump history must survive"
    assert written["quarantined"] == 1
