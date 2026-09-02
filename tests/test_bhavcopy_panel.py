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
