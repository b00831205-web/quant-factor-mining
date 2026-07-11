"""Unit tests for the ConstituentsSource protocol and its implementations.

Pitfalls covered (all encountered during development):
1. empty end_date = still a member today (open-ended semantics)
2. boundary dates are inclusive (join/exit day counts as a member day)
3. ticker normalization: BRK.B in the membership table must match the price
   column BRK-B, otherwise those tickers are silently excluded from the universe
4. dates are parsed once at construction + queries are cached (the legacy
   get_constitunents_at_date re-ran pd.to_datetime over the whole table on
   every rebalance day)
"""
import pandas as pd
import pytest

from quantmine.datareader import MembershipTableSource, StaticUniverse


@pytest.fixture
def membership_table():
    return pd.DataFrame({
        "ticker": ["AAPL", "BRK.B", "OLD1", "NEW1"],
        "start_date": ["2010-01-01", "2010-01-01", "2010-01-01", "2024-06-01"],
        "end_date": [None, None, "2023-05-15", None],
    })


def test_open_ended_membership(membership_table):
    src = MembershipTableSource(membership_table)
    assert "AAPL" in src.get_constituents("2024-01-02")


def test_exited_ticker_excluded_after_end_date(membership_table):
    src = MembershipTableSource(membership_table)
    assert "OLD1" not in src.get_constituents("2024-01-02")
    assert "OLD1" in src.get_constituents("2022-01-03")


def test_boundary_dates_inclusive(membership_table):
    src = MembershipTableSource(membership_table)
    assert "OLD1" in src.get_constituents("2023-05-15")   # still a member on the exit day
    assert "NEW1" in src.get_constituents("2024-06-01")   # counts from the join day
    assert "NEW1" not in src.get_constituents("2024-05-31")


def test_ticker_normalization_matches_yfinance_style(membership_table):
    src = MembershipTableSource(membership_table)
    universe = src.get_constituents("2024-01-02")
    assert "BRK-B" in universe
    assert "BRK.B" not in universe


def test_normalization_can_be_disabled(membership_table):
    src = MembershipTableSource(membership_table, normalize=False)
    assert "BRK.B" in src.get_constituents("2024-01-02")


def test_query_is_cached(membership_table):
    src = MembershipTableSource(membership_table)
    first = src.get_constituents("2024-01-02")
    assert src.get_constituents("2024-01-02") is first  # same date returns the cached object


def test_static_universe_ignores_date():
    src = StaticUniverse(["A", "B"])
    assert src.get_constituents("2020-01-01") == {"A", "B"}
    assert src.get_constituents("2030-01-01") == {"A", "B"}
