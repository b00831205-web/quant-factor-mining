"""Correctness contract tests for forward_return.

History note: this file used to be a golden-value comparison against an
archived different_holding_period.parquet. That archive was generated on
2026-07-01 from the pre-survivorship-fix close data (2021+, 504 tickers); on
2026-07-04 processed_close was rebuilt from re-downloaded survivorship-fixed
data (2015+, 570 tickers), so the golden anchor's data provenance became
invalid (the ~0.01 max diff came from the data itself, not the function).
It is now a self-contained mathematical contract: forward_return[p][t] must
equal the return of buying at t and holding for p trading days, verified
value-by-value on hand-made price series with no archive dependency.
"""
import numpy as np
import pandas as pd
import pytest

from quantmine.ic_calculator import forward_return
from conftest import requires_real_data, REAL_CLOSE_PATH

TOLERANCE = 1e-12


@pytest.fixture
def hand_made_close():
    """Prices 1, 2, 4, 8, ... (doubling daily): any holding-period return is exactly hand-computable."""
    dates = pd.date_range("2024-01-01", periods=8, freq="B")
    return pd.DataFrame({"A": 2.0 ** np.arange(8),
                         "B": 3.0 ** np.arange(8)}, index=dates)


def test_value_is_forward_looking_holding_return(hand_made_close):
    fwd = forward_return(hand_made_close, periods=[1, 3])
    # value at t = return over the next p days: doubling series gives p=1 -> 1.0, p=3 -> 7.0
    assert fwd[1]["A"].iloc[0] == pytest.approx(1.0, abs=TOLERANCE)
    assert fwd[3]["A"].iloc[0] == pytest.approx(7.0, abs=TOLERANCE)
    assert fwd[3]["B"].iloc[0] == pytest.approx(26.0, abs=TOLERANCE)  # 3^3 - 1


def test_tail_rows_are_nan(hand_made_close):
    """The last p rows have no future data and must be NaN (anything else is look-ahead leakage)."""
    fwd = forward_return(hand_made_close, periods=[3])
    assert fwd[3]["A"].iloc[-3:].isna().all()
    assert fwd[3]["A"].iloc[:-3].notna().all()


def test_matches_manual_pct_change_identity(hand_made_close):
    fwd = forward_return(hand_made_close, periods=[1, 5, 20])
    for p, df in fwd.items():
        expected = hand_made_close.pct_change(p).shift(-p)
        pd.testing.assert_frame_equal(df, expected)


def test_int_period_and_default_periods(hand_made_close):
    assert set(forward_return(hand_made_close, periods=5).keys()) == {5}
    assert set(forward_return(hand_made_close).keys()) == {1, 5, 20}


def test_tickers_subset_selection(hand_made_close):
    fwd = forward_return(hand_made_close, tickers=["A"], periods=[1])
    assert list(fwd[1].columns) == ["A"]


@requires_real_data
def test_identity_holds_on_real_close():
    """Same-source consistency on real data (one close file: function output vs manual)."""
    close = pd.read_parquet(REAL_CLOSE_PATH)
    fwd = forward_return(close, periods=[20])
    expected = close.pct_change(20).shift(-20)
    diff = (fwd[20] - expected).abs().max().max()
    assert diff < TOLERANCE
