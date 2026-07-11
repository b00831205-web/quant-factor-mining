"""Shared fixtures. Real-data / legacy-file paths are centralized here so a
path change only ever touches this file.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

#repo root on sys.path for "from quantmine.X import ..." package imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ============================================================
# Real data paths -- adjust these to your local project layout
# ============================================================
#anchored to the repo root: relative paths change with pytest's invocation
#directory and would silently skip the real-data tests
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REAL_CLOSE_PATH = os.path.join(_REPO_ROOT, "data/processed/processed_close.parquet")
REAL_VOLUME_PATH = os.path.join(_REPO_ROOT, "data/processed/processed_volume.parquet")
OLD_FACTORS_PATH = os.path.join(_REPO_ROOT, "tmp/factors/factors.parquet")
OLD_DHP_PATH = os.path.join(_REPO_ROOT, "tmp/ic_test/different_holding_period.parquet")
OLD_CS_IC_PATH = os.path.join(_REPO_ROOT, "tmp/ic_test/cs_df.parquet")

REAL_DATA_AVAILABLE = os.path.exists(REAL_CLOSE_PATH) and os.path.exists(REAL_VOLUME_PATH)
OLD_FACTORS_AVAILABLE = os.path.exists(OLD_FACTORS_PATH)
OLD_DHP_AVAILABLE = os.path.exists(OLD_DHP_PATH)
OLD_CS_IC_AVAILABLE = os.path.exists(OLD_CS_IC_PATH)

# mark tests that need real legacy data: absent files skip instead of failing
requires_real_data = pytest.mark.skipif(
    not REAL_DATA_AVAILABLE, reason="real close/volume parquet not found locally, skipping"
)
requires_old_factors = pytest.mark.skipif(
    not OLD_FACTORS_AVAILABLE, reason="legacy factors.parquet not found, skipping golden comparison"
)
requires_old_dhp = pytest.mark.skipif(
    not OLD_DHP_AVAILABLE, reason="legacy different_holding_period.parquet not found, skipping"
)
requires_old_cs_ic = pytest.mark.skipif(
    not OLD_CS_IC_AVAILABLE, reason="legacy CS_IC.parquet not found, skipping"
)


# ============================================================
# Dummy-data fixtures -- fast unit tests, no real files required
# ============================================================
@pytest.fixture
def dummy_close():
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    np.random.seed(42)
    data = {
        t: np.cumprod(1 + np.random.randn(60) * 0.01) * 100
        for t in ["AAPL", "MSFT", "GOOG", "SPY"]
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def dummy_volume(dummy_close):
    np.random.seed(43)
    data = {t: np.random.randint(1_000_000, 5_000_000, size=len(dummy_close))
            for t in dummy_close.columns}
    return pd.DataFrame(data, index=dummy_close.index)


@pytest.fixture
def dummy_tickers(dummy_close):
    return [t for t in dummy_close.columns if t != "SPY"]


# ============================================================
# Real-data fixtures -- for golden-value tests
# ============================================================
@pytest.fixture(scope="session")
def real_close():
    if not REAL_DATA_AVAILABLE:
        pytest.skip("real close data not available")
    return pd.read_parquet(REAL_CLOSE_PATH)


@pytest.fixture(scope="session")
def real_volume():
    if not REAL_DATA_AVAILABLE:
        pytest.skip("real volume data not available")
    return pd.read_parquet(REAL_VOLUME_PATH)
