"""
共享 fixture。真实数据/旧文件路径集中写在这里，
以后路径变了只改这一处。
"""
import os
import numpy as np
import pandas as pd
import pytest
import sys
import os

#两条路径都要: 包内有 "import Datareader" 的裸模块导入,
#也有 "from quantmine.ic_calculator import ..." 的包路径导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "quantmine"))

# ============================================================
# 真实数据路径 —— 按你的实际项目结构调整这几行
# ============================================================
#锚定到repo根目录: 相对路径会随pytest的调用目录变化, 导致真实数据测试被静默跳过
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

# 用这个装饰器标记"需要真实旧数据才能跑"的测试，本地没有文件时自动跳过而不是报错
requires_real_data = pytest.mark.skipif(
    not REAL_DATA_AVAILABLE, reason="本地找不到真实 close/volume parquet，跳过"
)
requires_old_factors = pytest.mark.skipif(
    not OLD_FACTORS_AVAILABLE, reason="本地找不到旧 factors.parquet，跳过黄金值对比"
)
requires_old_dhp = pytest.mark.skipif(
    not OLD_DHP_AVAILABLE, reason="本地找不到旧 different_holding_period.parquet，跳过"
)
requires_old_cs_ic = pytest.mark.skipif(
    not OLD_CS_IC_AVAILABLE, reason="本地找不到旧 CS_IC.parquet，跳过"
)


# ============================================================
# Dummy 数据 fixture —— 单元测试用，不依赖任何真实文件，跑得快
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
# 真实数据 fixture —— 黄金值测试用
# ============================================================
@pytest.fixture(scope="session")
def real_close():
    if not REAL_DATA_AVAILABLE:
        pytest.skip("真实close数据不存在")
    return pd.read_parquet(REAL_CLOSE_PATH)


@pytest.fixture(scope="session")
def real_volume():
    if not REAL_DATA_AVAILABLE:
        pytest.skip("真实volume数据不存在")
    return pd.read_parquet(REAL_VOLUME_PATH)
