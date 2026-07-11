"""forward_return 的正确性契约测试。

历史note: 这里原本是对旧存档 different_holding_period.parquet 的黄金值对比,
但该存档是2026-07-01用幸存者偏差修复前的旧close(2021起/504只)生成的,
2026-07-04起 processed_close 换成了幸存者修复后的重下数据(2015起/570只),
黄金锚的数据来源已失效(最大差~0.01来自数据本身, 不是函数)。
故改为自包含的数学契约: forward_return[p][t] 必须等于 t 日买入、
持有p个交易日的收益, 用手造价格序列逐值验证, 不依赖任何存档文件。
"""
import numpy as np
import pandas as pd
import pytest

from quantmine.ic_calculator import forward_return
from conftest import requires_real_data, REAL_CLOSE_PATH

TOLERANCE = 1e-12


@pytest.fixture
def hand_made_close():
    """价格=1,2,4,8,...(每天翻倍): 任意持有期收益都可精确手算。"""
    dates = pd.date_range("2024-01-01", periods=8, freq="B")
    return pd.DataFrame({"A": 2.0 ** np.arange(8),
                         "B": 3.0 ** np.arange(8)}, index=dates)


def test_value_is_forward_looking_holding_return(hand_made_close):
    fwd = forward_return(hand_made_close, periods=[1, 3])
    # t日的值 = 未来p日的收益: 翻倍序列 p=1 -> 1.0, p=3 -> 7.0
    assert fwd[1]["A"].iloc[0] == pytest.approx(1.0, abs=TOLERANCE)
    assert fwd[3]["A"].iloc[0] == pytest.approx(7.0, abs=TOLERANCE)
    assert fwd[3]["B"].iloc[0] == pytest.approx(26.0, abs=TOLERANCE)  # 3^3-1


def test_tail_rows_are_nan(hand_made_close):
    """最后p行没有未来数据, 必须是NaN(否则就是前视泄漏)。"""
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
    """真实数据上的同源一致性(同一份close, 函数输出 vs 手算)。"""
    close = pd.read_parquet(REAL_CLOSE_PATH)
    fwd = forward_return(close, periods=[20])
    expected = close.pct_change(20).shift(-20)
    diff = (fwd[20] - expected).abs().max().max()
    assert diff < TOLERANCE
