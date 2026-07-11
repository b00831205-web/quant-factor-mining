"""quantile_backtest 的单元测试(合成数据, 手算期望值)。

这些测试把"新旧架构数值等价性验证"(对旧复合列名版逐值差=0)沉淀为
永久回归测试: 旧实现删除后, 正确性证据仍然在。

覆盖对话中实际踩过的坑:
1. curr_date 必须在调仓循环内取值(曾写在循环外引用未定义index)
2. 结果落库必须在 period 循环内(曾缩进错误导致每因子只留最后一个period)
3. available_tickers 必须按 factor 列序取交集(set顺序会让并列排名的
   分组结果不可复现)
4. ticker_history 的结构: {'date':..., 'Q1': set,..., 'Q5': set}
"""
import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from quantmine.back_testing import quantile_backtest
from quantmine.datareader import StaticUniverse

N_DAYS = 12
TICKERS = [f"T{i}" for i in range(10)]


@pytest.fixture
def dates():
    return pd.date_range("2024-01-01", periods=N_DAYS, freq="B")


@pytest.fixture
def synthetic_factors(dates):
    """因子值恒等于 ticker 序号: T0=0, T1=1, ... 排名完全确定, 无并列。"""
    values = {t: float(i) for i, t in enumerate(TICKERS)}
    df = pd.DataFrame({t: [v] * N_DAYS for t, v in values.items()}, index=dates)
    return {"toy": df}


@pytest.fixture
def synthetic_forward_returns(dates):
    """前向收益恒等于 序号/100: 分组均值可手算。"""
    df = pd.DataFrame({t: [i / 100] * N_DAYS for i, t in enumerate(TICKERS)},
                      index=dates)
    return {1: df, 5: df}


def test_quantile_means_match_hand_computation(synthetic_factors, synthetic_forward_returns):
    result, _ = quantile_backtest(None, synthetic_factors, ["toy"], synthetic_forward_returns)
    df = result[("toy", 1)]
    # 升序分5组, 每组2只: Q1={T0,T1}→均值0.005, Q5={T8,T9}→均值0.085
    assert df["Q1"].iloc[0] == pytest.approx(0.005)
    assert df["Q3"].iloc[0] == pytest.approx(0.045)
    assert df["Q5"].iloc[0] == pytest.approx(0.085)
    assert df["long_short"].iloc[0] == pytest.approx(0.08)


def test_all_factor_period_combinations_present(synthetic_factors, synthetic_forward_returns):
    """曾经的缩进bug: 每个因子只留最后一个period的结果。"""
    result, history = quantile_backtest(None, synthetic_factors, ["toy"], synthetic_forward_returns)
    assert set(result.keys()) == {("toy", 1), ("toy", 5)}
    assert set(history.keys()) == {("toy", 1), ("toy", 5)}


def test_rebalance_dates_follow_period_stride(synthetic_factors, synthetic_forward_returns, dates):
    result, _ = quantile_backtest(None, synthetic_factors, ["toy"], synthetic_forward_returns)
    assert len(result[("toy", 1)]) == N_DAYS
    assert list(result[("toy", 5)].index) == list(dates[::5])


def test_ticker_history_structure(synthetic_factors, synthetic_forward_returns):
    _, history = quantile_backtest(None, synthetic_factors, ["toy"], synthetic_forward_returns)
    snap = history[("toy", 1)][0]
    assert set(snap.keys()) == {"date", "Q1", "Q2", "Q3", "Q4", "Q5"}
    assert snap["Q1"] == {"T0", "T1"}
    assert snap["Q5"] == {"T8", "T9"}


def test_constituents_filtering_excludes_non_members(synthetic_factors, synthetic_forward_returns):
    """宇宙里没有的票不能进分组(point-in-time过滤)。"""
    universe = StaticUniverse(TICKERS[:5])  # 只允许 T0-T4
    _, history = quantile_backtest(universe, synthetic_factors, ["toy"], synthetic_forward_returns)
    snap = history[("toy", 1)][0]
    members = set().union(*(snap[f"Q{i}"] for i in range(1, 6)))
    assert members == set(TICKERS[:5])


def test_membership_dataframe_is_auto_wrapped(synthetic_factors, synthetic_forward_returns):
    """直接传 (ticker, start_date, end_date) 表也能用(自动包装)。"""
    table = pd.DataFrame({
        "ticker": TICKERS[:5],
        "start_date": ["2020-01-01"] * 5,
        "end_date": [None] * 5,
    })
    _, history = quantile_backtest(table, synthetic_factors, ["toy"], synthetic_forward_returns)
    snap = history[("toy", 1)][0]
    members = set().union(*(snap[f"Q{i}"] for i in range(1, 6)))
    assert members == set(TICKERS[:5])


def test_all_nan_cross_section_skipped(synthetic_factors, synthetic_forward_returns, dates):
    factors = {"toy": synthetic_factors["toy"].copy()}
    factors["toy"].loc[dates[0]] = np.nan
    result, _ = quantile_backtest(None, factors, ["toy"], synthetic_forward_returns)
    assert dates[0] not in result[("toy", 1)].index


def test_date_missing_from_forward_returns_skipped(synthetic_factors, synthetic_forward_returns, dates):
    fwd = {1: synthetic_forward_returns[1].drop(index=dates[0])}
    result, _ = quantile_backtest(None, synthetic_factors, ["toy"], fwd)
    assert dates[0] not in result[("toy", 1)].index


def test_deterministic_across_runs(synthetic_factors, synthetic_forward_returns):
    """曾经的set迭代序bug: 结果随运行漂移, 不可复现。"""
    universe = StaticUniverse(TICKERS)
    r1, h1 = quantile_backtest(universe, synthetic_factors, ["toy"], synthetic_forward_returns)
    r2, h2 = quantile_backtest(universe, synthetic_factors, ["toy"], synthetic_forward_returns)
    for key in r1:
        pdt.assert_frame_equal(r1[key], r2[key])
        assert h1[key] == h2[key]
