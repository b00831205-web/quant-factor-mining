"""换手率 / 交易成本 / 逐日展开的单元测试(手算期望值)。

覆盖对话中确立的成本模型语义:
1. 成本 = 实际换手率 * 2 * cost_per_trade(买卖双边),
   不是固定100%换手假设(那个假设曾把1日持有期扣费后砸到年化-58%)
2. long_short 按 Q1+Q5 两条腿的换手率之和计费
3. expand_to_daily_returns: 窗口含调仓日锚点, 成本记在调仓后首个交易日,
   初始建仓视为全额换手
"""
import numpy as np
import pandas as pd
import pytest

from quantfactor.back_testing import (
    apply_transcation_cost,
    calculate_turnover,
    expand_to_daily_returns,
)


@pytest.fixture
def ticker_history():
    """两次调仓: Q1换掉一半(0.5), Q5全保留(0.0), 其余组全换(1.0)。"""
    d1, d2 = pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-09")
    return [
        {"date": d1, "Q1": {"A", "B"}, "Q2": {"C"}, "Q3": {"D"}, "Q4": {"E"}, "Q5": {"F", "G"}},
        {"date": d2, "Q1": {"B", "X"}, "Q2": {"Y"}, "Q3": {"Z"}, "Q4": {"W"}, "Q5": {"F", "G"}},
    ]


def test_turnover_hand_computed(ticker_history):
    t_q1 = calculate_turnover(ticker_history, "Q1")
    t_q5 = calculate_turnover(ticker_history, "Q5")
    assert np.isnan(t_q1.iloc[0])            # 首期换手率未定义
    assert t_q1.iloc[1] == pytest.approx(0.5)  # {A,B}->{B,X}: 保留1/2
    assert t_q5.iloc[1] == pytest.approx(0.0)  # 全保留


def test_cost_scales_with_actual_turnover(ticker_history):
    result_df = pd.DataFrame(
        {"Q1": [0.01, 0.01], "Q2": [0.01, 0.01], "Q3": [0.01, 0.01],
         "Q4": [0.01, 0.01], "Q5": [0.01, 0.01], "long_short": [0.0, 0.0]},
        index=[h["date"] for h in ticker_history],
    )
    after = apply_transcation_cost(result_df, ticker_history, cost_per_trade=0.001)
    # Q1第二期: 0.01 - 0.5*2*0.001 = 0.009 ; Q5全保留零成本
    assert after["Q1"].iloc[1] == pytest.approx(0.009)
    assert after["Q5"].iloc[1] == pytest.approx(0.01)
    # long_short 双腿计费: 0 - (0.5+0.0)*2*0.001 = -0.001
    assert after["long_short"].iloc[1] == pytest.approx(-0.001)
    # 首期换手率NaN -> 扣费后为NaN(已知语义: 首期被performance_summary的dropna丢弃;
    # 注意与expand_to_daily_returns的"初始建仓全额换手"约定不一致, 属已记录差异)
    assert np.isnan(after["Q1"].iloc[0])


def test_expand_to_daily_returns_window_and_cost():
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    # A: 每天+10%; B: 不动 —— 手算无歧义
    close = pd.DataFrame(
        {"A": 100 * 1.1 ** np.arange(5), "B": [100.0] * 5},
        index=dates,
    )
    hold_a = {f"Q{i}": {"A"} for i in range(1, 6)}
    hold_b = {f"Q{i}": {"B"} for i in range(1, 6)}
    history = [
        {"date": dates[0], **hold_a},
        {"date": dates[2], **hold_b},   # 第二窗口起点
        {"date": dates[4], **hold_b},   # 终点锚(最后一段不展开)
    ]
    daily = expand_to_daily_returns(history, close, cost_per_trade=0.001)

    # 窗口1(持有A): d1,d2两天; 初始建仓turnover=1 -> d1扣0.002
    assert daily.loc[dates[1], "Q1"] == pytest.approx(0.10 - 0.002)
    assert daily.loc[dates[2], "Q1"] == pytest.approx(0.10)
    # 窗口2(持有B): A->B全换手 -> d3扣0.002, B收益0
    assert daily.loc[dates[3], "Q1"] == pytest.approx(0.0 - 0.002)
    assert daily.loc[dates[4], "Q1"] == pytest.approx(0.0)
    # 每个交易日恰好归属一个窗口, 不重不漏
    assert list(daily.index) == list(dates[1:])
    # 同持仓下 long_short = Q5 - Q1 = 0
    assert (daily["long_short"].abs() < 1e-12).all()
