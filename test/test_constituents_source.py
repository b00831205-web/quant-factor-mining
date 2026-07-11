"""ConstituentsSource 协议及其实现的单元测试。

覆盖对话中踩过/发现的坑:
1. end_date 为空 = 至今在指数内(开区间语义)
2. 边界日期是闭区间(加入日/退出日当天算在内)
3. ticker 归一化: 成分股表里的 BRK.B 要能匹配行情列名 BRK-B,
   否则这些票被静默排除出宇宙
4. 日期解析只在构造时做一次 + 查询缓存(旧的 get_constitunents_at_date
   每个调仓日都对全表跑 pd.to_datetime)
"""
import pandas as pd
import pytest

from quantfactor.datareader import MembershipTableSource, StaticUniverse


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
    assert "OLD1" in src.get_constituents("2023-05-15")   # 退出日当天仍在
    assert "NEW1" in src.get_constituents("2024-06-01")   # 加入日当天算在内
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
    assert src.get_constituents("2024-01-02") is first  # 同日期返回缓存对象


def test_static_universe_ignores_date():
    src = StaticUniverse(["A", "B"])
    assert src.get_constituents("2020-01-01") == {"A", "B"}
    assert src.get_constituents("2030-01-01") == {"A", "B"}
