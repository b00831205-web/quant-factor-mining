"""
黄金值回归测试：新架构算出的8个因子，和旧的 factors.parquet（复合列名）
数值必须完全一致（浮点误差范围内）。

这个测试文件把这次对话里手动跑过的验证过程，固化成自动化测试。
以后任何改动 factor_mining.py 或 factor_register.py，
跑一次这个文件就能立刻知道有没有破坏正确性。
"""
import re
import pandas as pd
import pytest

from conftest import (
    requires_real_data, requires_old_factors,
    REAL_CLOSE_PATH, REAL_VOLUME_PATH, OLD_FACTORS_PATH,
)

# 因子注册名 -> 旧数据列名前缀（对话中逐一确认过的映射关系）
NAME_MAPPING = {
    "daily_return": "DailyReturn",
    "excess_return": "ExcessReturn",
    "TwentyDayVolatility": "TwentyDayVolatility",
    "TwentyDayNegVotality": "TwentyDayNegVolatility",  # 新代码拼写少了个l，暂不改
    "TwentyDayAvgVol": "TwentyDayAvgVol",
    "VolPriceCorr": "VolPriCorr",
    "ShortTermReversal": "ShortTermReversal",
}
MOMENTUM_DAY = 5  # 旧数据momentum用的是5日窗口，必须和param_pool['day']保持一致
TOLERANCE = 1e-8


def _extract_old_factor(old_factors: pd.DataFrame, prefix: str) -> pd.DataFrame:
    matching_cols = [c for c in old_factors.columns if c.rsplit("_", 1)[0] == prefix]
    if not matching_cols:
        return pd.DataFrame()
    df = old_factors[matching_cols].copy()
    df.columns = [c.rsplit("_", 1)[1] for c in matching_cols]
    return df


def _extract_old_momentum(old_factors: pd.DataFrame, day: int) -> pd.DataFrame:
    pattern = rf"^{day}DayMomentum_"
    matching_cols = [c for c in old_factors.columns if re.match(pattern, c)]
    if not matching_cols:
        return pd.DataFrame()
    df = old_factors[matching_cols].copy()
    df.columns = [c.rsplit("_", 1)[1] for c in matching_cols]
    return df


def _max_abs_diff(old_df: pd.DataFrame, new_df: pd.DataFrame) -> float:
    common_tickers = list(set(old_df.columns) & set(new_df.columns))
    common_dates = old_df.index.intersection(new_df.index)
    old_aligned = old_df.loc[common_dates, common_tickers]
    new_aligned = new_df.loc[common_dates, common_tickers]
    return (old_aligned - new_aligned).abs().max().max()


@requires_real_data
@requires_old_factors
class TestFactorsGoldenValues:
    """每个因子一个独立测试用例，方便单独看是哪个因子出问题。"""

    @pytest.fixture(scope="class")
    def computed_factors(self):
        import quantmine.factor_mining  # noqa: F401  触发装饰器注册，必须在这里import
        from quantmine.factor_register import calculate_all_factors, build_param_pool
        from quantmine.datareader import MarketData

        close = pd.read_parquet(REAL_CLOSE_PATH)
        volume = pd.read_parquet(REAL_VOLUME_PATH)
        market_data = MarketData(close=close, volume=volume)
        param_pool = build_param_pool(market_data, day=MOMENTUM_DAY, halflife=10, period=20)

        failures, factors = calculate_all_factors(param_pool)
        assert not failures, f"因子计算不应该有失败，但有: {failures}"
        return factors

    @pytest.fixture(scope="class")
    def old_factors(self):
        return pd.read_parquet(OLD_FACTORS_PATH)

    @pytest.mark.parametrize("factor_name", list(NAME_MAPPING.keys()))
    def test_factor_matches_old_value(self, factor_name, computed_factors, old_factors):
        old_prefix = NAME_MAPPING[factor_name]
        old_df = _extract_old_factor(old_factors, old_prefix)
        assert not old_df.empty, f"旧数据里找不到 {old_prefix} 对应的列"

        new_df = computed_factors[factor_name]
        max_diff = _max_abs_diff(old_df, new_df)
        assert max_diff < TOLERANCE, f"{factor_name} 最大差异 {max_diff} 超出容忍范围"

    def test_momentum_matches_old_value(self, computed_factors, old_factors):
        old_df = _extract_old_momentum(old_factors, MOMENTUM_DAY)
        assert not old_df.empty, "旧数据里找不到momentum对应的列"

        new_df = computed_factors["momentum"]
        max_diff = _max_abs_diff(old_df, new_df)
        assert max_diff < TOLERANCE, f"momentum 最大差异 {max_diff} 超出容忍范围"
