"""
黄金值回归测试：完整链路(因子计算 -> forward_returns -> CS_IC)
和旧的 CS_IC.parquet 数值对比。

对话中最终确认：用 Pearson method + momentum day=5，
新旧数值逐项完全一致(差异在1e-16量级的浮点精度极限)。
"""
import pandas as pd
import pytest

from conftest import (
    requires_real_data, requires_old_cs_ic,
    REAL_CLOSE_PATH, REAL_VOLUME_PATH, OLD_CS_IC_PATH,
)

TOLERANCE = 1e-6  # IC是相关系数，容忍度比因子原始值稍宽松一点
MOMENTUM_DAY = 5

NAME_MAPPING = {
    "daily_return": "DailyReturn",
    "excess_return": "ExcessReturn",
    "TwentyDayVolatility": "TwentyDayVolatility",
    "TwentyDayNegVotality": "TwentyDayNegVolatility",
    "TwentyDayAvgVol": "TwentyDayAvgVol",
    "VolPriceCorr": "VolPriCorr",
    "ShortTermReversal": "ShortTermReversal",
}


def _find_old_column(old_ic_df: pd.DataFrame, factor_name: str, period: int) -> str | None:
    if factor_name == "momentum":
        prefix = f"{MOMENTUM_DAY}DayMomentum"
    else:
        prefix = NAME_MAPPING.get(factor_name)
        if prefix is None:
            return None
    candidate = f"{prefix}_{period}DaysHoldingPeriod"
    return candidate if candidate in old_ic_df.columns else None


@requires_real_data
@requires_old_cs_ic
class TestCSInformationCorrelationGolden:

    @pytest.fixture(scope="class")
    def new_ic_df(self, tmp_path_factory):
        import quantfactor.factor_mining  # noqa: F401
        from quantfactor.factor_register import calculate_all_factors, build_param_pool
        from quantfactor.datareader import MarketData
        from quantfactor.ic_calculator import forward_return, CS_Information_Correlation

        close = pd.read_parquet(REAL_CLOSE_PATH)
        volume = pd.read_parquet(REAL_VOLUME_PATH)
        market_data = MarketData(close=close, volume=volume)
        param_pool = build_param_pool(market_data, day=MOMENTUM_DAY, halflife=10, period=20)

        failures, factors = calculate_all_factors(param_pool)
        assert not failures

        forward_returns = forward_return(close, periods=[1, 5, 20])

        out_path = str(tmp_path_factory.mktemp("ic") / "ic_test.parquet")
        return CS_Information_Correlation(factors, forward_returns, output_path=out_path)

    @pytest.fixture(scope="class")
    def old_ic_df(self):
        return pd.read_parquet(OLD_CS_IC_PATH)

    @pytest.mark.parametrize("factor_name", list(NAME_MAPPING.keys()) + ["momentum"])
    @pytest.mark.parametrize("period", [1, 5, 20])
    def test_ic_matches_old_value(self, factor_name, period, new_ic_df, old_ic_df):
        old_col = _find_old_column(old_ic_df, factor_name, period)
        assert old_col is not None, f"旧数据里找不到 {factor_name}, period={period} 对应的列"

        old_series = old_ic_df[old_col]
        new_series = new_ic_df[(factor_name, period)]
        common_dates = old_series.index.intersection(new_series.index)

        diff = (old_series.loc[common_dates] - new_series.loc[common_dates]).abs()
        max_diff = diff.max()

        assert max_diff < TOLERANCE, f"{factor_name}, period={period} 最大差异 {max_diff} 超出容忍范围"
