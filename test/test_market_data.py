"""
测试 MarketData 的基础行为：require()方法、可选字段的None处理。
对应对话中确认过的设计：不用可变默认值、require按需检查。
"""
import pytest
from quantfactor.datareader import MarketData  # 改成你实际的模块名


def test_marketdata_optional_fields_default_to_none():
    """不传market_cap，应该自动是None，不应该报错。"""
    md = MarketData(close="fake_close", volume="fake_volume")
    assert md.market_cap is None


def test_require_passes_when_field_present():
    """require检查一个确实存在的字段，不应该抛异常。"""
    md = MarketData(close="fake_close", volume="fake_volume")
    md.require("close", "volume")  # 不应该抛异常


def test_require_raises_when_field_missing():
    """require检查一个缺失(None)的字段，应该抛出清晰的ValueError。"""
    md = MarketData(close="fake_close", volume="fake_volume")
    with pytest.raises(ValueError, match="market_cap"):
        md.require("market_cap")


def test_require_only_checks_requested_fields():
    """require只检查传入的字段名，不会因为其他None字段而误报。"""
    md = MarketData(close="fake_close", volume=None)
    md.require("close")  # 只检查close，volume是None也不应该报错
