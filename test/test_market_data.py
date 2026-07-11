"""Basic MarketData behavior: require() and None handling for optional fields.

Pins down the agreed design: no mutable defaults, require() checks on demand.
"""
import pytest
from quantmine.datareader import MarketData


def test_marketdata_optional_fields_default_to_none():
    """Omitting market_cap should default to None without raising."""
    md = MarketData(close="fake_close", volume="fake_volume")
    assert md.market_cap is None


def test_require_passes_when_field_present():
    """require() on fields that exist must not raise."""
    md = MarketData(close="fake_close", volume="fake_volume")
    md.require("close", "volume")


def test_require_raises_when_field_missing():
    """require() on a missing (None) field must raise a clear ValueError."""
    md = MarketData(close="fake_close", volume="fake_volume")
    with pytest.raises(ValueError, match="market_cap"):
        md.require("market_cap")


def test_require_only_checks_requested_fields():
    """require() checks only the requested names; other None fields must not trigger it."""
    md = MarketData(close="fake_close", volume=None)
    md.require("close")
