import pytest

from taurus.features.moving_average import calculate_sma
from taurus.features.registry import get_indicator_function
from taurus.features.rsi import calculate_rsi


def test_get_indicator_function_returns_registered_function():
    result = get_indicator_function("rsi")

    assert result is calculate_rsi


def test_get_indicator_function_returns_other_registered_function():
    result = get_indicator_function("sma")

    assert result is calculate_sma


def test_get_indicator_function_rejects_unknown_indicator():
    with pytest.raises(ValueError):
        get_indicator_function("does_not_exist")