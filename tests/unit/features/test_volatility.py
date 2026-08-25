import pytest
from statistics import stdev

from taurus.features.volatility import calculate_volatility


def test_calculate_volatility():
    returns = [
        0.01, -0.02, 0.015, 0.005, -0.01,
        0.02, -0.005, 0.01, -0.015, 0.025,
    ]

    result = calculate_volatility(returns, period=10)

    assert result == stdev(returns)


def test_calculate_volatility_uses_most_recent_returns():
    returns = [
        0.50,
        0.01,
        -0.01,
        0.02,
        -0.02,
    ]

    result = calculate_volatility(returns, period=4)

    expected = stdev([
        0.01,
        -0.01,
        0.02,
        -0.02,
    ])

    assert result == expected


def test_calculate_volatility_requires_at_least_two_periods():
    with pytest.raises(ValueError):
        calculate_volatility([0.01, 0.02], period=1)


def test_calculate_volatility_requires_enough_returns():
    with pytest.raises(ValueError):
        calculate_volatility([0.01, 0.02], period=5)