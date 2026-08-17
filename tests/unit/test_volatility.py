import pytest
from statistics import stdev

from taurus.features.volatility import calculate_volatility_20d


def test_calculate_volatility_20d():
    daily_returns = [
        0.01, -0.02, 0.015, 0.005, -0.01,
        0.02, -0.005, 0.01, -0.015, 0.025,
        0.005, -0.01, 0.015, -0.005, 0.01,
        0.02, -0.02, 0.005, 0.01, -0.005,
    ]

    result = calculate_volatility_20d(daily_returns)

    assert result == stdev(daily_returns)

def test_calculate_volatility_20d_requires_20_returns():
    daily_returns = [0.01, -0.02, 0.015]

    with pytest.raises(ValueError):
        calculate_volatility_20d(daily_returns)