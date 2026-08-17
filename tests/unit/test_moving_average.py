import pytest

from taurus.features.moving_average import (
    calculate_sma_20,
    calculate_sma_50,
)


def test_calculate_sma_20():
    closing_prices = [
        100.0, 101.0, 102.0, 103.0, 104.0,
        105.0, 106.0, 107.0, 108.0, 109.0,
        110.0, 111.0, 112.0, 113.0, 114.0,
        115.0, 116.0, 117.0, 118.0, 119.0,
    ]

    result = calculate_sma_20(closing_prices)

    assert result == 109.5


def test_calculate_sma_20_requires_20_prices():
    closing_prices = [100.0, 101.0, 102.0]

    with pytest.raises(ValueError):
        calculate_sma_20(closing_prices)

def test_calculate_sma_50():
    closing_prices = [float(price) for price in range(100, 150)]

    result = calculate_sma_50(closing_prices)

    assert result == 124.5


def test_calculate_sma_50_requires_50_prices():
    closing_prices = [100.0, 101.0, 102.0]

    with pytest.raises(ValueError):
        calculate_sma_50(closing_prices)