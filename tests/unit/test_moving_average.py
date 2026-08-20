import pytest

from taurus.features.moving_average import calculate_sma


def test_calculate_sma_20():
    closing_prices = [float(price) for price in range(100, 120)]

    result = calculate_sma(closing_prices, period=20)

    assert result == 109.5


def test_calculate_sma_50():
    closing_prices = [float(price) for price in range(100, 150)]

    result = calculate_sma(closing_prices, period=50)

    assert result == 124.5


def test_calculate_sma_uses_most_recent_prices():
    closing_prices = [50.0, 100.0, 110.0, 120.0]

    result = calculate_sma(closing_prices, period=3)

    assert result == 110.0


def test_calculate_sma_requires_positive_period():
    with pytest.raises(ValueError):
        calculate_sma([100.0, 101.0], period=0)


def test_calculate_sma_requires_enough_prices():
    with pytest.raises(ValueError):
        calculate_sma([100.0, 101.0], period=20)