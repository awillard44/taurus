import pytest
from taurus.features.returns import calculate_return

def test_calculate_return_one_period():
    closing_prices = [100.0, 103.0]

    result = calculate_return(closing_prices, periods=1)

    assert result == 0.03

def test_calculate_return_five_periods():
    closing_prices = [
        100.0,
        101.0,
        102.0,
        104.0,
        107.0,
        110.0,
    ]

    result = calculate_return(closing_prices, periods=5)

    assert result == 0.10


def test_calculate_negative_return():
    closing_prices = [100.0, 98.0]

    result = calculate_return(closing_prices, periods=1)

    assert result == -0.02


def test_calculate_return_requires_positive_periods():
    with pytest.raises(ValueError):
        calculate_return([100.0, 101.0], periods=0)


def test_calculate_return_requires_enough_prices():
    with pytest.raises(ValueError):
        calculate_return([100.0, 101.0], periods=5)