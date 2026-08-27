import pytest
from statistics import mean, pstdev

from taurus.features.bollinger_bands import calculate_bollinger_bands


def test_calculate_bollinger_bands():
    closing_prices = [
        100.0,
        102.0,
        104.0,
        106.0,
        108.0,
    ]

    result = calculate_bollinger_bands(
        closing_prices=closing_prices,
        period=5,
        stddev_multiplier=2.0,
    )

    middle = mean(closing_prices)
    deviation = pstdev(closing_prices)

    assert result["middle"] == pytest.approx(middle)
    assert result["lower"] == pytest.approx(
        middle - (2.0 * deviation)
    )
    assert result["upper"] == pytest.approx(
        middle + (2.0 * deviation)
    )


def test_calculate_bollinger_bands_uses_most_recent_prices():
    closing_prices = [
        50.0,
        100.0,
        102.0,
        104.0,
        106.0,
        108.0,
    ]

    result = calculate_bollinger_bands(
        closing_prices=closing_prices,
        period=5,
    )

    expected_window = [
        100.0,
        102.0,
        104.0,
        106.0,
        108.0,
    ]

    assert result["middle"] == pytest.approx(
        mean(expected_window)
    )


def test_calculate_bollinger_bands_requires_positive_period():
    with pytest.raises(ValueError):
        calculate_bollinger_bands(
            closing_prices=[100.0, 101.0],
            period=0,
        )


def test_calculate_bollinger_bands_requires_enough_prices():
    with pytest.raises(ValueError):
        calculate_bollinger_bands(
            closing_prices=[100.0, 101.0],
            period=5,
        )


def test_calculate_bollinger_bands_requires_positive_multiplier():
    with pytest.raises(ValueError):
        calculate_bollinger_bands(
            closing_prices=[
                100.0,
                101.0,
                102.0,
            ],
            period=3,
            stddev_multiplier=0.0,
        )