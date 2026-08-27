import pytest

from taurus.features.ema import calculate_ema
from taurus.features.macd import calculate_macd


def test_calculate_macd():
    closing_prices = [
        float(value)
        for value in range(100, 140)
    ]

    result = calculate_macd(
        closing_prices=closing_prices,
        fast_period=3,
        slow_period=5,
        signal_period=2,
    )

    macd_values = []

    for end_index in range(
        5,
        len(closing_prices) + 1,
    ):
        window = closing_prices[:end_index]

        fast_ema = calculate_ema(
            window,
            period=3,
        )

        slow_ema = calculate_ema(
            window,
            period=5,
        )

        macd_values.append(
            fast_ema - slow_ema
        )

    expected_macd = macd_values[-1]

    expected_signal = calculate_ema(
        macd_values,
        period=2,
    )

    expected_histogram = (
        expected_macd - expected_signal
    )

    assert result["macd"] == pytest.approx(
        expected_macd
    )

    assert result["signal"] == pytest.approx(
        expected_signal
    )

    assert result["histogram"] == pytest.approx(
        expected_histogram
    )


def test_calculate_macd_requires_fast_period_smaller_than_slow_period():
    with pytest.raises(ValueError):
        calculate_macd(
            closing_prices=[
                float(value)
                for value in range(100, 140)
            ],
            fast_period=5,
            slow_period=5,
            signal_period=2,
        )


def test_calculate_macd_requires_enough_prices():
    with pytest.raises(ValueError):
        calculate_macd(
            closing_prices=[
                100.0,
                101.0,
                102.0,
            ],
            fast_period=3,
            slow_period=5,
            signal_period=2,
        )


def test_calculate_macd_requires_positive_periods():
    closing_prices = [
        float(value)
        for value in range(100, 140)
    ]

    with pytest.raises(ValueError):
        calculate_macd(
            closing_prices=closing_prices,
            fast_period=0,
            slow_period=5,
            signal_period=2,
        )

    with pytest.raises(ValueError):
        calculate_macd(
            closing_prices=closing_prices,
            fast_period=3,
            slow_period=0,
            signal_period=2,
        )

    with pytest.raises(ValueError):
        calculate_macd(
            closing_prices=closing_prices,
            fast_period=3,
            slow_period=5,
            signal_period=0,
        )