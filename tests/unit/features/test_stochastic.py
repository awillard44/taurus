import pytest
from statistics import mean

from taurus.features.stochastic import calculate_stochastic


def test_calculate_stochastic():
    highs = [
        10.0,
        12.0,
        14.0,
        15.0,
        16.0,
    ]

    lows = [
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
    ]

    closes = [
        9.0,
        11.0,
        13.0,
        14.0,
        15.0,
    ]

    result = calculate_stochastic(
        highs=highs,
        lows=lows,
        closes=closes,
        k_period=3,
        d_period=2,
    )

    k_values = []

    for end_index in range(
        3,
        len(closes) + 1,
    ):
        start_index = end_index - 3

        highest_high = max(
            highs[start_index:end_index]
        )

        lowest_low = min(
            lows[start_index:end_index]
        )

        current_close = closes[end_index - 1]

        k_value = (
            (current_close - lowest_low)
            / (highest_high - lowest_low)
        ) * 100.0

        k_values.append(k_value)

    expected_k = k_values[-1]
    expected_d = mean(k_values[-2:])

    assert result["k"] == pytest.approx(expected_k)
    assert result["d"] == pytest.approx(expected_d)


def test_calculate_stochastic_returns_neutral_when_range_is_zero():
    highs = [
        100.0,
        100.0,
        100.0,
        100.0,
    ]

    lows = [
        100.0,
        100.0,
        100.0,
        100.0,
    ]

    closes = [
        100.0,
        100.0,
        100.0,
        100.0,
    ]

    result = calculate_stochastic(
        highs=highs,
        lows=lows,
        closes=closes,
        k_period=3,
        d_period=2,
    )

    assert result["k"] == 50.0
    assert result["d"] == 50.0


def test_calculate_stochastic_requires_positive_k_period():
    with pytest.raises(ValueError):
        calculate_stochastic(
            highs=[10.0, 11.0],
            lows=[8.0, 9.0],
            closes=[9.0, 10.0],
            k_period=0,
            d_period=2,
        )


def test_calculate_stochastic_requires_positive_d_period():
    with pytest.raises(ValueError):
        calculate_stochastic(
            highs=[10.0, 11.0],
            lows=[8.0, 9.0],
            closes=[9.0, 10.0],
            k_period=2,
            d_period=0,
        )


def test_calculate_stochastic_requires_enough_values():
    with pytest.raises(ValueError):
        calculate_stochastic(
            highs=[10.0, 11.0],
            lows=[8.0, 9.0],
            closes=[9.0, 10.0],
            k_period=3,
            d_period=2,
        )