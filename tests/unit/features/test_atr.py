import pytest

from taurus.features.atr import calculate_atr


def test_calculate_atr():
    highs = [
        10.0,
        12.0,
        13.0,
        15.0,
    ]

    lows = [
        8.0,
        9.0,
        10.0,
        11.0,
    ]

    closes = [
        9.0,
        11.0,
        12.0,
        14.0,
    ]

    result = calculate_atr(
        highs=highs,
        lows=lows,
        closes=closes,
        period=3,
    )

    expected_true_ranges = [
        3.0,
        3.0,
        4.0,
    ]

    expected_atr = sum(expected_true_ranges) / 3

    assert result == pytest.approx(expected_atr)


def test_calculate_atr_requires_positive_period():
    with pytest.raises(ValueError):
        calculate_atr(
            highs=[10.0, 11.0],
            lows=[8.0, 9.0],
            closes=[9.0, 10.0],
            period=0,
        )


def test_calculate_atr_requires_enough_values():
    with pytest.raises(ValueError):
        calculate_atr(
            highs=[10.0, 11.0],
            lows=[8.0, 9.0],
            closes=[9.0, 10.0],
            period=3,
        )