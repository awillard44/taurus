import pytest

from taurus.features.adx import calculate_adx


def test_calculate_adx_strong_uptrend():
    highs = [
        float(value)
        for value in range(101, 131)
    ]

    lows = [
        float(value)
        for value in range(99, 129)
    ]

    closes = [
        float(value)
        for value in range(100, 130)
    ]

    result = calculate_adx(
        highs=highs,
        lows=lows,
        closes=closes,
        period=5,
    )

    assert result["plus_di"] > result["minus_di"]
    assert result["adx"] > 0.0


def test_calculate_adx_strong_downtrend():
    highs = [
        float(value)
        for value in range(130, 100, -1)
    ]

    lows = [
        float(value)
        for value in range(128, 98, -1)
    ]

    closes = [
        float(value)
        for value in range(129, 99, -1)
    ]

    result = calculate_adx(
        highs=highs,
        lows=lows,
        closes=closes,
        period=5,
    )

    assert result["minus_di"] > result["plus_di"]
    assert result["adx"] > 0.0


def test_calculate_adx_flat_market():
    highs = [100.0] * 10
    lows = [100.0] * 10
    closes = [100.0] * 10

    result = calculate_adx(
        highs=highs,
        lows=lows,
        closes=closes,
        period=5,
    )

    assert result["adx"] == pytest.approx(0.0)
    assert result["plus_di"] == pytest.approx(0.0)
    assert result["minus_di"] == pytest.approx(0.0)


def test_calculate_adx_requires_positive_period():
    with pytest.raises(ValueError):
        calculate_adx(
            highs=[101.0, 102.0],
            lows=[99.0, 100.0],
            closes=[100.0, 101.0],
            period=0,
        )


def test_calculate_adx_requires_enough_values():
    with pytest.raises(ValueError):
        calculate_adx(
            highs=[
                101.0,
                102.0,
                103.0,
            ],
            lows=[
                99.0,
                100.0,
                101.0,
            ],
            closes=[
                100.0,
                101.0,
                102.0,
            ],
            period=5,
        )


def test_calculate_adx_requires_equal_length_series():
    highs = [
        float(value)
        for value in range(101, 111)
    ]

    lows = [
        float(value)
        for value in range(99, 109)
    ]

    closes = [
        float(value)
        for value in range(100, 109)
    ]

    with pytest.raises(ValueError):
        calculate_adx(
            highs=highs,
            lows=lows,
            closes=closes,
            period=5,
        )