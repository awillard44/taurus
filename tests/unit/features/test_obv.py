import pytest

from taurus.features.obv import calculate_obv


def test_calculate_obv():
    closes = [
        100.0,
        105.0,
        103.0,
        110.0,
    ]

    volumes = [
        500_000.0,
        1_000_000.0,
        2_000_000.0,
        3_000_000.0,
    ]

    result = calculate_obv(
        closes=closes,
        volumes=volumes,
        period=3,
    )

    # 100 -> 105: +1,000,000
    # 105 -> 103: -2,000,000
    # 103 -> 110: +3,000,000
    #
    # OBV = 2,000,000

    assert result == pytest.approx(2_000_000.0)


def test_calculate_obv_ignores_unchanged_price():
    closes = [
        100.0,
        105.0,
        105.0,
        110.0,
    ]

    volumes = [
        500_000.0,
        1_000_000.0,
        10_000_000.0,
        3_000_000.0,
    ]

    result = calculate_obv(
        closes=closes,
        volumes=volumes,
        period=3,
    )

    # +1M, unchanged, +3M

    assert result == pytest.approx(4_000_000.0)


def test_calculate_obv_uses_most_recent_values():
    closes = [
        50.0,
        100.0,
        105.0,
        103.0,
        110.0,
    ]

    volumes = [
        100_000_000.0,
        500_000.0,
        1_000_000.0,
        2_000_000.0,
        3_000_000.0,
    ]

    result = calculate_obv(
        closes=closes,
        volumes=volumes,
        period=3,
    )

    assert result == pytest.approx(2_000_000.0)


def test_calculate_obv_requires_positive_period():
    with pytest.raises(ValueError):
        calculate_obv(
            closes=[100.0, 105.0],
            volumes=[1_000_000.0, 2_000_000.0],
            period=0,
        )


def test_calculate_obv_requires_enough_values():
    with pytest.raises(ValueError):
        calculate_obv(
            closes=[100.0, 105.0],
            volumes=[1_000_000.0, 2_000_000.0],
            period=3,
        )