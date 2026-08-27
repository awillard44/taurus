import pytest

from taurus.features.vwap import calculate_vwap


def test_calculate_vwap():
    highs = [
        101.0,
        111.0,
    ]

    lows = [
        99.0,
        109.0,
    ]

    closes = [
        100.0,
        110.0,
    ]

    volumes = [
        1_000_000.0,
        9_000_000.0,
    ]

    result = calculate_vwap(
        highs=highs,
        lows=lows,
        closes=closes,
        volumes=volumes,
        period=2,
    )

    typical_price_1 = (101.0 + 99.0 + 100.0) / 3.0
    typical_price_2 = (111.0 + 109.0 + 110.0) / 3.0

    expected = (
        typical_price_1 * 1_000_000.0
        + typical_price_2 * 9_000_000.0
    ) / 10_000_000.0

    assert result == pytest.approx(expected)


def test_calculate_vwap_uses_most_recent_values():
    highs = [
        50.0,
        101.0,
        111.0,
    ]

    lows = [
        50.0,
        99.0,
        109.0,
    ]

    closes = [
        50.0,
        100.0,
        110.0,
    ]

    volumes = [
        100_000_000.0,
        1_000_000.0,
        9_000_000.0,
    ]

    result = calculate_vwap(
        highs=highs,
        lows=lows,
        closes=closes,
        volumes=volumes,
        period=2,
    )

    expected = (
        100.0 * 1_000_000.0
        + 110.0 * 9_000_000.0
    ) / 10_000_000.0

    assert result == pytest.approx(expected)


def test_calculate_vwap_requires_positive_period():
    with pytest.raises(ValueError):
        calculate_vwap(
            highs=[101.0],
            lows=[99.0],
            closes=[100.0],
            volumes=[1_000_000.0],
            period=0,
        )


def test_calculate_vwap_requires_enough_values():
    with pytest.raises(ValueError):
        calculate_vwap(
            highs=[101.0],
            lows=[99.0],
            closes=[100.0],
            volumes=[1_000_000.0],
            period=2,
        )


def test_calculate_vwap_requires_positive_total_volume():
    with pytest.raises(ValueError):
        calculate_vwap(
            highs=[101.0, 111.0],
            lows=[99.0, 109.0],
            closes=[100.0, 110.0],
            volumes=[0.0, 0.0],
            period=2,
        )