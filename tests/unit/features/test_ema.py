import pytest

from taurus.features.ema import calculate_ema


def test_calculate_ema_with_sma_seed():
    values = [
        10.0,
        12.0,
        14.0,
        16.0,
        18.0,
    ]

    result = calculate_ema(
        values=values,
        period=3,
    )

    # Initial SMA seed:
    # (10 + 12 + 14) / 3 = 12
    #
    # Multiplier:
    # 2 / (3 + 1) = 0.5
    #
    # Next value 16:
    # EMA = 16(0.5) + 12(0.5) = 14
    #
    # Next value 18:
    # EMA = 18(0.5) + 14(0.5) = 16

    assert result == pytest.approx(16.0)


def test_calculate_ema_returns_sma_when_values_equal_period():
    values = [
        10.0,
        12.0,
        14.0,
    ]

    result = calculate_ema(
        values=values,
        period=3,
    )

    assert result == pytest.approx(12.0)


def test_calculate_ema_requires_positive_period():
    with pytest.raises(ValueError):
        calculate_ema(
            values=[10.0, 12.0],
            period=0,
        )


def test_calculate_ema_requires_enough_values():
    with pytest.raises(ValueError):
        calculate_ema(
            values=[10.0, 12.0],
            period=3,
        )