import pytest

from taurus.features.volume import calculate_volume_ratio


def test_calculate_volume_ratio():
    volumes = [
        40.0,
        40.0,
        40.0,
        80.0,
    ]

    result = calculate_volume_ratio(volumes, period=3)

    assert result == 2.0


def test_calculate_volume_ratio_below_average():
    volumes = [
        40.0,
        40.0,
        40.0,
        20.0,
    ]

    result = calculate_volume_ratio(volumes, period=3)

    assert result == 0.5


def test_calculate_volume_ratio_uses_most_recent_periods():
    volumes = [
        1_000.0,
        40.0,
        40.0,
        40.0,
        80.0,
    ]

    result = calculate_volume_ratio(volumes, period=3)

    assert result == 2.0


def test_calculate_volume_ratio_requires_positive_period():
    with pytest.raises(ValueError):
        calculate_volume_ratio(
            [40.0, 80.0],
            period=0,
        )


def test_calculate_volume_ratio_requires_enough_volumes():
    with pytest.raises(ValueError):
        calculate_volume_ratio(
            [40.0, 80.0],
            period=3,
        )


def test_calculate_volume_ratio_rejects_zero_average():
    volumes = [
        0.0,
        0.0,
        0.0,
        80.0,
    ]

    with pytest.raises(ValueError):
        calculate_volume_ratio(volumes, period=3)