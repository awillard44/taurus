import pytest

from taurus.features.rsi import calculate_rsi


def test_calculate_rsi_with_mixed_gains_and_losses():
    closing_prices = [
        100.0,
        103.0,
        101.0,
        105.0,
    ]

    result = calculate_rsi(closing_prices, period=3)

    expected_rsi = 77.77777777777777

    assert result == pytest.approx(expected_rsi)


def test_calculate_rsi_all_gains_returns_100():
    closing_prices = [
        100.0,
        101.0,
        102.0,
        103.0,
        104.0,
    ]

    result = calculate_rsi(closing_prices, period=4)

    assert result == 100.0


def test_calculate_rsi_all_losses_returns_0():
    closing_prices = [
        104.0,
        103.0,
        102.0,
        101.0,
        100.0,
    ]

    result = calculate_rsi(closing_prices, period=4)

    assert result == 0.0


def test_calculate_rsi_uses_most_recent_prices():
    closing_prices = [
        50.0,
        100.0,
        103.0,
        101.0,
        105.0,
    ]

    result = calculate_rsi(closing_prices, period=3)

    expected_rsi = 77.77777777777777

    assert result == pytest.approx(expected_rsi)


def test_calculate_rsi_requires_positive_period():
    with pytest.raises(ValueError):
        calculate_rsi([100.0, 101.0], period=0)


def test_calculate_rsi_requires_enough_prices():
    with pytest.raises(ValueError):
        calculate_rsi(
            [100.0, 101.0, 102.0],
            period=5,
        )