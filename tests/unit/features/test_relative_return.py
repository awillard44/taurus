import pytest

from taurus.features.relative_return import calculate_relative_return


def test_calculate_relative_return():
    asset_prices = [
        100.0,
        105.0,
    ]

    benchmark_prices = [
        100.0,
        101.0,
    ]

    result = calculate_relative_return(
        asset_prices,
        benchmark_prices,
        periods=1,
    )

    assert result == pytest.approx(0.04)


def test_calculate_negative_relative_return():
    asset_prices = [
        100.0,
        101.0,
    ]

    benchmark_prices = [
        100.0,
        105.0,
    ]

    result = calculate_relative_return(
        asset_prices,
        benchmark_prices,
        periods=1,
    )

    assert result == pytest.approx(-0.04)


def test_calculate_relative_return_uses_most_recent_periods():
    asset_prices = [
        50.0,
        100.0,
        110.0,
    ]

    benchmark_prices = [
        500.0,
        100.0,
        105.0,
    ]

    result = calculate_relative_return(
        asset_prices,
        benchmark_prices,
        periods=1,
    )

    assert result == pytest.approx(0.05)


def test_calculate_relative_return_requires_positive_periods():
    with pytest.raises(ValueError):
        calculate_relative_return(
            [100.0, 101.0],
            [100.0, 101.0],
            periods=0,
        )


def test_calculate_relative_return_requires_enough_asset_prices():
    with pytest.raises(ValueError):
        calculate_relative_return(
            [100.0, 101.0],
            [100.0, 101.0, 102.0],
            periods=2,
        )


def test_calculate_relative_return_requires_enough_benchmark_prices():
    with pytest.raises(ValueError):
        calculate_relative_return(
            [100.0, 101.0, 102.0],
            [100.0, 101.0],
            periods=2,
        )