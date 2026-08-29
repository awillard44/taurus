import pytest

from datetime import datetime, timedelta

from taurus.data.schemas import BarInterval, PriceBar
from taurus.environment.feature_state_builder import (
    build_feature_state_sequence,
)
from taurus.features.presets import DEFAULT_FEATURE_SET


def build_price_bar(
        symbol: str,
        timestamp: datetime,
        close: float,
) -> PriceBar:
    return PriceBar(
        symbol=symbol,
        timestamp=timestamp,
        open=close - 1.0,
        high=close + 1.0,
        low=close - 2.0,
        close=close,
        volume=1_000_000.0,
        source="test",
        interval=BarInterval.ONE_DAY,
    )


def build_test_bars(
        count: int = 60,
) -> tuple[list[PriceBar], list[PriceBar]]:
    start = datetime(2026, 1, 1)

    asset_bars = []
    benchmark_bars = []

    for index in range(count):
        timestamp = start + timedelta(days=index)

        asset_bars.append(
            build_price_bar(
                symbol="TEST",
                timestamp=timestamp,
                close=100.0 + index,
            )
        )

        benchmark_bars.append(
            build_price_bar(
                symbol="SPY",
                timestamp=timestamp,
                close=200.0 + (index * 0.5),
            )
        )

    return asset_bars, benchmark_bars


def test_build_feature_state_sequence_returns_expected_number_of_states():
    asset_bars, benchmark_bars = build_test_bars(
        count=60,
    )

    feature_states = build_feature_state_sequence(
        bars=asset_bars,
        benchmark_bars=benchmark_bars,
        feature_set=DEFAULT_FEATURE_SET,
        minimum_history=50,
    )

    assert len(feature_states) == 11


def test_build_feature_state_sequence_preserves_timestamps():
    asset_bars, benchmark_bars = build_test_bars(
        count=60,
    )

    feature_states = build_feature_state_sequence(
        bars=asset_bars,
        benchmark_bars=benchmark_bars,
        feature_set=DEFAULT_FEATURE_SET,
        minimum_history=50,
    )

    assert (
        feature_states[0].market.timestamp
        == asset_bars[49].timestamp
    )

    assert (
        feature_states[-1].market.timestamp
        == asset_bars[-1].timestamp
    )


def test_build_feature_state_sequence_contains_default_features():
    asset_bars, benchmark_bars = build_test_bars(
        count=60,
    )

    feature_states = build_feature_state_sequence(
        bars=asset_bars,
        benchmark_bars=benchmark_bars,
        feature_set=DEFAULT_FEATURE_SET,
        minimum_history=50,
    )

    features = feature_states[-1].features

    assert set(features) == {
        "return_1",
        "return_5",
        "sma_20",
        "rsi_14",
        "atr_14",
        "volume_ratio_20",
        "relative_return_20",
        "adx_14_adx",
        "adx_14_plus_di",
        "adx_14_minus_di",
    }


def test_build_feature_state_sequence_does_not_use_future_bars():
    asset_bars, benchmark_bars = build_test_bars(
        count=60,
    )

    original_states = build_feature_state_sequence(
        bars=asset_bars,
        benchmark_bars=benchmark_bars,
        feature_set=DEFAULT_FEATURE_SET,
        minimum_history=50,
    )

    modified_asset_bars = list(asset_bars)

    modified_asset_bars[-1] = build_price_bar(
        symbol="TEST",
        timestamp=asset_bars[-1].timestamp,
        close=10_000.0,
    )

    modified_states = build_feature_state_sequence(
        bars=modified_asset_bars,
        benchmark_bars=benchmark_bars,
        feature_set=DEFAULT_FEATURE_SET,
        minimum_history=50,
    )

    assert (
        original_states[0].features
        == modified_states[0].features
    )

    assert (
        original_states[-1].features
        != modified_states[-1].features
    )


def test_build_feature_state_sequence_requires_positive_minimum_history():
    asset_bars, benchmark_bars = build_test_bars(
        count=60,
    )

    with pytest.raises(ValueError):
        build_feature_state_sequence(
            bars=asset_bars,
            benchmark_bars=benchmark_bars,
            feature_set=DEFAULT_FEATURE_SET,
            minimum_history=0,
        )


def test_build_feature_state_sequence_requires_enough_asset_bars():
    asset_bars, benchmark_bars = build_test_bars(
        count=20,
    )

    with pytest.raises(ValueError):
        build_feature_state_sequence(
            bars=asset_bars,
            benchmark_bars=benchmark_bars,
            feature_set=DEFAULT_FEATURE_SET,
            minimum_history=50,
        )


def test_build_feature_state_sequence_requires_equal_lengths():
    asset_bars, benchmark_bars = build_test_bars(
        count=60,
    )

    benchmark_bars = benchmark_bars[:-1]

    with pytest.raises(ValueError):
        build_feature_state_sequence(
            bars=asset_bars,
            benchmark_bars=benchmark_bars,
            feature_set=DEFAULT_FEATURE_SET,
            minimum_history=50,
        )


def test_build_feature_state_sequence_requires_matching_timestamps():
    asset_bars, benchmark_bars = build_test_bars(
        count=60,
    )

    benchmark_bars = list(benchmark_bars)

    benchmark_bars[10] = build_price_bar(
        symbol="SPY",
        timestamp=benchmark_bars[10].timestamp
        + timedelta(days=1),
        close=benchmark_bars[10].close,
    )

    with pytest.raises(ValueError):
        build_feature_state_sequence(
            bars=asset_bars,
            benchmark_bars=benchmark_bars,
            feature_set=DEFAULT_FEATURE_SET,
            minimum_history=50,
        )


def test_build_feature_state_sequence_requires_matching_intervals():
    asset_bars, benchmark_bars = build_test_bars(
        count=60,
    )

    original_bar = benchmark_bars[10]

    benchmark_bars = list(benchmark_bars)

    benchmark_bars[10] = PriceBar(
        symbol=original_bar.symbol,
        timestamp=original_bar.timestamp,
        open=original_bar.open,
        high=original_bar.high,
        low=original_bar.low,
        close=original_bar.close,
        volume=original_bar.volume,
        source=original_bar.source,
        interval=BarInterval.ONE_HOUR,
    )

    with pytest.raises(ValueError):
        build_feature_state_sequence(
            bars=asset_bars,
            benchmark_bars=benchmark_bars,
            feature_set=DEFAULT_FEATURE_SET,
            minimum_history=50,
        )