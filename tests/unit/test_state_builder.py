import pytest

from datetime import datetime, timedelta

from taurus.data.schemas import BarInterval, PriceBar
from taurus.features.state_builder import build_market_state


def test_build_market_state():
    start = datetime(2026, 6, 1)

    bars = [
        PriceBar(
            symbol="NVDA",
            timestamp=start + timedelta(days=i),
            open=100.0 + i,
            high=101.0 + i,
            low=99.0 + i,
            close=100.0 + i,
            volume=1_000_000 + (i * 10_000),
            source="test",
            interval=BarInterval.ONE_DAY,
        )
        for i in range(50)
    ]

    benchmark_bars = [
        PriceBar(
            symbol="SPY",
            timestamp=start + timedelta(days=i),
            open=100.0 + (i * 0.5),
            high=101.0 + (i * 0.5),
            low=99.0 + (i * 0.5),
            close=100.0 + (i * 0.5),
            volume=2_000_000,
            source="test",
            interval=BarInterval.ONE_DAY,
        )
        for i in range(50)
    ]

    state = build_market_state(
        bars=bars,
        benchmark_bars=benchmark_bars,
    )

    assert state.symbol == "NVDA"
    assert state.interval == BarInterval.ONE_DAY
    assert state.timestamp == bars[-1].timestamp
    assert state.close == bars[-1].close
    assert state.volume == bars[-1].volume


def test_build_market_state_requires_50_price_bars():
    start = datetime(2026, 6, 1)

    bars = [
        PriceBar(
            symbol="NVDA",
            timestamp=start + timedelta(days=i),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=1_000_000,
            source="test",
            interval=BarInterval.ONE_DAY,
        )
        for i in range(49)
    ]

    benchmark_bars = [
        PriceBar(
            symbol="SPY",
            timestamp=start + timedelta(days=i),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=2_000_000,
            source="test",
            interval=BarInterval.ONE_DAY,
        )
        for i in range(50)
    ]

    with pytest.raises(ValueError):
        build_market_state(bars, benchmark_bars)

def test_build_market_state_rejects_mixed_symbols():
    start = datetime(2026, 6, 1)

    bars = [
        PriceBar(
            symbol="NVDA" if i < 49 else "AMD",
            timestamp=start + timedelta(days=i),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0 + i,
            volume=1_000_000,
            source="test",
            interval=BarInterval.ONE_DAY,
        )
        for i in range(50)
    ]

    benchmark_bars = [
        PriceBar(
            symbol="SPY",
            timestamp=start + timedelta(days=i),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=2_000_000,
            source="test",
            interval=BarInterval.ONE_DAY,
        )
        for i in range(50)
    ]

    with pytest.raises(ValueError):
        build_market_state(bars, benchmark_bars)

def test_build_market_state_rejects_mixed_intervals():
    start = datetime(2026, 6, 1)

    bars = [
        PriceBar(
            symbol="NVDA",
            timestamp=start + timedelta(days=i),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0 + i,
            volume=1_000_000,
            source="test",
            interval=(
                BarInterval.ONE_DAY
                if i < 49
                else BarInterval.FIVE_MINUTES
            ),
        )
        for i in range(50)
    ]

    benchmark_bars = [
        PriceBar(
            symbol="SPY",
            timestamp=start + timedelta(days=i),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=2_000_000,
            source="test",
            interval=BarInterval.ONE_DAY,
        )
        for i in range(50)
    ]

    with pytest.raises(ValueError):
        build_market_state(bars, benchmark_bars)

def test_build_market_state_rejects_benchmark_interval_mismatch():
    start = datetime(2026, 6, 1)

    bars = [
        PriceBar(
            symbol="NVDA",
            timestamp=start + timedelta(days=i),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0 + i,
            volume=1_000_000,
            source="test",
            interval=BarInterval.ONE_DAY,
        )
        for i in range(50)
    ]

    benchmark_bars = [
        PriceBar(
            symbol="SPY",
            timestamp=start + timedelta(days=i),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=2_000_000,
            source="test",
            interval=BarInterval.FIVE_MINUTES,
        )
        for i in range(50)
    ]

    with pytest.raises(ValueError):
        build_market_state(bars, benchmark_bars)


def test_build_market_state_rejects_misaligned_benchmark_timestamp():
    start = datetime(2026, 6, 1)

    bars = [
        PriceBar(
            symbol="NVDA",
            timestamp=start + timedelta(days=i),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0 + i,
            volume=1_000_000,
            source="test",
            interval=BarInterval.ONE_DAY,
        )
        for i in range(50)
    ]

    benchmark_bars = [
        PriceBar(
            symbol="SPY",
            timestamp=start + timedelta(days=i),
            open=100.0,
            high=101.0,
            low=99.0,
            close=100.0,
            volume=2_000_000,
            source="test",
            interval=BarInterval.ONE_DAY,
        )
        for i in range(50)
    ]

    benchmark_bars[-1] = PriceBar(
        symbol="SPY",
        timestamp=benchmark_bars[-1].timestamp + timedelta(days=1),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        volume=2_000_000,
        source="test",
        interval=BarInterval.ONE_DAY,
    )

    with pytest.raises(ValueError):
        build_market_state(
            bars=bars,
            benchmark_bars=benchmark_bars,
        )