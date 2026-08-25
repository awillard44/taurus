from datetime import datetime, timedelta

import pytest

from taurus.data.schemas import BarInterval, PriceBar
from taurus.features.state_sequence import build_market_state_sequence


def build_price_bar(
    *,
    symbol: str,
    timestamp: datetime,
    close: float,
    volume: float = 1_000_000,
) -> PriceBar:
    return PriceBar(
        symbol=symbol,
        timestamp=timestamp,
        open=close,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=volume,
        source="test",
        interval=BarInterval.ONE_DAY,
    )


def test_build_market_state_sequence_creates_expected_number_of_states():
    start = datetime(2026, 6, 1)

    bars = [
        build_price_bar(
            symbol="NVDA",
            timestamp=start + timedelta(days=i),
            close=100.0 + i,
        )
        for i in range(52)
    ]

    benchmark_bars = [
        build_price_bar(
            symbol="SPY",
            timestamp=start + timedelta(days=i),
            close=100.0 + (i * 0.5),
            volume=2_000_000,
        )
        for i in range(52)
    ]

    states = build_market_state_sequence(
        bars=bars,
        benchmark_bars=benchmark_bars,
        minimum_history=50,
    )

    assert len(states) == 3
    assert states[0].timestamp == bars[49].timestamp
    assert states[1].timestamp == bars[50].timestamp
    assert states[2].timestamp == bars[51].timestamp

def test_build_market_state_sequence_requires_enough_history():
    start = datetime(2026, 6, 1)

    bars = [
        build_price_bar(
            symbol="NVDA",
            timestamp=start + timedelta(days=i),
            close=100.0 + i,
        )
        for i in range(49)
    ]

    benchmark_bars = [
        build_price_bar(
            symbol="SPY",
            timestamp=start + timedelta(days=i),
            close=100.0 + i,
        )
        for i in range(49)
    ]

    with pytest.raises(ValueError):
        build_market_state_sequence(
            bars=bars,
            benchmark_bars=benchmark_bars,
            minimum_history=50,
        )