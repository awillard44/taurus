import pytest
from datetime import datetime

from taurus.data.schemas import BarInterval
from taurus.features.market_state import MarketState


def test_market_state_stores_feature_snapshot():
    state = MarketState(
        symbol="NVDA",
        timestamp=datetime(2026, 8, 20),
        interval=BarInterval.ONE_DAY,
        close=182.75,
        volume=42_000_000,
        return_1=0.03,
        return_5=0.08,
        return_20=0.12,
        volatility=0.025,
        sma_20=175.0,
        sma_50=168.0,
        rsi_14=64.0,
        volume_ratio=1.4,
        relative_return=0.02,
    )

    assert state.symbol == "NVDA"
    assert state.interval == BarInterval.ONE_DAY
    assert state.rsi_14 == 64.0
    assert state.relative_return == 0.02


def test_market_state_is_immutable():
    state = MarketState(
        symbol="NVDA",
        timestamp=datetime(2026, 8, 20),
        interval=BarInterval.ONE_DAY,
        close=182.75,
        volume=42_000_000,
        return_1=0.03,
        return_5=0.08,
        return_20=0.12,
        volatility=0.025,
        sma_20=175.0,
        sma_50=168.0,
        rsi_14=64.0,
        volume_ratio=1.4,
        relative_return=0.02,
    )

    with pytest.raises(Exception):
        state.close = 200.0