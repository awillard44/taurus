from datetime import datetime

from taurus.data.schemas import BarInterval
from taurus.environment.feature_state import FeatureEnvironmentState
from taurus.features.market_state import MarketState


def test_feature_environment_state_stores_market_and_features():
    market = MarketState(
        symbol="TEST",
        timestamp=datetime(2026, 1, 1),
        interval=BarInterval.ONE_DAY,
        close=100.0,
        volume=1_000_000.0,
        return_1=0.01,
        return_5=0.02,
        return_20=0.05,
        volatility=0.02,
        sma_20=98.0,
        sma_50=95.0,
        rsi_14=55.0,
        volume_ratio=1.1,
        relative_return=0.01,
    )

    state = FeatureEnvironmentState(
        market=market,
        features={
            "return_1": 0.01,
            "rsi_14": 55.0,
        },
    )

    assert state.market is market
    assert state.features["return_1"] == 0.01
    assert state.features["rsi_14"] == 55.0