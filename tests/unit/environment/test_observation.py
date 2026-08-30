import pytest
import numpy as np

from datetime import datetime

from taurus.data.schemas import BarInterval
from taurus.environment.observation import build_observation
from taurus.environment.state import EnvironmentState
from taurus.features.market_state import MarketState
from taurus.simulation.portfolio import PortfolioState
from taurus.environment.observation import build_feature_observation


def test_build_observation_returns_expected_values():
    market = MarketState(
        symbol="NVDA",
        timestamp=datetime(2026, 8, 25),
        interval=BarInterval.ONE_DAY,
        close=216.85,
        volume=92_250_395,
        return_1=-0.003,
        return_5=-0.038,
        return_20=0.039,
        volatility=0.024,
        sma_20=212.78,
        sma_50=207.29,
        rsi_14=67.36,
        volume_ratio=0.78,
        relative_return=0.006,
    )

    portfolio = PortfolioState(
        cash=1_000.0,
        shares=5.0,
        asset_price=216.85,
        portfolio_value=2_084.25,
    )

    state = EnvironmentState(
        market=market,
        portfolio=portfolio,
        step_index=10,
    )

    observation = build_observation(state)

    assert observation == [
        216.85,
        -0.003,
        -0.038,
        0.039,
        0.024,
        212.78,
        207.29,
        67.36,
        0.78,
        0.006,
        1_000.0,
        5.0,
        2_084.25,
    ]


def test_build_observation_has_expected_length():
    market = MarketState(
        symbol="NVDA",
        timestamp=datetime(2026, 8, 25),
        interval=BarInterval.ONE_DAY,
        close=216.85,
        volume=92_250_395,
        return_1=-0.003,
        return_5=-0.038,
        return_20=0.039,
        volatility=0.024,
        sma_20=212.78,
        sma_50=207.29,
        rsi_14=67.36,
        volume_ratio=0.78,
        relative_return=0.006,
    )

    portfolio = PortfolioState(
        cash=1_000.0,
        shares=5.0,
        asset_price=216.85,
        portfolio_value=2_084.25,
    )

    state = EnvironmentState(
        market=market,
        portfolio=portfolio,
        step_index=10,
    )

    observation = build_observation(state)

    assert len(observation) == 13

def test_build_feature_observation():
    features = {
        "return_1": 0.01,
        "rsi_14": 55.0,
        "adx_14_adx": 25.0,
    }

    portfolio = PortfolioState(
        cash=500.0,
        shares=5.0,
        asset_price=100.0,
        portfolio_value=1000.0,
    )

    observation = build_feature_observation(
        features=features,
        portfolio=portfolio,
        current_price=100.0,
        initial_portfolio_value=1000.0,
    )

    expected = np.asarray(
        [
            0.01,   # return_1
            0.55,   # RSI
            0.25,   # ADX
            0.50,   # cash / initial value
            0.50,   # shares * price / initial value
            1.00,   # portfolio value / initial value
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(
        observation,
        expected,
    )


def test_build_feature_observation_returns_float32():
    features = {
        "return_1": 0.01,
    }

    portfolio = PortfolioState(
        cash=1000.0,
        shares=0.0,
        asset_price=100.0,
        portfolio_value=1000.0,
    )

    observation = build_feature_observation(
        features=features,
        portfolio=portfolio,
        current_price=100.0,
        initial_portfolio_value=1000.0,
    )

    assert observation.dtype == np.float32


def test_build_feature_observation_shape_changes_with_features():
    portfolio = PortfolioState(
        cash=1000.0,
        shares=0.0,
        asset_price=100.0,
        portfolio_value=1000.0,
    )

    small_observation = build_feature_observation(
        features={
            "a": 1.0,
        },
        portfolio=portfolio,
        current_price=100.0,
        initial_portfolio_value=1000.0,
    )

    large_observation = build_feature_observation(
        features={
            "a": 1.0,
            "b": 2.0,
            "c": 3.0,
        },
        portfolio=portfolio,
        current_price=100.0,
        initial_portfolio_value=1000.0,
    )

    assert small_observation.shape == (4,)

def test_build_feature_observation_requires_positive_initial_portfolio_value():
    portfolio = PortfolioState(
        cash=1000.0,
        shares=0.0,
        asset_price=100.0,
        portfolio_value=1000.0,
    )

    with pytest.raises(ValueError):
        build_feature_observation(
            features={
                "return_1": 0.01,
            },
            portfolio=portfolio,
            current_price=100.0,
            initial_portfolio_value=0.0,
        )