import numpy as np
import pytest

from datetime import datetime, timedelta

from taurus.data.schemas import BarInterval
from taurus.environment.observation import ObservationVersion
from taurus.environment.trading_environment import TaurusTradingEnvironment
from taurus.features.market_state import MarketState
from taurus.simulation.actions import TradingAction
from taurus.simulation.portfolio import PortfolioState
from taurus.environment.feature_state import FeatureEnvironmentState
from taurus.simulation.costs import ExecutionCosts

from gymnasium.utils.env_checker import check_env


def build_market_state(
    *,
    timestamp: datetime,
    close: float,
) -> MarketState:
    return MarketState(
        symbol="NVDA",
        timestamp=timestamp,
        interval=BarInterval.ONE_DAY,
        close=close,
        volume=1_000_000,
        return_1=0.0,
        return_5=0.0,
        return_20=0.0,
        volatility=0.02,
        sma_20=close,
        sma_50=close,
        rsi_14=50.0,
        volume_ratio=1.0,
        relative_return=0.0,
    )


def build_test_environment(
        observation_version: ObservationVersion = "initial-capital-v1",
) -> TaurusTradingEnvironment:
    start = datetime(2026, 8, 25)

    market_states = [
        build_market_state(
            timestamp=start,
            close=200.0,
        ),
        build_market_state(
            timestamp=start + timedelta(days=1),
            close=210.0,
        ),
        build_market_state(
            timestamp=start + timedelta(days=2),
            close=205.0,
        ),
    ]

    feature_states = [
        FeatureEnvironmentState(
            market=market_state,
            features={
                "return_1": market_state.return_1,
                "return_5": market_state.return_5,
                "return_20": market_state.return_20,
            },
        )
        for market_state in market_states
    ]

    initial_portfolio = PortfolioState(
        cash=1_000.0,
        shares=0.0,
        asset_price=200.0,
        portfolio_value=1_000.0,
    )

    return TaurusTradingEnvironment(
        feature_states=feature_states,
        initial_portfolio=initial_portfolio,
        observation_version=observation_version,
    )


def test_trading_environment_spaces():
    environment = build_test_environment()

    assert environment.action_space.n == 3
    assert environment.observation_space.shape == (6,)


def test_trading_environment_reset_returns_observation():
    environment = build_test_environment()

    observation, info = environment.reset()

    assert isinstance(observation, np.ndarray)
    assert observation.shape == (6,)
    assert observation.dtype == np.float32
    assert info == {}


def test_trading_environment_reset_restores_initial_state():
    environment = build_test_environment()

    environment.step(TradingAction.HOLD)

    environment.reset()

    assert environment.current_index == 0
    assert environment.state.step_index == 0
    assert environment.state.market.close == 200.0
    assert environment.state.portfolio.cash == 1_000.0
    assert environment.state.portfolio.shares == 0.0

    _, _, _, _, info = environment.step(
        TradingAction.HOLD
    )

    assert info["trade"] is None


def test_trading_environment_buy_then_price_increases():
    environment = build_test_environment()

    environment.reset()

    observation, reward, terminated, truncated, info = environment.step(
        TradingAction.BUY
    )

    assert info["trade"] is not None
    assert info["trade"].action == TradingAction.BUY
    assert info["trade"].price == 200.0
    assert info["trade"].shares == 5.0
    assert info["trade"].value == 1_000.0

    assert environment.current_index == 1

    assert environment.state.portfolio.cash == 0.0
    assert environment.state.portfolio.shares == 5.0
    assert environment.state.portfolio.asset_price == 210.0
    assert environment.state.portfolio.portfolio_value == 1_050.0

    assert reward == 0.05
    assert terminated is False
    assert truncated is False

    assert observation.shape == (6,)


def test_trading_environment_terminates_at_last_market_state():
    environment = build_test_environment()

    environment.reset()

    _, _, terminated, _, _ = environment.step(
        TradingAction.HOLD
    )

    assert terminated is False

    _, _, terminated, _, _ = environment.step(
        TradingAction.HOLD
    )

    assert terminated is True


def test_trading_environment_passes_gymnasium_checker():
    environment = build_test_environment()

    check_env(environment)

def test_trading_environment_applies_execution_costs():
    environment = build_test_environment()

    environment.costs = ExecutionCosts(
        commission_rate=0.001,
        slippage_rate=0.001,
    )

    environment.reset()

    observation, reward, terminated, truncated, info = environment.step(
        TradingAction.BUY
    )

    assert info["trade"] is not None
    assert info["trade"].price > 200.0

    assert environment.state.portfolio.portfolio_value < 1050.0

def test_allocation_v2_reset_matches_observation_space():
    environment = build_test_environment("allocation-v2")

    observation, _ = environment.reset()

    # Three market features plus two allocation inputs.
    assert environment.observation_space.shape == (5,)
    assert observation.shape == (5,)
    assert observation.dtype == np.float32
    assert np.isfinite(observation).all()
    assert environment.observation_space.contains(observation)

    np.testing.assert_allclose(
        observation[-2:],
        [1.0, 0.0],
    )


def test_allocation_v2_tracks_allocation_through_episode():
    environment = build_test_environment("allocation-v2")
    environment.reset()

    observation, reward, terminated, truncated, _ = (
        environment.step(TradingAction.BUY)
    )

    assert environment.state.portfolio.portfolio_value == 1050.0
    assert reward == pytest.approx(0.05)
    assert not terminated
    assert not truncated
    assert np.isfinite(observation).all()
    assert environment.observation_space.contains(observation)

    # Account growth does not turn full exposure into 1.05.
    np.testing.assert_allclose(observation[-2:], [0.0, 1.0])

    observation, _, terminated, _, _ = environment.step(
        TradingAction.SELL
    )

    assert terminated
    assert np.isfinite(observation).all()
    assert environment.observation_space.contains(observation)
    np.testing.assert_allclose(observation[-2:], [1.0, 0.0])

    observation, _ = environment.reset()

    assert environment.state.portfolio.portfolio_value == 1000.0
    assert environment.observation_space.contains(observation)
    np.testing.assert_allclose(observation[-2:], [1.0, 0.0])


def test_environment_rejects_unknown_observation_version():
    with pytest.raises(
        ValueError,
        match="Unsupported observation version",
    ):
        build_test_environment("unknown")