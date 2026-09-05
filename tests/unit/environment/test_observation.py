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

def test_allocation_v2_records_allocation_proportions():
    portfolio = PortfolioState(
        cash=500.0,
        shares=15.0,
        asset_price=100.0,
        portfolio_value=2000.0,
    )

    observation = build_feature_observation(
        features={"return_1": 0.01, "rsi_14": 60.0},
        portfolio=portfolio,
        current_price=100.0,
        initial_portfolio_value=1000.0,
        observation_version="allocation-v2",
    )

    np.testing.assert_allclose(
        observation,
        [0.01, 0.60, 0.25, 0.75],
    )
    assert observation.dtype == np.float32

@pytest.mark.parametrize("cash_fraction", [0.0, 0.25, 1.0])
def test_allocation_v2_is_independent_of_account_size(
    cash_fraction,
):
    observations = []

    for account_value in (500.0, 1000.0, 2000.0, 4000.0):
        portfolio = PortfolioState(
            cash=account_value * cash_fraction,
            shares=account_value * (1.0 - cash_fraction) / 100.0,
            asset_price=100.0,
            portfolio_value=account_value,
        )

        observations.append(
            build_feature_observation(
                features={"return_1": 0.01, "rsi_14": 60.0},
                portfolio=portfolio,
                current_price=100.0,
                initial_portfolio_value=1000.0,
                observation_version="allocation-v2",
            )
        )

    for observation in observations[1:]:
        np.testing.assert_array_equal(
            observation,
            observations[0],
        )

def test_explicit_v1_preserves_original_observation():
    portfolio = PortfolioState(
        cash=500.0,
        shares=15.0,
        asset_price=100.0,
        portfolio_value=2000.0,
    )

    arguments = {
        "features": {"return_1": 0.01},
        "portfolio": portfolio,
        "current_price": 100.0,
        "initial_portfolio_value": 1000.0,
    }

    default_observation = build_feature_observation(**arguments)
    explicit_observation = build_feature_observation(
        **arguments,
        observation_version="initial-capital-v1",
    )

    np.testing.assert_array_equal(
        explicit_observation,
        default_observation,
    )
    np.testing.assert_allclose(
        explicit_observation,
        [0.01, 0.50, 1.50, 2.00],
    )

@pytest.mark.parametrize("account_value", [0.0, -100.0])
def test_allocation_v2_rejects_nonpositive_account_value(
    account_value,
):
    portfolio = PortfolioState(
        cash=account_value,
        shares=0.0,
        asset_price=100.0,
        portfolio_value=account_value,
    )

    with pytest.raises(
        ValueError,
        match="Current portfolio value must be greater than zero",
    ):
        build_feature_observation(
            features={"return_1": 0.01},
            portfolio=portfolio,
            current_price=100.0,
            initial_portfolio_value=1000.0,
            observation_version="allocation-v2",
        )


def test_feature_observation_rejects_unknown_version():
    portfolio = PortfolioState(
        cash=1000.0,
        shares=0.0,
        asset_price=100.0,
        portfolio_value=1000.0,
    )

    with pytest.raises(
        ValueError,
        match="Unsupported observation version",
    ):
        build_feature_observation(
            features={"return_1": 0.01},
            portfolio=portfolio,
            current_price=100.0,
            initial_portfolio_value=1000.0,
            observation_version="unknown",
        )