import numpy as np

from taurus.simulation.portfolio import PortfolioState
from taurus.environment.state import EnvironmentState

def build_observation(
    state: EnvironmentState,
) -> list[float]:
    # Convert an EnvironmentState into a numeric model observation

    return [
        state.market.close,
        state.market.return_1,
        state.market.return_5,
        state.market.return_20,
        state.market.volatility,
        state.market.sma_20,
        state.market.sma_50,
        state.market.rsi_14,
        state.market.volume_ratio,
        state.market.relative_return,
        state.portfolio.cash,
        state.portfolio.shares,
        state.portfolio.portfolio_value,
    ]

def build_feature_observation(
    features: dict[str, float],
    portfolio: PortfolioState,
) -> np.ndarray:
    # Build an RL observation from market features and portfolio state

    market_values = [
        float(value)
        for value in features.values()
    ]

    portfolio_values = [
        float(portfolio.cash),
        float(portfolio.shares),
        float(portfolio.portfolio_value),
    ]

    return np.asarray(
        market_values + portfolio_values,
        dtype=np.float32,
    )