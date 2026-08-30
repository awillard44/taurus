import numpy as np

from taurus.simulation.portfolio import PortfolioState
from taurus.environment.state import EnvironmentState
from taurus.environment.normalization import normalize_market_features


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
    current_price: float,
    initial_portfolio_value: float,
) -> np.ndarray:

    if initial_portfolio_value <= 0:
        raise ValueError(
            "Initial portfolio value must be greater than zero."
        )

    normalized_features = normalize_market_features(
        features=features,
        current_price=current_price,
    )

    normalized_cash = (
        portfolio.cash
        / initial_portfolio_value
    )

    normalized_portfolio_value = (
        portfolio.portfolio_value
        / initial_portfolio_value
    )

    normalized_shares = (
        portfolio.shares
        * current_price
        / initial_portfolio_value
    )

    market_values = [
        float(value)
        for value in normalized_features.values()
    ]

    portfolio_values = [
        float(normalized_cash),
        float(normalized_shares),
        float(normalized_portfolio_value),
    ]

    return np.asarray(
        market_values + portfolio_values,
        dtype=np.float32,
    )