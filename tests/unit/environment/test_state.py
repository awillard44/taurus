from datetime import datetime

import pytest

from taurus.data.schemas import BarInterval
from taurus.environment.state import EnvironmentState
from taurus.features.market_state import MarketState
from taurus.simulation.portfolio import PortfolioState


def test_environment_state_stores_market_and_portfolio_state():
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

    assert state.market == market
    assert state.portfolio == portfolio
    assert state.step_index == 10


def test_environment_state_is_immutable():
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

    with pytest.raises(Exception):
        state.step_index = 11