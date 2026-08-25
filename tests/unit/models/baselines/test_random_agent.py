from datetime import datetime

from taurus.data.schemas import BarInterval
from taurus.environment.state import EnvironmentState
from taurus.features.market_state import MarketState
from taurus.models.baselines.random_agent import RandomAgent
from taurus.simulation.actions import TradingAction
from taurus.simulation.portfolio import PortfolioState


def test_random_agent_returns_valid_action():
    agent = RandomAgent()

    market = MarketState(
        symbol="NVDA",
        timestamp=datetime(2026, 8, 25),
        interval=BarInterval.ONE_DAY,
        close=200.0,
        volume=1_000_000,
        return_1=0.0,
        return_5=0.0,
        return_20=0.0,
        volatility=0.02,
        sma_20=200.0,
        sma_50=200.0,
        rsi_14=50.0,
        volume_ratio=1.0,
        relative_return=0.0,
    )

    portfolio = PortfolioState(
        cash=1_000.0,
        shares=0.0,
        asset_price=200.0,
        portfolio_value=1_000.0,
    )

    state = EnvironmentState(
        market=market,
        portfolio=portfolio,
        step_index=0,
    )

    action = agent.predict(state)

    assert action in TradingAction