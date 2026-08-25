from datetime import datetime

from taurus.data.schemas import BarInterval
from taurus.environment.state import EnvironmentState
from taurus.features.market_state import MarketState
from taurus.models.baselines.always_buy import AlwaysBuyAgent
from taurus.simulation.actions import TradingAction
from taurus.simulation.portfolio import PortfolioState


def build_state(cash: float, shares: float) -> EnvironmentState:
    market = MarketState(
        symbol="NVDA",
        timestamp=datetime(2026, 8, 25),
        interval=BarInterval.ONE_DAY,
        close=200.0,
        volume=1_000_000,
        return_1=0.0,
        return_5=0.05,
        return_20=0.10,
        volatility=0.02,
        sma_20=195.0,
        sma_50=190.0,
        rsi_14=60.0,
        volume_ratio=1.0,
        relative_return=0.02,
    )

    portfolio = PortfolioState(
        cash=cash,
        shares=shares,
        asset_price=200.0,
        portfolio_value=cash + (shares * 200.0),
    )

    return EnvironmentState(
        market=market,
        portfolio=portfolio,
        step_index=0,
    )


def test_always_buy_agent_buys_when_cash_is_available():
    agent = AlwaysBuyAgent()

    state = build_state(
        cash=1_000.0,
        shares=0.0,
    )

    action = agent.predict(state)

    assert action == TradingAction.BUY


def test_always_buy_agent_holds_when_fully_invested():
    agent = AlwaysBuyAgent()

    state = build_state(
        cash=0.0,
        shares=5.0,
    )

    action = agent.predict(state)

    assert action == TradingAction.HOLD