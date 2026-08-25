from datetime import datetime

from taurus.data.schemas import BarInterval
from taurus.features.market_state import MarketState
from taurus.models.baselines.momentum import MomentumAgent
from taurus.simulation.actions import TradingAction
from taurus.simulation.portfolio import PortfolioState


def build_market(return_5: float) -> MarketState:
    return MarketState(
        symbol="NVDA",
        timestamp=datetime(2026, 8, 25),
        interval=BarInterval.ONE_DAY,
        close=200.0,
        volume=1_000_000,
        return_1=0.0,
        return_5=return_5,
        return_20=0.0,
        volatility=0.02,
        sma_20=200.0,
        sma_50=200.0,
        rsi_14=50.0,
        volume_ratio=1.0,
        relative_return=0.0,
    )


def test_momentum_agent_buys_on_positive_momentum():
    agent = MomentumAgent()

    market = build_market(return_5=0.05)

    portfolio = PortfolioState(
        cash=1_000.0,
        shares=0.0,
        asset_price=200.0,
        portfolio_value=1_000.0,
    )

    action = agent.predict(market, portfolio)

    assert action == TradingAction.BUY


def test_momentum_agent_sells_on_negative_momentum():
    agent = MomentumAgent()

    market = build_market(return_5=-0.05)

    portfolio = PortfolioState(
        cash=0.0,
        shares=5.0,
        asset_price=200.0,
        portfolio_value=1_000.0,
    )

    action = agent.predict(market, portfolio)

    assert action == TradingAction.SELL


def test_momentum_agent_holds_when_no_action_is_needed():
    agent = MomentumAgent()

    market = build_market(return_5=0.0)

    portfolio = PortfolioState(
        cash=1_000.0,
        shares=0.0,
        asset_price=200.0,
        portfolio_value=1_000.0,
    )

    action = agent.predict(market, portfolio)

    assert action == TradingAction.HOLD