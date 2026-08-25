from taurus.models.baselines.always_buy import AlwaysBuyAgent
from taurus.simulation.actions import TradingAction
from taurus.simulation.portfolio import PortfolioState


def test_always_buy_agent_buys_when_cash_is_available():
    agent = AlwaysBuyAgent()

    portfolio = PortfolioState(
        cash=1_000.0,
        shares=0.0,
        asset_price=200.0,
        portfolio_value=1_000.0,
    )

    action = agent.predict(portfolio)

    assert action == TradingAction.BUY


def test_always_buy_agent_holds_when_fully_invested():
    agent = AlwaysBuyAgent()

    portfolio = PortfolioState(
        cash=0.0,
        shares=5.0,
        asset_price=200.0,
        portfolio_value=1_000.0,
    )

    action = agent.predict(portfolio)

    assert action == TradingAction.HOLD