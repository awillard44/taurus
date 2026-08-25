import pytest

from taurus.simulation.actions import TradingAction
from taurus.simulation.portfolio import PortfolioState
from taurus.simulation.step import step_portfolio


def test_step_portfolio_buy_then_price_increases():
    state = PortfolioState(
        cash=1_000.0,
        shares=0.0,
        asset_price=200.0,
        portfolio_value=1_000.0,
    )

    result = step_portfolio(
        state=state,
        action=TradingAction.BUY,
        next_asset_price=210.0,
    )

    assert result.portfolio.cash == 0.0
    assert result.portfolio.shares == 5.0
    assert result.portfolio.asset_price == 210.0
    assert result.portfolio.portfolio_value == 1_050.0
    assert result.reward == pytest.approx(0.05)


def test_step_portfolio_hold_then_price_changes():
    state = PortfolioState(
        cash=100.0,
        shares=5.0,
        asset_price=200.0,
        portfolio_value=1_100.0,
    )

    result = step_portfolio(
        state=state,
        action=TradingAction.HOLD,
        next_asset_price=190.0,
    )

    assert result.portfolio.cash == 100.0
    assert result.portfolio.shares == 5.0
    assert result.portfolio.asset_price == 190.0
    assert result.portfolio.portfolio_value == 1_050.0
    assert result.reward == pytest.approx(-50.0 / 1_100.0)