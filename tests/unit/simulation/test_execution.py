from taurus.simulation.actions import TradingAction
from taurus.simulation.execution import execute_action
from taurus.simulation.portfolio import PortfolioState


def test_hold_returns_same_state():
    state = PortfolioState(
        cash=10_000.0,
        shares=5.0,
        asset_price=200.0,
        portfolio_value=11_000.0,
    )

    result = execute_action(
        state,
        TradingAction.HOLD,
    )

    assert result == state


def test_buy_converts_cash_to_shares():
    state = PortfolioState(
        cash=1_000.0,
        shares=0.0,
        asset_price=200.0,
        portfolio_value=1_000.0,
    )

    result = execute_action(
        state,
        TradingAction.BUY,
    )

    assert result.cash == 0.0
    assert result.shares == 5.0
    assert result.portfolio_value == 1_000.0


def test_sell_converts_shares_to_cash():
    state = PortfolioState(
        cash=100.0,
        shares=5.0,
        asset_price=200.0,
        portfolio_value=1_100.0,
    )

    result = execute_action(
        state,
        TradingAction.SELL,
    )

    assert result.cash == 1_100.0
    assert result.shares == 0.0
    assert result.portfolio_value == 1_100.0