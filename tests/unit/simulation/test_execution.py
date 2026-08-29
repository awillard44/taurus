from datetime import datetime

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
        timestamp=datetime(2026, 8, 25),
    )

    assert result.portfolio == state
    assert result.trade is None


def test_buy_converts_cash_to_shares():
    state = PortfolioState(
        cash=1_000.0,
        shares=0.0,
        asset_price=200.0,
        portfolio_value=1_000.0,
    )

    timestamp = datetime(2026, 8, 25)

    result = execute_action(
        state,
        TradingAction.BUY,
        timestamp=timestamp,
    )

    assert result.portfolio.cash == 0.0
    assert result.portfolio.shares == 5.0
    assert result.portfolio.portfolio_value == 1_000.0

    assert result.trade is not None
    assert result.trade.timestamp == timestamp
    assert result.trade.action == TradingAction.BUY
    assert result.trade.price == 200.0
    assert result.trade.shares == 5.0
    assert result.trade.value == 1_000.0


def test_sell_converts_shares_to_cash():
    state = PortfolioState(
        cash=100.0,
        shares=5.0,
        asset_price=200.0,
        portfolio_value=1_100.0,
    )

    timestamp = datetime(2026, 8, 25)

    result = execute_action(
        state,
        TradingAction.SELL,
        timestamp=timestamp,
    )

    assert result.portfolio.cash == 1_100.0
    assert result.portfolio.shares == 0.0
    assert result.portfolio.portfolio_value == 1_100.0

    assert result.trade is not None
    assert result.trade.timestamp == timestamp
    assert result.trade.action == TradingAction.SELL
    assert result.trade.price == 200.0
    assert result.trade.shares == 5.0
    assert result.trade.value == 1_000.0

def test_buy_with_no_cash_does_nothing():
    state = PortfolioState(
        cash=0.0,
        shares=5.0,
        asset_price=100.0,
        portfolio_value=500.0,
    )

    result = execute_action(
        state=state,
        action=TradingAction.BUY,
        timestamp=None,
    )

    assert result.portfolio == state
    assert result.trade is None

def test_sell_with_no_shares_does_nothing():
    state = PortfolioState(
        cash=1000.0,
        shares=0.0,
        asset_price=100.0,
        portfolio_value=1000.0,
    )

    result = execute_action(
        state=state,
        action=TradingAction.SELL,
        timestamp=None,
    )

    assert result.portfolio == state
    assert result.trade is None