import pytest

from datetime import datetime

from taurus.simulation.actions import TradingAction
from taurus.simulation.execution import execute_action
from taurus.simulation.portfolio import PortfolioState
from taurus.simulation.costs import ExecutionCosts


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

def test_execute_action_with_zero_costs_matches_default_behavior():
    state = PortfolioState(
        cash=1000.0,
        shares=0.0,
        asset_price=100.0,
        portfolio_value=1000.0,
    )

    default_result = execute_action(
        state=state,
        action=TradingAction.BUY,
        timestamp=None,
    )

    zero_cost_result = execute_action(
        state=state,
        action=TradingAction.BUY,
        timestamp=None,
        costs=ExecutionCosts(),
    )

    assert zero_cost_result == default_result

def test_buy_applies_commission_and_slippage():
    state = PortfolioState(
        cash=1000.0,
        shares=0.0,
        asset_price=100.0,
        portfolio_value=1000.0,
    )

    result = execute_action(
        state=state,
        action=TradingAction.BUY,
        timestamp=None,
        costs=ExecutionCosts(
            commission_rate=0.001,
            slippage_rate=0.001,
        ),
    )

    expected_execution_price = 100.1
    expected_commission = 1.0
    expected_cash_for_shares = 999.0
    expected_shares = (
        expected_cash_for_shares
        / expected_execution_price
    )

    assert result.trade is not None
    assert result.trade.price == pytest.approx(
        expected_execution_price
    )
    assert result.trade.value == pytest.approx(
        expected_cash_for_shares
    )
    assert result.trade.shares == pytest.approx(
        expected_shares
    )

    assert result.portfolio.cash == 0.0
    assert result.portfolio.shares == pytest.approx(
        expected_shares
    )
    assert result.portfolio.portfolio_value == pytest.approx(
        999.0
    )

def test_sell_applies_commission_and_slippage():
    state = PortfolioState(
        cash=0.0,
        shares=10.0,
        asset_price=100.0,
        portfolio_value=1000.0,
    )

    result = execute_action(
        state=state,
        action=TradingAction.SELL,
        timestamp=None,
        costs=ExecutionCosts(
            commission_rate=0.001,
            slippage_rate=0.001,
        ),
    )

    expected_execution_price = 99.9
    expected_gross_value = 999.0
    expected_commission = 0.999
    expected_net_value = 998.001

    assert result.trade is not None

    assert result.trade.price == pytest.approx(
        expected_execution_price
    )

    assert result.trade.shares == pytest.approx(
        10.0
    )

    assert result.trade.value == pytest.approx(
        expected_net_value
    )

    assert result.portfolio.cash == pytest.approx(
        expected_net_value
    )

    assert result.portfolio.shares == 0.0

    assert result.portfolio.portfolio_value == pytest.approx(
        expected_net_value
    )