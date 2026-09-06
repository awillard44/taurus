import pytest

from datetime import datetime

from taurus.simulation.actions import TradingAction
from taurus.simulation.portfolio import PortfolioState
from taurus.simulation.step import step_portfolio
from taurus.simulation.costs import ExecutionCosts


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
        timestamp=datetime(2026, 8, 25),
    )

    assert result.portfolio.cash == 0.0
    assert result.portfolio.shares == 5.0
    assert result.portfolio.asset_price == 210.0
    assert result.portfolio.portfolio_value == 1_050.0
    assert result.reward == pytest.approx(0.05)
    assert result.trade is not None


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
        timestamp=datetime(2026, 8, 25),
    )

    assert result.portfolio.cash == 100.0
    assert result.portfolio.shares == 5.0
    assert result.portfolio.asset_price == 190.0
    assert result.portfolio.portfolio_value == 1_050.0
    assert result.reward == pytest.approx(-50.0 / 1_100.0)
    assert result.trade is None

def test_step_portfolio_applies_execution_costs():
    state = PortfolioState(
        cash=1000.0,
        shares=0.0,
        asset_price=100.0,
        portfolio_value=1000.0,
    )

    result = step_portfolio(
        state=state,
        action=TradingAction.BUY,
        next_asset_price=100.0,
        timestamp=None,
        costs=ExecutionCosts(
            commission_rate=0.001,
            slippage_rate=0.001,
        ),
    )

    assert result.trade is not None
    assert result.trade.price == pytest.approx(100.1)

    assert result.portfolio.portfolio_value < 1000.0

@pytest.mark.parametrize(
    (
        "action, starting_cash, starting_shares, "
        "expected_cash, expected_shares, expected_value"
    ),
    [
        pytest.param(
            TradingAction.BUY,
            1000.0,
            0.0,
            0.0,
            1000.0 / 110.0,
            (1000.0 / 110.0) * 105.0,
            id="cash-buys-at-next-open",
        ),
        pytest.param(
            TradingAction.SELL,
            0.0,
            10.0,
            1100.0,
            0.0,
            1100.0,
            id="sell-retains-overnight-gain",
        ),
        pytest.param(
            TradingAction.HOLD,
            0.0,
            10.0,
            0.0,
            10.0,
            1050.0,
            id="hold-retains-full-close-to-close-move",
        ),
    ],
)
def test_next_open_execution(
    action,
    starting_cash,
    starting_shares,
    expected_cash,
    expected_shares,
    expected_value,
):
    # Monday's closing state: every case starts worth $1,000.
    state = PortfolioState(
        cash=starting_cash,
        shares=starting_shares,
        asset_price=100.0,
        portfolio_value=1000.0,
    )

    execution_time = datetime(2024, 1, 9, 9, 30)

    result = step_portfolio(
        state=state,
        action=action,
        next_asset_price=105.0,  # Tuesday close
        timestamp=execution_time,
        costs=ExecutionCosts(
            commission_rate=0.0,
            slippage_rate=0.0,
        ),
        execution_version="next-open-v2",
        next_open_price=110.0,
    )

    assert result.portfolio.cash == pytest.approx(expected_cash)
    assert result.portfolio.shares == pytest.approx(expected_shares)
    assert result.portfolio.asset_price == pytest.approx(105.0)
    assert result.portfolio.portfolio_value == pytest.approx(
        expected_value
    )

    # Reward includes the entire change since Monday's close.
    assert result.reward == pytest.approx(
        (expected_value - 1000.0) / 1000.0
    )

    if action == TradingAction.HOLD:
        assert result.trade is None
    else:
        assert result.trade is not None
        assert result.trade.action == action
        assert result.trade.price == pytest.approx(110.0)
        assert result.trade.timestamp == execution_time
        assert result.trade.shares == pytest.approx(
            expected_shares
            if action == TradingAction.BUY
            else starting_shares
        )

def test_next_open_buy_applies_costs_at_open():
    state = PortfolioState(
        cash=1000.0,
        shares=0.0,
        asset_price=100.0,
        portfolio_value=1000.0,
    )

    result = step_portfolio(
        state=state,
        action=TradingAction.BUY,
        next_asset_price=105.0,
        timestamp=datetime(2024, 1, 9, 9, 30),
        costs=ExecutionCosts(
            commission_rate=0.001,
            slippage_rate=0.001,
        ),
        execution_version="next-open-v2",
        next_open_price=110.0,
    )

    # $1 commission; buy at the open plus adverse slippage.
    expected_price = 110.0 * 1.001
    expected_shares = 999.0 / expected_price
    expected_value = expected_shares * 105.0

    assert result.trade is not None
    assert result.trade.price == pytest.approx(expected_price)
    assert result.trade.shares == pytest.approx(expected_shares)
    assert result.portfolio.cash == pytest.approx(0.0)
    assert result.portfolio.portfolio_value == pytest.approx(
        expected_value
    )
    assert result.reward == pytest.approx(
        (expected_value - 1000.0) / 1000.0
    )


def test_next_open_sell_applies_costs_after_overnight_move():
    state = PortfolioState(
        cash=0.0,
        shares=10.0,
        asset_price=100.0,
        portfolio_value=1000.0,
    )

    result = step_portfolio(
        state=state,
        action=TradingAction.SELL,
        next_asset_price=105.0,
        timestamp=datetime(2024, 1, 9, 9, 30),
        costs=ExecutionCosts(
            commission_rate=0.001,
            slippage_rate=0.001,
        ),
        execution_version="next-open-v2",
        next_open_price=110.0,
    )

    expected_price = 110.0 * 0.999
    gross_proceeds = 10.0 * expected_price
    expected_cash = gross_proceeds * 0.999

    assert result.trade is not None
    assert result.trade.price == pytest.approx(expected_price)
    assert result.trade.shares == pytest.approx(10.0)
    assert result.portfolio.shares == pytest.approx(0.0)
    assert result.portfolio.cash == pytest.approx(expected_cash)
    assert result.portfolio.portfolio_value == pytest.approx(
        expected_cash
    )
    assert result.reward == pytest.approx(
        (expected_cash - 1000.0) / 1000.0
    )