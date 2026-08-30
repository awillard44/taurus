from dataclasses import dataclass

from taurus.simulation.actions import TradingAction
from taurus.simulation.portfolio import PortfolioState
from taurus.simulation.trade import Trade
from taurus.simulation.costs import ExecutionCosts


@dataclass(frozen=True)
class ExecutionResult:
    portfolio: PortfolioState
    trade: Trade | None


def execute_action(
    state: PortfolioState,
    action: TradingAction,
    timestamp,
    costs: ExecutionCosts = ExecutionCosts(),
) -> ExecutionResult:
    # Apply a trading action and return the updated portfolio and trade

    if action == TradingAction.HOLD:
        return ExecutionResult(
            portfolio=state,
            trade=None,
        )

    if action == TradingAction.BUY and state.cash <= 0:
        return ExecutionResult(
            portfolio=state,
            trade=None,
        )

    if action == TradingAction.BUY:
        if state.asset_price <= 0:
            raise ValueError("Asset price must be greater than zero.")

        if state.asset_price <= 0:
            raise ValueError("Assert price must be greater than 0")

        execution_price = (
            state.asset_price * (1.0 + costs.slippage_rate)
        )

        available_cash = state.cash

        commission = (
            available_cash * costs.commission_rate
        )

        cash_for_shares = (available_cash - commission)

        shares_to_buy = (cash_for_shares / execution_price)

        next_portfolio = PortfolioState(
            cash=0.0,
            shares=state.shares + shares_to_buy,
            asset_price=state.asset_price,
            portfolio_value=state.portfolio_value - commission,
        )

        trade = Trade(
            timestamp=timestamp,
            action=TradingAction.BUY,
            price=execution_price,
            shares=shares_to_buy,
            value=cash_for_shares,
        )

        return ExecutionResult(
            portfolio=next_portfolio,
            trade=trade,
        )

    if action == TradingAction.SELL and state.shares <= 0:
        return ExecutionResult(
              portfolio=state,
              trade=None,
        )

    if action == TradingAction.SELL:
        execution_price = (
            state.asset_price * (1.0 - costs.slippage_rate)
        )

        shares_to_sell = state.shares

        gross_sale_value = (shares_to_sell * execution_price)

        commission = (gross_sale_value * costs.commission_rate)

        net_sale_value = (gross_sale_value - commission)

        next_portfolio = PortfolioState(
            cash=state.cash + net_sale_value,
            shares=0.0,
            asset_price=state.asset_price,
            portfolio_value=state.cash + net_sale_value,
        )

        trade = Trade(
            timestamp=timestamp,
            action=TradingAction.SELL,
            price=execution_price,
            shares=shares_to_sell,
            value=net_sale_value,
        )

        return ExecutionResult(
            portfolio=next_portfolio,
            trade=trade,
        )

    raise ValueError(f"Unsupported trading action: {action}")