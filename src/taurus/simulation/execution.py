from dataclasses import dataclass

from taurus.simulation.actions import TradingAction
from taurus.simulation.portfolio import PortfolioState
from taurus.simulation.trade import Trade


@dataclass(frozen=True)
class ExecutionResult:
    portfolio: PortfolioState
    trade: Trade | None


def execute_action(
    state: PortfolioState,
    action: TradingAction,
    timestamp,
) -> ExecutionResult:
    # Apply a trading action and return the updated portfolio and trade

    if action == TradingAction.HOLD:
        return ExecutionResult(
            portfolio=state,
            trade=None,
        )

    if action == TradingAction.BUY:
        if state.asset_price <= 0:
            raise ValueError("Asset price must be greater than zero.")

        shares_to_buy = state.cash / state.asset_price

        next_portfolio = PortfolioState(
            cash=0.0,
            shares=state.shares + shares_to_buy,
            asset_price=state.asset_price,
            portfolio_value=state.portfolio_value,
        )

        trade = Trade(
            timestamp=timestamp,
            action=TradingAction.BUY,
            price=state.asset_price,
            shares=shares_to_buy,
            value=state.cash,
        )

        return ExecutionResult(
            portfolio=next_portfolio,
            trade=trade,
        )

    if action == TradingAction.SELL:
        shares_to_sell = state.shares
        sale_value = shares_to_sell * state.asset_price

        next_portfolio = PortfolioState(
            cash=state.cash + sale_value,
            shares=0.0,
            asset_price=state.asset_price,
            portfolio_value=state.cash + sale_value,
        )

        trade = Trade(
            timestamp=timestamp,
            action=TradingAction.SELL,
            price=state.asset_price,
            shares=shares_to_sell,
            value=sale_value,
        )

        return ExecutionResult(
            portfolio=next_portfolio,
            trade=trade,
        )

    raise ValueError(f"Unsupported trading action: {action}")