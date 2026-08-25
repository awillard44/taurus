from taurus.simulation.actions import TradingAction
from taurus.simulation.portfolio import PortfolioState


def execute_action(
    state: PortfolioState,
    action: TradingAction,
) -> PortfolioState:
    """Apply a trading action and return the resulting portfolio state."""

    if action == TradingAction.HOLD:
        return state

    if action == TradingAction.BUY:
        if state.asset_price <= 0:
            raise ValueError("Asset price must be greater than zero.")

        shares_to_buy = state.cash / state.asset_price

        return PortfolioState(
            cash=0.0,
            shares=state.shares + shares_to_buy,
            asset_price=state.asset_price,
            portfolio_value=state.portfolio_value,
        )

    if action == TradingAction.SELL:
        cash_after_sale = (
            state.cash
            + state.shares * state.asset_price
        )

        return PortfolioState(
            cash=cash_after_sale,
            shares=0.0,
            asset_price=state.asset_price,
            portfolio_value=cash_after_sale,
        )

    raise ValueError(f"Unsupported trading action: {action}")