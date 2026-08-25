from taurus.simulation.portfolio import PortfolioState


def revalue_portfolio(
    state: PortfolioState,
    new_asset_price: float,
) -> PortfolioState:
    """Update portfolio value using a new market price."""

    if new_asset_price <= 0:
        raise ValueError("Asset price must be greater than zero.")

    portfolio_value = (
        state.cash
        + state.shares * new_asset_price
    )

    return PortfolioState(
        cash=state.cash,
        shares=state.shares,
        asset_price=new_asset_price,
        portfolio_value=portfolio_value,
    )