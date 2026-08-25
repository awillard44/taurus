import pytest

from taurus.simulation.portfolio import PortfolioState
from taurus.simulation.valuation import revalue_portfolio


def test_revalue_portfolio_updates_value():
    state = PortfolioState(
        cash=100.0,
        shares=5.0,
        asset_price=200.0,
        portfolio_value=1_100.0,
    )

    result = revalue_portfolio(
        state,
        new_asset_price=210.0,
    )

    assert result.cash == 100.0
    assert result.shares == 5.0
    assert result.asset_price == 210.0
    assert result.portfolio_value == 1_150.0


def test_revalue_portfolio_rejects_invalid_price():
    state = PortfolioState(
        cash=1_000.0,
        shares=0.0,
        asset_price=200.0,
        portfolio_value=1_000.0,
    )

    with pytest.raises(ValueError):
        revalue_portfolio(
            state,
            new_asset_price=0.0,
        )