import pytest

from taurus.simulation.portfolio import PortfolioState


def test_portfolio_state_stores_values():
    state = PortfolioState(
        cash=10_000.0,
        shares=5.0,
        asset_price=200.0,
        portfolio_value=11_000.0,
    )

    assert state.cash == 10_000.0
    assert state.shares == 5.0
    assert state.asset_price == 200.0
    assert state.portfolio_value == 11_000.0


def test_portfolio_state_is_immutable():
    state = PortfolioState(
        cash=10_000.0,
        shares=5.0,
        asset_price=200.0,
        portfolio_value=11_000.0,
    )

    with pytest.raises(Exception):
        state.cash = 9_000.0