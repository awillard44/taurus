import pytest

from datetime import datetime
from statistics import mean, stdev

from taurus.evaluation.metrics import (
    calculate_average_gain,
    calculate_average_loss,
    calculate_excess_return,
    calculate_max_drawdown,
    calculate_portfolio_volatility,
    calculate_profit_factor,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_total_return,
    calculate_trade_count,
    calculate_win_rate,
)
from taurus.simulation.closed_trade import ClosedTrade

def build_closed_trade(
    realized_pnl: float,
) -> ClosedTrade:
    entry_value = 1_000.0
    exit_value = entry_value + realized_pnl

    return ClosedTrade(
        entry_timestamp=datetime(2026, 8, 25),
        exit_timestamp=datetime(2026, 8, 26),
        entry_price=100.0,
        exit_price=exit_value / 10.0,
        shares=10.0,
        entry_value=entry_value,
        exit_value=exit_value,
        realized_pnl=realized_pnl,
        return_pct=realized_pnl / entry_value,
    )

def test_calculate_trade_count():
    trades = [
        build_closed_trade(200.0),
        build_closed_trade(-50.0),
        build_closed_trade(100.0),
    ]

    assert calculate_trade_count(trades) == 3


def test_calculate_win_rate():
    trades = [
        build_closed_trade(200.0),
        build_closed_trade(-50.0),
        build_closed_trade(100.0),
        build_closed_trade(-25.0),
    ]

    assert calculate_win_rate(trades) == pytest.approx(0.50)


def test_calculate_win_rate_requires_closed_trade():
    with pytest.raises(ValueError):
        calculate_win_rate([])


def test_calculate_average_gain():
    trades = [
        build_closed_trade(200.0),
        build_closed_trade(-50.0),
        build_closed_trade(100.0),
    ]

    assert calculate_average_gain(trades) == pytest.approx(150.0)


def test_calculate_average_gain_returns_zero_without_winners():
    trades = [
        build_closed_trade(-50.0),
        build_closed_trade(-25.0),
    ]

    assert calculate_average_gain(trades) == 0.0


def test_calculate_average_loss():
    trades = [
        build_closed_trade(200.0),
        build_closed_trade(-50.0),
        build_closed_trade(-25.0),
    ]

    assert calculate_average_loss(trades) == pytest.approx(-37.5)


def test_calculate_average_loss_returns_zero_without_losers():
    trades = [
        build_closed_trade(200.0),
        build_closed_trade(100.0),
    ]

    assert calculate_average_loss(trades) == 0.0


def test_calculate_profit_factor():
    trades = [
        build_closed_trade(200.0),
        build_closed_trade(-50.0),
        build_closed_trade(100.0),
        build_closed_trade(-25.0),
    ]

    result = calculate_profit_factor(trades)

    assert result == pytest.approx(4.0)


def test_calculate_profit_factor_requires_realized_loss():
    trades = [
        build_closed_trade(200.0),
        build_closed_trade(100.0),
    ]

    with pytest.raises(ValueError):
        calculate_profit_factor(trades)



def test_calculate_total_return_positive():
    result = calculate_total_return(
        initial_value=1_000.0,
        final_value=1_200.0,
    )

    assert result == pytest.approx(0.20)


def test_calculate_total_return_negative():
    result = calculate_total_return(
        initial_value=1_000.0,
        final_value=900.0,
    )

    assert result == pytest.approx(-0.10)


def test_calculate_total_return_rejects_invalid_initial_value():
    with pytest.raises(ValueError):
        calculate_total_return(
            initial_value=0.0,
            final_value=1_000.0,
        )


def test_calculate_max_drawdown():
    portfolio_values = [
        1_000.0,
        1_100.0,
        1_050.0,
        900.0,
        950.0,
    ]

    result = calculate_max_drawdown(portfolio_values)

    assert result == pytest.approx(200.0 / 1_100.0)


def test_calculate_max_drawdown_is_zero_when_portfolio_only_rises():
    portfolio_values = [
        1_000.0,
        1_050.0,
        1_100.0,
        1_200.0,
    ]

    result = calculate_max_drawdown(portfolio_values)

    assert result == 0.0


def test_calculate_max_drawdown_rejects_empty_values():
    with pytest.raises(ValueError):
        calculate_max_drawdown([])

def test_calculate_portfolio_volatility():
    portfolio_values = [
        1_000.0,
        1_100.0,
        990.0,
        1_089.0,
    ]

    result = calculate_portfolio_volatility(
        portfolio_values
    )

    expected_returns = [
        0.10,
        -0.10,
        0.10,
    ]

    assert result == pytest.approx(
        stdev(expected_returns)
    )


def test_calculate_portfolio_volatility_rejects_insufficient_values():
    with pytest.raises(ValueError):
        calculate_portfolio_volatility(
            [1_000.0, 1_100.0]
        )


def test_calculate_excess_return():
    result = calculate_excess_return(
        portfolio_return=0.12,
        benchmark_return=0.08,
    )

    assert result == pytest.approx(0.04)


def test_calculate_negative_excess_return():
    result = calculate_excess_return(
        portfolio_return=0.05,
        benchmark_return=0.08,
    )

    assert result == pytest.approx(-0.03)

def test_calculate_sharpe_ratio():
    returns = [
        0.02,
        -0.01,
        0.03,
        0.01,
    ]

    result = calculate_sharpe_ratio(returns)

    expected = mean(returns) / stdev(returns)

    assert result == pytest.approx(expected)


def test_calculate_sharpe_ratio_rejects_zero_volatility():
    with pytest.raises(ValueError):
        calculate_sharpe_ratio([
            0.01,
            0.01,
            0.01,
        ])


def test_calculate_sortino_ratio():
    returns = [
        0.02,
        -0.01,
        0.03,
        -0.02,
    ]

    result = calculate_sortino_ratio(returns)

    downside_deviation = (
        ((-0.01) ** 2 + (-0.02) ** 2)
        / len(returns)
    ) ** 0.5

    expected = mean(returns) / downside_deviation

    assert result == pytest.approx(expected)


def test_calculate_sortino_ratio_requires_downside_returns():
    with pytest.raises(ValueError):
        calculate_sortino_ratio([
            0.01,
            0.02,
            0.03,
        ])