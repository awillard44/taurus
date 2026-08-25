from statistics import stdev, mean


def calculate_portfolio_volatility(
    portfolio_values: list[float],
) -> float:
    # Calculate volatility of period-to-period portfolio returns

    if len(portfolio_values) < 3:
        raise ValueError(
            "At least three portfolio values are required."
        )

    returns = [
        (portfolio_values[i] - portfolio_values[i - 1])
        / portfolio_values[i - 1]
        for i in range(1, len(portfolio_values))
    ]

    return stdev(returns)


def calculate_excess_return(
    portfolio_return: float,
    benchmark_return: float,
) -> float:
    # Calculate portfolio return relative to a benchmark

    return portfolio_return - benchmark_return

def calculate_total_return(
    initial_value: float,
    final_value: float,
) -> float:
    # Calculate total portfolio return

    if initial_value <= 0:
        raise ValueError("Initial portfolio value must be greater than zero.")

    return (final_value - initial_value) / initial_value


def calculate_max_drawdown(
    portfolio_values: list[float],
) -> float:
    # Calculate maximum portfolio drawdown

    if not portfolio_values:
        raise ValueError("Portfolio values cannot be empty.")

    peak = portfolio_values[0]
    max_drawdown = 0.0

    for value in portfolio_values:
        if value > peak:
            peak = value

        drawdown = (peak - value) / peak

        if drawdown > max_drawdown:
            max_drawdown = drawdown

    return max_drawdown

def calculate_sharpe_ratio(
        returns: list[float],
        risk_free_return: float = 0.0,
) -> float:
    # Calculate a period-based Sharpe ratio

    if len(returns) < 2:
        raise ValueError("At least two returns are required.")

    excess_returns = [
        value - risk_free_return
        for value in returns
    ]

    volatility = stdev(excess_returns)

    if volatility == 0:
        raise ValueError("Sharpe ratio is undefined when volatility is zero.")

    return mean(excess_returns) / volatility

def calculate_sortino_ratio(
        returns: list[float],
        target_return: float = 0.0,
) -> float:
    # Calculate a period-based Sortino ratio

    if len(returns) < 2:
        raise ValueError("At least two returns required.")

    excess_returns = [
        value - target_return
        for value in returns
    ]

    downside_returns = [
        value
        for value in excess_returns
        if value < 0
    ]

    if not downside_returns:
        raise ValueError(
            "Sortino ratio is undefined when there are no downside returns."
        )

    downside_deviation = (
        sum(value ** 2 for value in downside_returns)
        / len(excess_returns)
    ) ** 0.5

    if downside_deviation == 0:
        raise ValueError(
            "Sortino ratio is undefined when downside deviation is zero."
        )

    return mean(excess_returns) / downside_deviation