from statistics import stdev

def calculate_volatility_20d(daily_returns: list[float]) -> float:
    # Calculate the standard deviation of 20-day returns

    if len(daily_returns) != 20:
        raise ValueError("expected exactly 20 daily returns")

    return stdev(daily_returns)