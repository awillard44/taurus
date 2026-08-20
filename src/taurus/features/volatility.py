from statistics import stdev

def calculate_volatility(
    returns: list[float],
    period: int,
) -> float:
    # Calculate return volatility over a specified period

    if period < 2:
        raise ValueError("Period must be at least two.")

    if len(returns) < period:
        raise ValueError(
            f"Expected at least {period} returns."
        )

    window = returns[-period:]

    return stdev(window)