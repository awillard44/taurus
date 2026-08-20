def calculate_sma(
    closing_prices: list[float],
    period: int,
) -> float:
    # Calculate a simple moving average over a specified period

    if period <= 0:
        raise ValueError("Period must be greater than zero.")

    if len(closing_prices) < period:
        raise ValueError(
            f"Expected at least {period} closing prices."
        )

    window = closing_prices[-period:]

    return sum(window) / period