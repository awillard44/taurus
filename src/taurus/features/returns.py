def calculate_return(
    closing_prices: list[float],
    periods: int,
) -> float:
    """Calculate return over a specified number of periods."""

    if periods <= 0:
        raise ValueError("Periods must be greater than zero.")

    if len(closing_prices) < periods + 1:
        raise ValueError(
            f"Expected at least {periods + 1} closing prices."
        )

    starting_price = closing_prices[-(periods + 1)]
    current_price = closing_prices[-1]

    return (current_price - starting_price) / starting_price