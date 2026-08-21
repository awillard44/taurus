def calculate_rsi(
    closing_prices: list[float],
    period: int,
) -> float:
    """Calculate the Relative Strength Index over a specified period."""

    if period <= 0:
        raise ValueError("Period must be greater than zero.")

    if len(closing_prices) < period + 1:
        raise ValueError(
            f"Expected at least {period + 1} closing prices."
        )

    prices = closing_prices[-(period + 1):]

    changes = [
        prices[i] - prices[i - 1]
        for i in range(1, len(prices))
    ]

    gains = [
        change if change > 0 else 0.0
        for change in changes
    ]

    losses = [
        -change if change < 0 else 0.0
        for change in changes
    ]

    average_gain = sum(gains) / period
    average_loss = sum(losses) / period

    if average_loss == 0:
        return 100.0

    relative_strength = average_gain / average_loss

    return 100 - (100 / (1 + relative_strength))