def calculate_ema(
    values: list[float],
    period: int,
) -> float:
    # Calculate an SMA-seeded exponential moving average

    if period <= 0:
        raise ValueError("Period must be greater than zero.")

    if len(values) < period:
        raise ValueError(
            f"Expected at least {period} values."
        )

    multiplier = 2.0 / (period + 1)

    initial_window = values[:period]
    ema = sum(initial_window) / period

    for value in values[period:]:
        ema = (
            value * multiplier
            + ema * (1.0 - multiplier)
        )

    return ema