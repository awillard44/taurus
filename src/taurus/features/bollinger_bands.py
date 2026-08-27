from statistics import mean, pstdev


def calculate_bollinger_bands(
    closing_prices: list[float],
    period: int,
    stddev_multiplier: float = 2.0,
) -> dict[str, float]:
    # Calculate Bollinger Bands over a specified period

    if period <= 0:
        raise ValueError("Period must be greater than zero.")

    if stddev_multiplier <= 0:
        raise ValueError(
            "Standard deviation multiplier must be greater than zero."
        )

    if len(closing_prices) < period:
        raise ValueError(
            f"Expected at least {period} closing prices."
        )

    window = closing_prices[-period:]

    middle = mean(window)
    deviation = pstdev(window)

    return {
        "lower": middle - (stddev_multiplier * deviation),
        "middle": middle,
        "upper": middle + (stddev_multiplier * deviation),
    }