def calculate_atr(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int,
) -> float:
    # Calculate Average True Range over a specified period

    if period <= 0:
        raise ValueError("Period must be greater than zero.")

    required_values = period + 1

    if (
        len(highs) < required_values
        or len(lows) < required_values
        or len(closes) < required_values
    ):
        raise ValueError(
            f"Expected at least {required_values} high, low, and close values."
        )

    highs = highs[-required_values:]
    lows = lows[-required_values:]
    closes = closes[-required_values:]

    true_ranges = []

    for i in range(1, len(closes)):
        current_high = highs[i]
        current_low = lows[i]
        previous_close = closes[i - 1]

        true_range = max(
            current_high - current_low,
            abs(current_high - previous_close),
            abs(current_low - previous_close),
        )

        true_ranges.append(true_range)

    return sum(true_ranges) / period