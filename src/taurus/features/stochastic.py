from statistics import mean


def calculate_stochastic(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    k_period: int = 14,
    d_period: int = 3,
) -> dict[str, float]:
    # Calculate stochastic oscillator %K and %D

    if k_period <= 0:
        raise ValueError("K period must be greater than zero.")

    if d_period <= 0:
        raise ValueError("D period must be greater than zero.")

    minimum_values = k_period + d_period - 1

    if (
        len(highs) < minimum_values
        or len(lows) < minimum_values
        or len(closes) < minimum_values
    ):
        raise ValueError(
            f"Expected at least {minimum_values} high, low, and close values."
        )

    k_values = []

    for end_index in range(
        k_period,
        len(closes) + 1,
    ):
        start_index = end_index - k_period

        window_highs = highs[start_index:end_index]
        window_lows = lows[start_index:end_index]

        highest_high = max(window_highs)
        lowest_low = min(window_lows)
        current_close = closes[end_index - 1]

        price_range = highest_high - lowest_low

        if price_range == 0:
            k_value = 50.0
        else:
            k_value = (
                (current_close - lowest_low)
                / price_range
            ) * 100.0

        k_values.append(k_value)

    current_k = k_values[-1]

    current_d = mean(
        k_values[-d_period:]
    )

    return {
        "k": current_k,
        "d": current_d,
    }