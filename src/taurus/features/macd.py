from taurus.features.ema import calculate_ema


def calculate_macd(
    closing_prices: list[float],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> dict[str, float]:
    # Calculate MACD line, signal line, and histogram

    if fast_period <= 0:
        raise ValueError("Fast period must be greater than zero.")

    if slow_period <= 0:
        raise ValueError("Slow period must be greater than zero.")

    if signal_period <= 0:
        raise ValueError("Signal period must be greater than zero.")

    if fast_period >= slow_period:
        raise ValueError(
            "Fast period must be smaller than slow period."
        )

    minimum_values = slow_period + signal_period - 1

    if len(closing_prices) < minimum_values:
        raise ValueError(
            f"Expected at least {minimum_values} closing prices."
        )

    macd_values = []

    for end_index in range(
        slow_period,
        len(closing_prices) + 1,
    ):
        window = closing_prices[:end_index]

        fast_ema = calculate_ema(
            window,
            period=fast_period,
        )

        slow_ema = calculate_ema(
            window,
            period=slow_period,
        )

        macd_values.append(
            fast_ema - slow_ema
        )

    signal = calculate_ema(
        macd_values,
        period=signal_period,
    )

    macd = macd_values[-1]

    return {
        "macd": macd,
        "signal": signal,
        "histogram": macd - signal,
    }