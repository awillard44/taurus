def calculate_vwap(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float],
    period: int,
) -> float:
    # Calculate Volume Weighted Average Price over a specified period

    if period <= 0:
        raise ValueError("Period must be greater than zero.")

    if (
        len(highs) < period
        or len(lows) < period
        or len(closes) < period
        or len(volumes) < period
    ):
        raise ValueError(
            f"Expected at least {period} high, low, close, and volume values."
        )

    highs = highs[-period:]
    lows = lows[-period:]
    closes = closes[-period:]
    volumes = volumes[-period:]

    total_volume = sum(volumes)

    if total_volume <= 0:
        raise ValueError("Total volume must be greater than zero.")

    weighted_value = 0.0

    for high, low, close, volume in zip(
        highs,
        lows,
        closes,
        volumes,
    ):
        typical_price = (
            high + low + close
        ) / 3.0

        weighted_value += (
            typical_price * volume
        )

    return weighted_value / total_volume