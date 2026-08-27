def calculate_obv(
    closes: list[float],
    volumes: list[float],
    period: int,
) -> float:
    # Calculate On-Balance Volume over a specified period

    if period <= 0:
        raise ValueError("Period must be greater than zero.")

    required_values = period + 1

    if (
        len(closes) < required_values
        or len(volumes) < required_values
    ):
        raise ValueError(
            f"Expected at least {required_values} close and volume values."
        )

    closes = closes[-required_values:]
    volumes = volumes[-required_values:]

    obv = 0.0

    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv += volumes[i]

        elif closes[i] < closes[i - 1]:
            obv -= volumes[i]

    return obv