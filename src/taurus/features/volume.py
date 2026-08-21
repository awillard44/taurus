def calculate_volume_ratio(
    volumes: list[float],
    period: int,
) -> float:
    # Compare the current volume with the average volume of prior periods

    if period <= 0:
        raise ValueError("Period must be greater than zero.")

    if len(volumes) < period + 1:
        raise ValueError(
            f"Expected at least {period + 1} volume observations."
        )

    previous_volumes = volumes[-(period + 1):-1]
    current_volume = volumes[-1]

    average_volume = sum(previous_volumes) / period

    if average_volume == 0:
        raise ValueError("Average historical volume cannot be zero.")

    return current_volume / average_volume