def calculate_adx(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 14,
) -> dict[str, float]:
    # Calculate ADX, +DI, and -DI using Wilder's smoothing

    if period <= 0:
        raise ValueError("Period must be greater than zero.")

    minimum_values = period * 2

    if (
        len(highs) < minimum_values
        or len(lows) < minimum_values
        or len(closes) < minimum_values
    ):
        raise ValueError(
            f"Expected at least {minimum_values} high, low, and close values."
        )

    if not (
        len(highs) == len(lows) == len(closes)
    ):
        raise ValueError(
            "High, low, and close series must have the same length."
        )

    true_ranges = []
    plus_dm_values = []
    minus_dm_values = []

    for i in range(1, len(closes)):
        upward_move = highs[i] - highs[i - 1]
        downward_move = lows[i - 1] - lows[i]

        plus_dm = (
            upward_move
            if upward_move > downward_move
            and upward_move > 0
            else 0.0
        )

        minus_dm = (
            downward_move
            if downward_move > upward_move
            and downward_move > 0
            else 0.0
        )

        true_range = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

        true_ranges.append(true_range)
        plus_dm_values.append(plus_dm)
        minus_dm_values.append(minus_dm)

    smoothed_tr = sum(
        true_ranges[:period]
    )

    smoothed_plus_dm = sum(
        plus_dm_values[:period]
    )

    smoothed_minus_dm = sum(
        minus_dm_values[:period]
    )

    dx_values = []

    def calculate_directional_values(
        smoothed_true_range: float,
        smoothed_plus: float,
        smoothed_minus: float,
    ) -> tuple[float, float, float]:
        if smoothed_true_range == 0:
            return 0.0, 0.0, 0.0

        plus_di = (
            100.0
            * smoothed_plus
            / smoothed_true_range
        )

        minus_di = (
            100.0
            * smoothed_minus
            / smoothed_true_range
        )

        directional_sum = plus_di + minus_di

        if directional_sum == 0:
            dx = 0.0
        else:
            dx = (
                100.0
                * abs(plus_di - minus_di)
                / directional_sum
            )

        return plus_di, minus_di, dx

    plus_di, minus_di, dx = (
        calculate_directional_values(
            smoothed_tr,
            smoothed_plus_dm,
            smoothed_minus_dm,
        )
    )

    dx_values.append(dx)

    for i in range(
        period,
        len(true_ranges),
    ):
        smoothed_tr = (
            smoothed_tr
            - (smoothed_tr / period)
            + true_ranges[i]
        )

        smoothed_plus_dm = (
            smoothed_plus_dm
            - (smoothed_plus_dm / period)
            + plus_dm_values[i]
        )

        smoothed_minus_dm = (
            smoothed_minus_dm
            - (smoothed_minus_dm / period)
            + minus_dm_values[i]
        )

        plus_di, minus_di, dx = (
            calculate_directional_values(
                smoothed_tr,
                smoothed_plus_dm,
                smoothed_minus_dm,
            )
        )

        dx_values.append(dx)

    adx = sum(
        dx_values[:period]
    ) / period

    for dx in dx_values[period:]:
        adx = (
            (adx * (period - 1))
            + dx
        ) / period

    return {
        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di,
    }