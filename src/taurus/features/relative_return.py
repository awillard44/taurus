def calculate_relative_return(
    asset_prices: list[float],
    benchmark_prices: list[float],
    periods: int,
) -> float:
    """Calculate asset return relative to a benchmark over a specified period."""

    if periods <= 0:
        raise ValueError("Periods must be greater than zero.")

    required_prices = periods + 1

    if len(asset_prices) < required_prices:
        raise ValueError(
            f"Expected at least {required_prices} asset prices."
        )

    if len(benchmark_prices) < required_prices:
        raise ValueError(
            f"Expected at least {required_prices} benchmark prices."
        )

    asset_start = asset_prices[-required_prices]
    asset_end = asset_prices[-1]

    benchmark_start = benchmark_prices[-required_prices]
    benchmark_end = benchmark_prices[-1]

    asset_return = (asset_end - asset_start) / asset_start
    benchmark_return = (
        benchmark_end - benchmark_start
    ) / benchmark_start

    return asset_return - benchmark_return