from taurus.data.schemas import PriceBar
from taurus.features.market_state import MarketState
from taurus.features.moving_average import calculate_sma
from taurus.features.relative_return import calculate_relative_return
from taurus.features.returns import calculate_return
from taurus.features.rsi import calculate_rsi
from taurus.features.volume import calculate_volume_ratio
from taurus.features.volatility import calculate_volatility


def build_market_state(
    bars: list[PriceBar],
    benchmark_bars: list[PriceBar],
) -> MarketState:
    # Build a MarketState from market price bars

    if not bars:
        raise ValueError("Price bars cannot be empty.")

    if not benchmark_bars:
        raise ValueError("Benchmark price bars cannot be empty.")

    symbol = bars[0].symbol
    interval = bars[0].interval

    if any(bar.symbol != symbol for bar in bars):
        raise ValueError("All price bars must have the same symbol.")

    if any(bar.interval != interval for bar in bars):
        raise ValueError("All price bars must have the same interval.")

    if len(bars) < 50:
        raise ValueError("At least 50 price bars are required to build MarketState.")

    if len(benchmark_bars) < 21:
        raise ValueError("At least 21 benchmark bars are required to build MarketState")

    if any(bar.interval != interval for bar in benchmark_bars):
        raise ValueError("Benchmark bars must use the same interval as price bars.")

    if bars[-1].timestamp != benchmark_bars[-1].timestamp:
        raise ValueError(
            "Price bars and benchmark bars must end at the same timestamp."
        )

    closing_prices = [bar.close for bar in bars]
    volumes = [bar.volume for bar in bars]
    benchmark_prices = [bar.close for bar in benchmark_bars]

    returns = [
        calculate_return(closing_prices[:i + 1], periods=1)
        for i in range(1, len(closing_prices))
    ]

    return MarketState(
        symbol=symbol,
        timestamp=bars[-1].timestamp,
        interval=interval,
        close=closing_prices[-1],
        volume=volumes[-1],
        return_1=calculate_return(closing_prices, periods=1),
        return_5=calculate_return(closing_prices, periods=5),
        return_20=calculate_return(closing_prices, periods=20),
        volatility=calculate_volatility(returns, period=20),
        sma_20=calculate_sma(closing_prices, period=20),
        sma_50=calculate_sma(closing_prices, period=50),
        rsi_14=calculate_rsi(closing_prices, period=14),
        volume_ratio=calculate_volume_ratio(volumes, period=20),
        relative_return=calculate_relative_return(
            closing_prices,
            benchmark_prices,
            periods=20,
        ),
    )