from taurus.data.schemas import PriceBar
from taurus.features.market_state import MarketState
from taurus.features.state_builder import build_market_state


def build_market_state_sequence(
    bars: list[PriceBar],
    benchmark_bars: list[PriceBar],
    minimum_history: int = 50,
) -> list[MarketState]:
    # Build sequential MarketState snapshots from historical price bars

    if len(bars) < minimum_history:
        raise ValueError(
            f"At least {minimum_history} price bars are required."
        )

    if len(benchmark_bars) < minimum_history:
        raise ValueError(
            f"At least {minimum_history} benchmark bars are required."
        )

    states = []

    for end_index in range(
        minimum_history,
        len(bars) + 1,
    ):
        state = build_market_state(
            bars=bars[:end_index],
            benchmark_bars=benchmark_bars[:end_index],
        )

        states.append(state)

    return states