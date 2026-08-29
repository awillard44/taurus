from taurus.data.schemas import PriceBar
from taurus.environment.feature_state import FeatureEnvironmentState
from taurus.features.config import FeatureSetConfig
from taurus.features.feature_set_executor import execute_feature_set
from taurus.features.state_builder import build_market_state


def build_feature_state_sequence(
    bars: list[PriceBar],
    benchmark_bars: list[PriceBar],
    feature_set: FeatureSetConfig,
    minimum_history: int = 50,
) -> list[FeatureEnvironmentState]:

    if minimum_history <= 0:
        raise ValueError("Minimum history must be great than 0")

    if len(bars) < minimum_history:
        raise ValueError("Not enough asset bars to build feature states")

    if len(benchmark_bars) < minimum_history:
        raise ValueError("Not enough bars to build feature states")

    if len(bars) != len(benchmark_bars):
        raise ValueError("Asset and benchmark bar sequences must have equal length")

    for asset_bar, benchmark_bar in zip(
        bars,
        benchmark_bars,
    ):
        if asset_bar.timestamp != benchmark_bar.timestamp:
            raise ValueError("Asset and benchmark bars must have matching timestamps")

        if asset_bar.interval != benchmark_bar.interval:
            raise ValueError("Asset and benchmark bars must have matching intervals")

    feature_states = []

    for end_index in range(
        minimum_history,
        len(bars) + 1,
    ):
        current_bars = bars[:end_index]
        current_benchmark_bars = benchmark_bars[:end_index]

        market_state = build_market_state(
            bars=current_bars,
            benchmark_bars=current_benchmark_bars,
        )    

        input_data = {
            "close": [
                bar.close
                for bar in current_bars
            ],
            "high": [
                bar.high
                for bar in current_bars
            ],
            "low": [
                bar.low
                for bar in current_bars
            ],
            "volume": [
                bar.volume
                for bar in current_bars
            ],
            "benchmark_close": [
                bar.close
                for bar in current_benchmark_bars
            ],
        }

        features = execute_feature_set(
            config=feature_set,
            input_data=input_data,
        )

        feature_state = FeatureEnvironmentState(
            market=market_state,
            features=features,
        )

        feature_states.append(feature_state)

    return feature_states