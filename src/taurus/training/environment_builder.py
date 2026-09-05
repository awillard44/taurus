from typing import Literal

from taurus.data.schemas import BarInterval
from taurus.data.sqlite_repository import SQLitePriceBarRepository
from taurus.environment.feature_state_builder import (
    build_feature_state_sequence,
)
from taurus.environment.trading_environment import (
    TaurusTradingEnvironment,
)
from taurus.features.config import FeatureSetConfig
from taurus.features.presets import DEFAULT_FEATURE_SET
from taurus.simulation.costs import ExecutionCosts
from taurus.simulation.portfolio import PortfolioState
from taurus.training.experiment_config import (
    DateRange,
    TrainingExperimentConfig,
)



ExperimentSplit = Literal[
    "training",
    "validation",
    "test",
]


def _get_split_range(
    config: TrainingExperimentConfig,
    split: ExperimentSplit,
) -> DateRange:
    if split == "training":
        return config.training

    if split == "validation":
        return config.validation

    if split == "test":
        return config.test

    raise ValueError(f"Unsupported experiment split: {split}")


def build_experiment_environment(
    repository: SQLitePriceBarRepository,
    config: TrainingExperimentConfig,
    split: ExperimentSplit,
    feature_set: FeatureSetConfig = DEFAULT_FEATURE_SET,
    minimum_history: int = 50,
    initial_cash: float = 1_000.0,
    costs: ExecutionCosts = ExecutionCosts(
        commission_rate=0.001,
        slippage_rate=0.001,
    ),
) -> TaurusTradingEnvironment:
    if minimum_history <= 0:
        raise ValueError("Minimum history must be positive.")

    if initial_cash <= 0:
        raise ValueError("Initial cash must be positive.")

    split_range = _get_split_range(
        config=config,
        split=split,
    )

    bars = repository.get_bars(
        config.symbol,
        BarInterval.ONE_DAY,
    )

    benchmark_bars = repository.get_bars(
        config.benchmark_symbol,
        BarInterval.ONE_DAY,
    )

    context_bars = [
        bar
        for bar in bars
        if bar.timestamp.date() <= split_range.end
    ]

    context_benchmark_bars = [
        bar
        for bar in benchmark_bars
        if bar.timestamp.date() <= split_range.end
    ]

    context_feature_states = build_feature_state_sequence(
        bars=context_bars,
        benchmark_bars=context_benchmark_bars,
        feature_set=feature_set,
        minimum_history=minimum_history,
    )

    feature_states = [
        state
        for state in context_feature_states
        if (
            split_range.start
            <= state.market.timestamp.date()
            <= split_range.end
        )
    ]

    if not feature_states:
        raise ValueError(
            f"No feature states available for {split} split."
        )

    initial_portfolio = PortfolioState(
        cash=initial_cash,
        shares=0.0,
        asset_price=feature_states[0].market.close,
        portfolio_value=initial_cash,
    )

    return TaurusTradingEnvironment(
        feature_states=feature_states,
        initial_portfolio=initial_portfolio,
        costs=costs,
    )