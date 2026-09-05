from dataclasses import dataclass

from taurus.data.sqlite_repository import SQLitePriceBarRepository
from taurus.environment.trading_environment import (
    TaurusTradingEnvironment,
)
from taurus.features.config import FeatureSetConfig
from taurus.features.presets import DEFAULT_FEATURE_SET
from taurus.simulation.costs import ExecutionCosts
from taurus.training.environment_builder import (
    build_experiment_environment,
)
from taurus.training.experiment_config import (
    TrainingExperimentConfig,
)
from taurus.training.target_position_environment import (
    TargetPositionEnvironment,
)


@dataclass(frozen=True)
class TrainingEnvironment:
    environment: TargetPositionEnvironment


def build_training_environment(
    repository: SQLitePriceBarRepository,
    config: TrainingExperimentConfig,
    feature_set: FeatureSetConfig = DEFAULT_FEATURE_SET,
    minimum_history: int = 50,
    initial_cash: float = 1_000.0,
    costs: ExecutionCosts = ExecutionCosts(
        commission_rate=0.001,
        slippage_rate=0.001,
    ),
) -> TrainingEnvironment:
    environment = build_experiment_environment(
        repository=repository,
        config=config,
        split="training",
        feature_set=feature_set,
        minimum_history=minimum_history,
        initial_cash=initial_cash,
        costs=costs,
    )

    target_position_environment = TargetPositionEnvironment(
        environment,
    )

    return TrainingEnvironment(
        environment=target_position_environment,
    )