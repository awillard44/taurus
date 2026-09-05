from datetime import date

from taurus.training.experiment_config import (
    DateRange,
    TrainingExperimentConfig,
)


NVDA_INITIAL_EXPERIMENT = TrainingExperimentConfig(
    name="nvda_initial",
    symbol="NVDA",
    benchmark_symbol="SPY",
    training=DateRange(
        start=date(2022, 1, 1),
        end=date(2024, 12, 31),
    ),
    validation=DateRange(
        start=date(2025, 1, 1),
        end=date(2025, 12, 31),
    ),
    test=DateRange(
        start=date(2026, 1, 1),
        end=date(2026, 8, 21),
    ),
    seed=42,
)