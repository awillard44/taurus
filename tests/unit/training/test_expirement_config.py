from datetime import date

import pytest

from taurus.training.experiment_config import (
    DateRange,
    TrainingExperimentConfig,
)


def test_date_range_rejects_reversed_dates():
    with pytest.raises(ValueError):
        DateRange(
            start=date(2025, 1, 2),
            end=date(2025, 1, 1),
        )


def test_training_experiment_requires_chronological_splits():
    with pytest.raises(ValueError):
        TrainingExperimentConfig(
            name="invalid",
            symbol="NVDA",
            benchmark_symbol="SPY",
            training=DateRange(
                start=date(2022, 1, 1),
                end=date(2025, 6, 1),
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

def test_training_experiment_accepts_separate_splits():
    config = TrainingExperimentConfig(
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

    assert config.symbol == "NVDA"
    assert config.seed == 42