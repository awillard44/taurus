import json
from datetime import date
from pathlib import Path

import pytest

from taurus.features.presets import DEFAULT_FEATURE_SET
from taurus.simulation.costs import ExecutionCosts
from taurus.training.experiment_config import (
    DateRange,
    TrainingExperimentConfig,
)
from taurus.training.run_metadata import (
    build_training_run_metadata,
    save_training_run_metadata,
)


def make_config() -> TrainingExperimentConfig:
    return TrainingExperimentConfig(
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


def test_build_training_run_metadata_records_experiment():
    metadata = build_training_run_metadata(
        run_name="nvda_ppo_100k",
        algorithm="PPO",
        config=make_config(),
        feature_set=DEFAULT_FEATURE_SET,
        costs=ExecutionCosts(
            commission_rate=0.001,
            slippage_rate=0.001,
        ),
        total_timesteps=100_000,
        action_space_version="target-position-v1",
        model_path=Path(
            "artifacts/training/nvda_ppo_100k/model.zip"
        ),
    )

    assert metadata.symbol == "NVDA"
    assert metadata.benchmark_symbol == "SPY"
    assert metadata.training_start == "2022-01-01"
    assert metadata.training_end == "2024-12-31"
    assert metadata.validation_start == "2025-01-01"
    assert metadata.validation_end == "2025-12-31"
    assert metadata.test_start == "2026-01-01"
    assert metadata.test_end == "2026-08-21"
    assert metadata.total_timesteps == 100_000
    assert metadata.seed == 42
    assert metadata.feature_set == "default"
    assert metadata.test_evaluated is False


def test_save_training_run_metadata_writes_json(
    tmp_path: Path,
):
    metadata = build_training_run_metadata(
        run_name="nvda_ppo_smoke",
        algorithm="PPO",
        config=make_config(),
        feature_set=DEFAULT_FEATURE_SET,
        costs=ExecutionCosts(
            commission_rate=0.001,
            slippage_rate=0.001,
        ),
        total_timesteps=10_000,
        action_space_version="target-position-v1",
        model_path=Path("model.zip"),
    )

    output_path = tmp_path / "metadata.json"

    save_training_run_metadata(
        metadata=metadata,
        path=output_path,
    )

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        stored = json.load(file)

    assert stored["run_name"] == "nvda_ppo_smoke"
    assert stored["algorithm"] == "PPO"
    assert stored["seed"] == 42
    assert stored["test_evaluated"] is False


def test_build_training_run_metadata_rejects_invalid_timesteps():
    with pytest.raises(
        ValueError,
        match="Total timesteps must be positive",
    ):
        build_training_run_metadata(
            run_name="invalid",
            algorithm="PPO",
            config=make_config(),
            feature_set=DEFAULT_FEATURE_SET,
            costs=ExecutionCosts(),
            total_timesteps=0,
            action_space_version="target-position-v1",
            model_path=Path("model.zip"),
        )