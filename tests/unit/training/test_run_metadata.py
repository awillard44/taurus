import json
import pytest
import subprocess

from datetime import date
from pathlib import Path
from unittest.mock import Mock

import taurus.training.run_metadata as run_metadata

from taurus.features.presets import DEFAULT_FEATURE_SET
from taurus.simulation.costs import ExecutionCosts
from taurus.training.experiment_config import (
    DateRange,
    TrainingExperimentConfig,
)
from taurus.training.run_metadata import (
    build_training_run_metadata,
    load_observation_version,
    save_training_run_metadata,
)


@pytest.fixture
def reproducibility_values(monkeypatch):
    monkeypatch.setattr(
        run_metadata.platform,
        "python_version",
        lambda: "3.11.9",
    )
    monkeypatch.setattr(
        run_metadata,
        "version",
        lambda package: "2.7.0",
    )
    monkeypatch.setattr(
        run_metadata,
        "get_git_commit",
        lambda: "a" * 40,
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


def test_build_training_run_metadata_records_experiment(
        reproducibility_values,
):
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
    assert metadata.python_version == "3.11.9"
    assert metadata.stable_baselines3_version == "2.7.0"
    assert metadata.git_commit == "a" * 40
    assert metadata.observation_version == "initial-capital-v1"

@pytest.mark.parametrize(
    "observation_version",
    ["initial-capital-v1", "allocation-v2"],
)
def test_save_training_run_metadata_writes_json(
    tmp_path: Path,
    reproducibility_values,
    observation_version,
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
        observation_version=observation_version,
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
    assert stored["python_version"] == "3.11.9"
    assert stored["stable_baselines3_version"] == "2.7.0"
    assert stored["git_commit"] == "a" * 40
    assert metadata.observation_version == observation_version
    assert stored["observation_version"] == observation_version


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

def test_get_git_commit_uses_repository_root(monkeypatch):
    commit = "b" * 40
    run = Mock(
        return_value=subprocess.CompletedProcess(
            args=["git", "rev-parse", "--verify", "HEAD"],
            returncode=0,
            stdout=f"{commit}\n",
            stderr="",
        )
    )
    monkeypatch.setattr(run_metadata.subprocess, "run", run)

    assert run_metadata.get_git_commit() == commit

    run.assert_called_once_with(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=Path(run_metadata.__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=True,
    )


def test_get_git_commit_propagates_failure(monkeypatch):
    run = Mock(
        side_effect=subprocess.CalledProcessError(
            returncode=128,
            cmd=["git", "rev-parse", "--verify", "HEAD"],
        )
    )
    monkeypatch.setattr(run_metadata.subprocess, "run", run)

    with pytest.raises(subprocess.CalledProcessError):
        run_metadata.get_git_commit()

def test_metadata_rejects_unknown_observation_version(
    reproducibility_values,
):
    with pytest.raises(
        ValueError,
        match="Unsupported observation version",
    ):
        build_training_run_metadata(
            run_name="invalid_version",
            algorithm="PPO",
            config=make_config(),
            feature_set=DEFAULT_FEATURE_SET,
            costs=ExecutionCosts(),
            total_timesteps=100_000,
            action_space_version="target-position-v1",
            model_path=Path("model.zip"),
            observation_version="unknown",
        )

@pytest.mark.parametrize(
    "observation_version",
    ["initial-capital-v1", "allocation-v2"],
)
def test_load_observation_version_reads_explicit_version(
    tmp_path,
    observation_version,
):
    path = tmp_path / "metadata.json"
    path.write_text(
        json.dumps({"observation_version": observation_version}),
        encoding="utf-8",
    )

    assert load_observation_version(path) == observation_version


def test_load_observation_version_handles_missing_file(tmp_path):
    path = tmp_path / "metadata.json"

    with pytest.warns(UserWarning, match="No metadata found"):
        result = load_observation_version(path)

    assert result == "initial-capital-v1"
    assert not path.exists()


def test_load_observation_version_handles_legacy_metadata(tmp_path):
    path = tmp_path / "metadata.json"
    original = json.dumps({"run_name": "legacy_run"})
    path.write_text(original, encoding="utf-8")

    with pytest.warns(
        UserWarning,
        match="No observation version recorded",
    ):
        result = load_observation_version(path)

    assert result == "initial-capital-v1"
    assert path.read_text(encoding="utf-8") == original


@pytest.mark.parametrize(
    "invalid_version",
    ["unknown", None, 2, ["allocation-v2"]],
)
def test_load_observation_version_rejects_invalid_version(
    tmp_path,
    invalid_version,
):
    path = tmp_path / "metadata.json"
    path.write_text(
        json.dumps({"observation_version": invalid_version}),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported observation version",
    ):
        load_observation_version(path)


def test_load_observation_version_rejects_malformed_json(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text("{broken", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        load_observation_version(path)


@pytest.mark.parametrize("metadata", [[], None, "allocation-v2"])
def test_load_observation_version_requires_json_object(
    tmp_path,
    metadata,
):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Run metadata must be a JSON object",
    ):
        load_observation_version(path)