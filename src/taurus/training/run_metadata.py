import json
import platform
import subprocess

from importlib.metadata import version
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from taurus.features.config import FeatureSetConfig
from taurus.simulation.costs import ExecutionCosts
from taurus.training.experiment_config import (
    TrainingExperimentConfig,
)


@dataclass(frozen=True)
class TrainingRunMetadata:
    run_name: str
    algorithm: str
    symbol: str
    benchmark_symbol: str
    training_start: str
    training_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str
    total_timesteps: int
    seed: int
    feature_set: str
    commission_rate: float
    slippage_rate: float
    action_space_version: str
    model_path: str
    created_at_utc: str
    test_evaluated: bool
    python_version: str
    stable_baselines3_version: str
    git_commit: str

def get_git_commit() -> str:
    repository_root = Path(__file__).resolve().parents[3]

    result = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=True,
    )

    return result.stdout.strip()


def build_training_run_metadata(
    run_name: str,
    algorithm: str,
    config: TrainingExperimentConfig,
    feature_set: FeatureSetConfig,
    costs: ExecutionCosts,
    total_timesteps: int,
    action_space_version: str,
    model_path: Path,
) -> TrainingRunMetadata:
    if not run_name:
        raise ValueError("Run name must not be empty.")

    if not algorithm:
        raise ValueError("Algorithm must not be empty.")

    if total_timesteps <= 0:
        raise ValueError(
            "Total timesteps must be positive."
        )

    if not action_space_version:
        raise ValueError(
            "Action space version must not be empty."
        )

    return TrainingRunMetadata(
        run_name=run_name,
        algorithm=algorithm,
        symbol=config.symbol,
        benchmark_symbol=config.benchmark_symbol,
        training_start=config.training.start.isoformat(),
        training_end=config.training.end.isoformat(),
        validation_start=config.validation.start.isoformat(),
        validation_end=config.validation.end.isoformat(),
        test_start=config.test.start.isoformat(),
        test_end=config.test.end.isoformat(),
        total_timesteps=total_timesteps,
        seed=config.seed,
        feature_set=feature_set.name,
        commission_rate=costs.commission_rate,
        slippage_rate=costs.slippage_rate,
        action_space_version=action_space_version,
        model_path=str(model_path),
        created_at_utc=datetime.now(
            timezone.utc
        ).isoformat(),
        test_evaluated=False,
        python_version=platform.python_version(),
        stable_baselines3_version=version("stable-baselines3"),
        git_commit=get_git_commit(),
    )


def save_training_run_metadata(
    metadata: TrainingRunMetadata,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            asdict(metadata),
            file,
            indent=2,
            sort_keys=True,
        )

        file.write("\n")