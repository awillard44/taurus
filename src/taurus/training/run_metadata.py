import json
import platform
import subprocess
import warnings

from importlib.metadata import version
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from taurus.features.config import FeatureSetConfig
from taurus.environment.observation import ObservationVersion
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
    observation_version: ObservationVersion = "initial-capital-v1"

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
    observation_version: ObservationVersion = "initial-capital-v1",
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

    if observation_version not in (
        "initial-capital-v1",
        "allocation-v2",
    ):
        raise ValueError(
            f"Unsupported observation version: {observation_version}"
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
        observation_version=observation_version,
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

def load_observation_version(
    metadata_path: Path,
) -> ObservationVersion:
    try:
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
    except FileNotFoundError:
        warnings.warn(
            f"No metadata found at {metadata_path}; "
            "assuming legacy initial-capital-v1 observations.",
            UserWarning,
            stacklevel=2,
        )
        return "initial-capital-v1"

    if not isinstance(metadata, dict):
        raise ValueError(
            "Run metadata must be a JSON object."
        )

    if "observation_version" not in metadata:
        warnings.warn(
            f"No observation version recorded in {metadata_path}; "
            "assuming legacy initial-capital-v1 observations.",
            UserWarning,
            stacklevel=2,
        )
        return "initial-capital-v1"

    observation_version = metadata["observation_version"]

    if observation_version == "initial-capital-v1":
        return "initial-capital-v1"

    if observation_version == "allocation-v2":
        return "allocation-v2"

    raise ValueError(
        f"Unsupported observation version: {observation_version}"
    )