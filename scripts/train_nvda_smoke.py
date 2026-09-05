from pathlib import Path

from taurus.data.sqlite_repository import SQLitePriceBarRepository
from taurus.features.presets import DEFAULT_FEATURE_SET
from taurus.simulation.costs import ExecutionCosts
from taurus.training.run_metadata import (
    build_training_run_metadata,
    save_training_run_metadata,
)
from taurus.training.ppo_trainer import train_ppo
from taurus.training.presets import NVDA_INITIAL_EXPERIMENT
from taurus.training.training_environment import (
    build_training_environment,
)


DATABASE_PATH = Path("data/taurus.db")

RUN_NAME = "nvda_ppo_100k"
TOTAL_TIMESTEPS = 100_000

ARTIFACT_DIRECTORY = (
    Path("artifacts")
    / "training"
    / RUN_NAME
)

MODEL_SAVE_PATH = (
    ARTIFACT_DIRECTORY
    / "model"
)

MODEL_ARTIFACT_PATH = (
    ARTIFACT_DIRECTORY
    / "model.zip"
)

METADATA_PATH = (
    ARTIFACT_DIRECTORY
    / "metadata.json"
)

FEATURE_SET = DEFAULT_FEATURE_SET

EXECUTION_COSTS = ExecutionCosts(
    commission_rate=0.001,
    slippage_rate=0.001,
)

ACTION_SPACE_VERSION = "target-position-v1"


def ensure_artifact_directory_available() -> None:
    if MODEL_ARTIFACT_PATH.exists():
        raise FileExistsError(
            f"Model artifact already exists: "
            f"{MODEL_ARTIFACT_PATH}"
        )

    if METADATA_PATH.exists():
        raise FileExistsError(
            f"Metadata artifact already exists: "
            f"{METADATA_PATH}"
        )

    ARTIFACT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )


def main() -> None:
    ensure_artifact_directory_available()

    repository = SQLitePriceBarRepository(
        DATABASE_PATH
    )

    training_environment = (
        build_training_environment(
            repository=repository,
            config=NVDA_INITIAL_EXPERIMENT,
            feature_set=FEATURE_SET,
            costs=EXECUTION_COSTS,
        )
    )

    print(
        f"Training {RUN_NAME}"
    )

    print(
        f"symbol="
        f"{NVDA_INITIAL_EXPERIMENT.symbol}"
    )

    print(
        f"training="
        f"{NVDA_INITIAL_EXPERIMENT.training.start} "
        f"-> "
        f"{NVDA_INITIAL_EXPERIMENT.training.end}"
    )

    print(
        f"timesteps={TOTAL_TIMESTEPS:,}"
    )

    print(
        f"seed="
        f"{NVDA_INITIAL_EXPERIMENT.seed}"
    )

    print(
        f"commission="
        f"{EXECUTION_COSTS.commission_rate:.4f}"
    )

    print(
        f"slippage="
        f"{EXECUTION_COSTS.slippage_rate:.4f}"
    )

    model = train_ppo(
        training_environment=training_environment,
        total_timesteps=TOTAL_TIMESTEPS,
        seed=NVDA_INITIAL_EXPERIMENT.seed,
    )

    model.save(
        MODEL_SAVE_PATH
    )

    metadata = build_training_run_metadata(
        run_name=RUN_NAME,
        algorithm="PPO",
        config=NVDA_INITIAL_EXPERIMENT,
        feature_set=FEATURE_SET,
        costs=EXECUTION_COSTS,
        total_timesteps=TOTAL_TIMESTEPS,
        action_space_version=ACTION_SPACE_VERSION,
        model_path=MODEL_ARTIFACT_PATH,
    )

    save_training_run_metadata(
        metadata=metadata,
        path=METADATA_PATH,
    )

    print(
        f"\nModel saved to "
        f"{MODEL_ARTIFACT_PATH}"
    )

    print(
        f"Metadata saved to "
        f"{METADATA_PATH}"
    )


if __name__ == "__main__":
    main()