from pathlib import Path

from taurus.data.sqlite_repository import SQLitePriceBarRepository
from taurus.training.ppo_trainer import train_ppo
from taurus.training.presets import NVDA_INITIAL_EXPERIMENT
from taurus.training.training_environment import (
    build_training_environment,
)


database_path = Path("data/taurus.db")

artifact_directory = Path(
    "artifacts/training/nvda_ppo_smoke_v2"
)

artifact_directory.mkdir(
    parents=True,
    exist_ok=True,
)

repository = SQLitePriceBarRepository(
    database_path,
)

training_environment = build_training_environment(
    repository=repository,
    config=NVDA_INITIAL_EXPERIMENT,
)

model = train_ppo(
    training_environment=training_environment,
    total_timesteps=10_000,
    seed=NVDA_INITIAL_EXPERIMENT.seed,
)

model.save(
    artifact_directory / "model"
)

print(
    f"Saved model to {artifact_directory / 'model.zip'}"
)