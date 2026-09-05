from stable_baselines3 import PPO

from taurus.training.training_environment import TrainingEnvironment


def train_ppo(
    training_environment: TrainingEnvironment,
    total_timesteps: int,
    seed: int,
) -> PPO:
    if total_timesteps <= 0:
        raise ValueError("Total timesteps must be positive.")

    model = PPO(
        policy="MlpPolicy",
        env=training_environment.environment,
        seed=seed,
        verbose=1,
    )

    model.learn(
        total_timesteps=total_timesteps,
    )

    return model