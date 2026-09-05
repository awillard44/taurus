from dataclasses import dataclass
from datetime import datetime

from stable_baselines3 import PPO

from taurus.environment.normalization import (
    normalize_market_features,
)
from taurus.training.validation_environment import (
    ValidationEnvironment,
)


@dataclass(frozen=True)
class PPOPositionTransition:
    timestamp: datetime
    previous_target: int
    new_target: int
    portfolio_value: float
    asset_price: float
    cash_probability: float
    long_probability: float
    feature_values: tuple[tuple[str, float], ...]
    normalized_feature_values: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class PPOEvaluationResult:
    initial_portfolio_value: float
    final_portfolio_value: float
    total_return: float
    total_reward: float
    steps: int
    cash_count: int
    long_count: int
    transitions: tuple[PPOPositionTransition, ...]


def _get_target_probabilities(
    model: PPO,
    observation,
) -> tuple[float, float]:
    observation_tensor, _ = model.policy.obs_to_tensor(
        observation
    )

    distribution = model.policy.get_distribution(
        observation_tensor
    )

    probabilities = (
        distribution
        .distribution
        .probs[0]
        .detach()
        .cpu()
        .tolist()
    )

    if len(probabilities) != 2:
        raise ValueError(
            "Target-position policy must have exactly two actions."
        )

    return (
        float(probabilities[0]),
        float(probabilities[1]),
    )


def evaluate_ppo(
    model: PPO,
    validation_environment: ValidationEnvironment,
) -> PPOEvaluationResult:
    environment = validation_environment.environment
    taurus_environment = environment.taurus_environment

    observation, _ = environment.reset()

    initial_portfolio_value = (
        taurus_environment.state.portfolio.portfolio_value
    )

    total_reward = 0.0
    steps = 0
    transitions = []
    previous_target = None
    cash_count = 0
    long_count = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        cash_probability, long_probability = (
            _get_target_probabilities(
                model=model,
                observation=observation,
            )
        )

        action, _ = model.predict(
            observation,
            deterministic=True,
        )

        action_value = int(action)

        current_feature_state = (
            taurus_environment
            .feature_states[
                taurus_environment.current_index
            ]
        )

        if (
            previous_target is not None
            and action_value != previous_target
        ):
            normalized_features = normalize_market_features(
                features=current_feature_state.features,
                current_price=current_feature_state.market.close,
            )

            transitions.append(
                PPOPositionTransition(
                    timestamp=(
                        current_feature_state
                        .market
                        .timestamp
                    ),
                    previous_target=previous_target,
                    new_target=action_value,
                    portfolio_value=(
                        taurus_environment
                        .state
                        .portfolio
                        .portfolio_value
                    ),
                    asset_price=(
                        current_feature_state
                        .market
                        .close
                    ),
                    cash_probability=cash_probability,
                    long_probability=long_probability,
                    feature_values=tuple(
                        (
                            key,
                            float(value),
                        )
                        for key, value
                        in current_feature_state
                        .features
                        .items()
                    ),
                    normalized_feature_values=tuple(
                        (
                            key,
                            float(value),
                        )
                        for key, value
                        in normalized_features.items()
                    ),
                )
            )

        previous_target = action_value

        if action_value == 0:
            cash_count += 1
        elif action_value == 1:
            long_count += 1

        (
            observation,
            reward,
            terminated,
            truncated,
            _,
        ) = environment.step(action_value)

        total_reward += float(reward)
        steps += 1

    final_portfolio_value = (
        taurus_environment.state.portfolio.portfolio_value
    )

    total_return = (
        final_portfolio_value - initial_portfolio_value
    ) / initial_portfolio_value

    return PPOEvaluationResult(
        initial_portfolio_value=initial_portfolio_value,
        final_portfolio_value=final_portfolio_value,
        total_return=total_return,
        total_reward=total_reward,
        steps=steps,
        cash_count=cash_count,
        long_count=long_count,
        transitions=tuple(transitions),
    )