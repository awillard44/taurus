from dataclasses import dataclass
from datetime import datetime
from math import log

from stable_baselines3 import PPO

from taurus.environment.normalization import (
    normalize_market_features,
)
from taurus.training.validation_environment import (
    ValidationEnvironment,
)


@dataclass(frozen=True)
class PPOStepRecord:
    timestamp: datetime
    asset_price: float
    target: int
    target_changed: bool
    cash_probability: float
    long_probability: float
    confidence_margin: float
    policy_entropy: float
    portfolio_cash: float
    portfolio_shares: float
    portfolio_value: float
    exposure_ratio: float
    reward: float
    feature_values: tuple[tuple[str, float], ...]
    normalized_feature_values: tuple[tuple[str, float], ...]


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
    step_records: tuple[PPOStepRecord, ...]


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


def _calculate_policy_entropy(
    cash_probability: float,
    long_probability: float,
) -> float:
    probabilities = (
        cash_probability,
        long_probability,
    )

    return -sum(
        probability * log(probability)
        for probability in probabilities
        if probability > 0.0
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
    step_records = []
    previous_target = None
    cash_count = 0
    long_count = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        current_feature_state = (
            taurus_environment.feature_states[
                taurus_environment.current_index
            ]
        )

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

        target_changed = (
            previous_target is not None
            and action_value != previous_target
        )

        normalized_features = normalize_market_features(
            features=current_feature_state.features,
            current_price=current_feature_state.market.close,
        )

        portfolio = taurus_environment.state.portfolio

        if portfolio.portfolio_value > 0:
            exposure_ratio = (
                portfolio.shares
                * current_feature_state.market.close
                / portfolio.portfolio_value
            )
        else:
            exposure_ratio = 0.0

        confidence_margin = abs(
            cash_probability - long_probability
        )

        policy_entropy = _calculate_policy_entropy(
            cash_probability=cash_probability,
            long_probability=long_probability,
        )

        if target_changed:
            transitions.append(
                PPOPositionTransition(
                    timestamp=current_feature_state.market.timestamp,
                    previous_target=previous_target,
                    new_target=action_value,
                    portfolio_value=portfolio.portfolio_value,
                    asset_price=current_feature_state.market.close,
                    cash_probability=cash_probability,
                    long_probability=long_probability,
                    feature_values=tuple(
                        (
                            key,
                            float(value),
                        )
                        for key, value
                        in current_feature_state.features.items()
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

        reward_value = float(reward)

        step_records.append(
            PPOStepRecord(
                timestamp=current_feature_state.market.timestamp,
                asset_price=current_feature_state.market.close,
                target=action_value,
                target_changed=target_changed,
                cash_probability=cash_probability,
                long_probability=long_probability,
                confidence_margin=confidence_margin,
                policy_entropy=policy_entropy,
                portfolio_cash=portfolio.cash,
                portfolio_shares=portfolio.shares,
                portfolio_value=portfolio.portfolio_value,
                exposure_ratio=exposure_ratio,
                reward=reward_value,
                feature_values=tuple(
                    (
                        key,
                        float(value),
                    )
                    for key, value
                    in current_feature_state.features.items()
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
        total_reward += reward_value
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
        step_records=tuple(step_records),
    )