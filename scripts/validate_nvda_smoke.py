import argparse

from pathlib import Path

from stable_baselines3 import PPO

from taurus.data.sqlite_repository import (
    SQLitePriceBarRepository,
)
from taurus.evaluation.baseline_runner import (
    compare_baselines,
)
from taurus.evaluation.metrics import calculate_max_drawdown
from taurus.models.baselines.always_buy import (
    AlwaysBuyAgent,
)
from taurus.models.baselines.always_hold import (
    AlwaysHoldAgent,
)
from taurus.models.baselines.momentum import (
    MomentumAgent,
)
from taurus.training.ppo_evaluator import (
    evaluate_ppo,
    check_model_observation_compatibility,
)
from taurus.training.presets import NVDA_INITIAL_EXPERIMENT
from taurus.training.validation_environment import (
    build_validation_environment,
)
from taurus.training.environment_builder import (
    build_experiment_environment,
)
from taurus.training.validation_analysis import (
    calculate_feature_policy_associations,
)
from taurus.training.run_metadata import load_observation_version


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a named NVDA training run."
    )
    parser.add_argument(
        "--run-name",
        required=True,
        help="Run directory name under artifacts/training.",
    )
    return parser.parse_args(argv)


def resolve_model_path(run_name: str) -> Path:
    if (
        not run_name.strip()
        or run_name in {".", ".."}
        or "/" in run_name
        or "\\" in run_name
    ):
        raise ValueError(
            "Run name must be a single directory name."
        )

    model_path = (
        Path("artifacts/training")
        / run_name
        / "model.zip"
    )

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}"
        )

    return model_path

def main(argv=None) -> None:
    args = parse_args(argv)
    database_path = Path("data/taurus.db")
    model_path = resolve_model_path(args.run_name)

    observation_version = load_observation_version(
        model_path.with_name("metadata.json")
    )
    print(f"Observation version: {observation_version}")

    repository = SQLitePriceBarRepository(
        database_path,
    )

    validation_environment = build_validation_environment(
        repository=repository,
        config=NVDA_INITIAL_EXPERIMENT,
        observation_version=observation_version,
    )

    model = PPO.load(
        model_path,
    )

    check_model_observation_compatibility(
        model=model,
        environment=validation_environment.environment,
    )

    ppo_result = evaluate_ppo(
        model=model,
        validation_environment=validation_environment,
    )

    print("\nNVDA PPO - 2025 validation")
    print(
        f"final=${ppo_result.final_portfolio_value:.2f} "
        f"| return={ppo_result.total_return:.4f} "
        f"| reward={ppo_result.total_reward:.4f} "
        f"| steps={ppo_result.steps}"
    )

    print(
        f"targets: "
        f"cash={ppo_result.cash_count} "
        f"| long={ppo_result.long_count}"
    )

    records = ppo_result.step_records

    portfolio_values = [
        record.portfolio_value
        for record in records
    ]
    portfolio_values.append(
        ppo_result.final_portfolio_value
    )

    ppo_max_drawdown = calculate_max_drawdown(portfolio_values)

    print(f"\nPPO maximum drawdown: {ppo_max_drawdown:.2%}")

    average_cash_probability = sum(
        record.cash_probability
        for record in records
    ) / len(records)

    average_long_probability = sum(
        record.long_probability
        for record in records
    ) / len(records)

    average_confidence_margin = sum(
        record.confidence_margin
        for record in records
    ) / len(records)

    average_entropy = sum(
        record.policy_entropy
        for record in records
    ) / len(records)

    low_confidence_steps = sum(
        1
        for record in records
        if record.confidence_margin < 0.10
    )

    average_exposure = sum(
        record.exposure_ratio
        for record in records
    ) / len(records)

    print("\nPolicy Summary")

    print(
        f"average probabilities: "
        f"CASH={average_cash_probability:.3f} "
        f"| LONG={average_long_probability:.3f}"
    )

    print(
        f"average confidence margin="
        f"{average_confidence_margin:.3f}"
    )

    print(
        f"average policy entropy="
        f"{average_entropy:.3f}"
    )

    print(
        f"low-confidence steps="
        f"{low_confidence_steps}/{len(records)}"
    )

    print(
        f"average market exposure="
        f"{average_exposure:.3f}"
    )

    associations = calculate_feature_policy_associations(
        ppo_result.step_records
    )

    print("\nFeature / Policy Associations")

    for association in associations:
        print(
            f"{association.feature}: "
            f"{association.correlation_with_long_probability:+.3f}"
        )

    print("\nTarget Transitions")

    for transition in ppo_result.transitions:
        previous = (
            "CASH"
            if transition.previous_target == 0
            else "LONG"
        )

        new = (
            "CASH"
            if transition.new_target == 0
            else "LONG"
        )

        print(
            f"\n{transition.timestamp.date()} "
            f"{previous} -> {new}"
        )

        print(
            f" portfolio=${transition.portfolio_value:.2f} "
            f"| NVDA=${transition.asset_price:.2f}"
        )

        print(
            f"  policy: "
            f"CASH={transition.cash_probability:.3f} "
            f"| LONG={transition.long_probability:.3f}"
        )

        raw_features = dict(
            transition.feature_values
        )

        normalized_features = dict(
            transition.normalized_feature_values
        )

        print("  features:")

        for key, raw_value in raw_features.items():
            normalized_value = normalized_features[key]

            print(
            f"    {key}: "
            f"raw={raw_value:.6f} "
            f"| normalized={normalized_value:.6f}"
            )

    baseline_environment = build_experiment_environment(
        repository=repository,
        config=NVDA_INITIAL_EXPERIMENT,
        split="validation",
        observation_version=observation_version,
    )

    agents = {
        "always_hold": AlwaysHoldAgent(),
        "always_buy": AlwaysBuyAgent(),
        "momentum": MomentumAgent(),
    }

    baseline_results = compare_baselines(
        environment=baseline_environment,
        agents=agents,
    )

    print("\nBaselines")

    for name, result in baseline_results.items():
        print(
            f"{name}: "
            f"${result.final_portfolio_value:.2f} "
            f"| return={result.total_return:.4f} "
            f"| reward={result.total_reward:.4f} "
            f"| max drawdown={result.max_drawdown:.2%}"
        )

if __name__ == "__main__":
    main()