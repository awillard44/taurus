import argparse
from pathlib import Path
from statistics import mean

from stable_baselines3 import PPO

from validate_nvda_smoke import resolve_model_path

from taurus.data.sqlite_repository import SQLitePriceBarRepository
from taurus.evaluation.baseline_runner import run_agent_episode
from taurus.evaluation.metrics import calculate_max_drawdown
from taurus.features.presets import DEFAULT_FEATURE_SET
from taurus.models.baselines.always_buy import AlwaysBuyAgent
from taurus.simulation.costs import ExecutionCosts
from taurus.training.environment_builder import (
    build_experiment_environment,
)
from taurus.training.ppo_evaluator import (
    check_model_observation_compatibility,
    evaluate_ppo_environment,
)
from taurus.training.presets import NVDA_INITIAL_EXPERIMENT
from taurus.training.run_metadata import load_observation_version
from taurus.training.training_environment import (
    build_training_environment,
)
from taurus.training.validation_environment import (
    build_validation_environment,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare saved NVDA models on training and validation."
    )
    parser.add_argument(
        "--run-names",
        nargs="+",
        required=True,
        help="Run directory names under artifacts/training.",
    )
    return parser.parse_args(argv)


def summarize_ppo(result):
    records = result.step_records

    if not records:
        raise ValueError("Cannot summarize an empty evaluation.")

    portfolio_values = [
        record.portfolio_value
        for record in records
    ]
    portfolio_values.append(result.final_portfolio_value)

    return {
        "final_value": result.final_portfolio_value,
        "return": result.total_return,
        "drawdown": calculate_max_drawdown(portfolio_values),
        "long_probability": mean(
            record.long_probability for record in records
        ),
        "entropy": mean(
            record.policy_entropy for record in records
        ),
        "exposure": mean(
            record.exposure_ratio for record in records
        ),
        "transitions": len(result.transitions),
    }


def main(argv=None):
    args = parse_args(argv)

    runs = []
    for run_name in dict.fromkeys(args.run_names):
        model_path = resolve_model_path(run_name)
        observation_version = load_observation_version(
            model_path.with_name("metadata.json")
        )
        model = PPO.load(model_path)
        runs.append((run_name, observation_version, model))

    repository = SQLitePriceBarRepository(Path("data/taurus.db"))
    config = NVDA_INITIAL_EXPERIMENT
    costs = ExecutionCosts(
        commission_rate=0.001,
        slippage_rate=0.001,
    )

    settings = {
        "repository": repository,
        "config": config,
        "feature_set": DEFAULT_FEATURE_SET,
        "initial_cash": 1000.0,
        "costs": costs,
    }

    name_width = max(
        len("always_buy"),
        *(len(name) for name, _, _ in runs),
    )

    print("\nObservation versions")
    for name, observation_version, _ in runs:
        print(f"{name}: {observation_version}")

    print("\nEach evaluation starts with $1,000.")
    print("Commission=0.1%; adverse slippage=0.1%.")

    periods = (
        ("training", config.training, build_training_environment),
        ("validation", config.validation, build_validation_environment),
    )

    for split, dates, builder in periods:
        baseline_environment = build_experiment_environment(
            **settings,
            split=split,
        )
        baseline = run_agent_episode(
            environment=baseline_environment,
            agent=AlwaysBuyAgent(),
        )

        summaries = []
        for name, observation_version, model in runs:
            wrapped = builder(
                **settings,
                observation_version=observation_version,
            )

            check_model_observation_compatibility(
                model=model,
                environment=wrapped.environment,
            )

            result = evaluate_ppo_environment(
                model=model,
                environment=wrapped.environment,
            )

            if result.steps != baseline.steps:
                raise ValueError(
                    f"Step count mismatch for {name} on {split}."
                )

            summaries.append((name, summarize_ppo(result)))

        print(
            f"\n{split.upper()}: {dates.start} -> {dates.end}"
            f" | {baseline.steps} steps"
        )
        print(
            f"{'Run':<{name_width}}"
            f" {'Final $':>10}"
            f" {'Return':>9}"
            f" {'Max DD':>9}"
            f" {'Vs buy $':>10}"
        )

        print(
            f"{'always_buy':<{name_width}}"
            f" {baseline.final_portfolio_value:>10.2f}"
            f" {baseline.total_return:>9.2%}"
            f" {baseline.max_drawdown:>9.2%}"
            f" {0.0:>10.2f}"
        )

        for name, summary in summaries:
            difference = (
                summary["final_value"] - baseline.final_portfolio_value
            )
            print(
                f"{name:<{name_width}}"
                f" {summary['final_value']:>10.2f}"
                f" {summary['return']:>9.2%}"
                f" {summary['drawdown']:>9.2%}"
                f" {difference:>+10.2f}"
            )

        print(
            f"\n{'Run':<{name_width}}"
            f" {'Mean P(LONG)':>13}"
            f" {'Entropy':>9}"
            f" {'Exposure':>10}"
            f" {'Switches':>9}"
        )

        for name, summary in summaries:
            print(
                f"{name:<{name_width}}"
                f" {summary['long_probability']:>13.2%}"
                f" {summary['entropy']:>9.3f}"
                f" {summary['exposure']:>10.2%}"
                f" {summary['transitions']:>9}"
            )


if __name__ == "__main__":
    main()