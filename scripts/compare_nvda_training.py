from pathlib import Path

from stable_baselines3 import PPO

from validate_nvda_smoke import parse_args, resolve_model_path

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
    evaluate_ppo_environment,
    compare_portfolio_state_probabilities,
    check_model_observation_compatibility,
)
from taurus.training.presets import NVDA_INITIAL_EXPERIMENT
from taurus.training.training_environment import (
    build_training_environment,
)
from taurus.training.run_metadata import load_observation_version


def main(argv=None) -> None:
    args = parse_args(argv)
    model_path = resolve_model_path(args.run_name)

    observation_version = load_observation_version(
        model_path.with_name("metadata.json")
    )
    print(f"Observation version: {observation_version}")

    repository = SQLitePriceBarRepository(
        Path("data/taurus.db")
    )

    config = NVDA_INITIAL_EXPERIMENT
    costs = ExecutionCosts(
        commission_rate=0.001,
        slippage_rate=0.001,
    )

    training_environment = build_training_environment(
        repository=repository,
        config=config,
        feature_set=DEFAULT_FEATURE_SET,
        costs=costs,
    )

    model = PPO.load(model_path)

    check_model_observation_compatibility(
        model=model,
        environment=training_environment.environment,
    )

    ppo_result = evaluate_ppo_environment(
        model=model,
        environment=training_environment.environment,
    )

    baseline_environment = build_experiment_environment(
        repository=repository,
        config=config,
        split="training",
        feature_set=DEFAULT_FEATURE_SET,
        costs=costs,
    )

    baseline_result = run_agent_episode(
        environment=baseline_environment,
        agent=AlwaysBuyAgent(),
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

    print(f"\nRun: {args.run_name}")
    print(
        f"Training-period evaluation: "
        f"{config.training.start} -> {config.training.end}"
    )
    print("Saved policy evaluated without further learning.")

    print(
        f"\nPPO: final=${ppo_result.final_portfolio_value:.2f}"
        f" | return={ppo_result.total_return:.2%}"
        f" | reward={ppo_result.total_reward:.4f}"
        f" | steps={ppo_result.steps}"
    )

    print(
        f"Always-buy: final=${baseline_result.final_portfolio_value:.2f}"
        f" | return={baseline_result.total_return:.2%}"
        f" | reward={baseline_result.total_reward:.4f}"
        f" | steps={baseline_result.steps}"
    )

    print(
        f"\nTargets: CASH={ppo_result.cash_count}"
        f" | LONG={ppo_result.long_count}"
    )
    print(
        f"Average LONG probability="
        f"{sum(r.long_probability for r in records) / len(records):.3f}"
    )
    print(
        f"Average policy entropy="
        f"{sum(r.policy_entropy for r in records) / len(records):.3f}"
    )
    print(
        f"Low-confidence steps="
        f"{sum(r.confidence_margin < 0.10 for r in records)}"
        f"/{len(records)}"
    )
    print(
        f"Average market exposure="
        f"{sum(r.exposure_ratio for r in records) / len(records):.3f}"
    )
    print(f"Target transitions={len(ppo_result.transitions)}")

    print(
        f"Always-buy maximum drawdown: "
        f"{baseline_result.max_drawdown:.2%}"
    )

    print("\nTarget transition dates")
    for transition in ppo_result.transitions:
        previous = (
            "CASH" if transition.previous_target == 0 else "LONG"
        )
        new = (
            "CASH" if transition.new_target == 0 else "LONG"
        )
        print(
            f"{transition.timestamp.date()}: "
            f"{previous} -> {new}"   
        )

    print("\nPortfolio-state sensitivity")
    print("Same market and account value; different holdings.")

    sample_indices = sorted({
        0,
        len(records) // 4,
        len(records) // 2,
        3 * len(records) // 4,
        len(records) - 1,
    })

    for index in sample_indices:
        record = records[index]

        cash_state_long, long_state_long = (
            compare_portfolio_state_probabilities(
                model=model,
                record=record,
                initial_portfolio_value=(
                    ppo_result.initial_portfolio_value
                ),
            )
        )

        print(
            f"{record.timestamp.date()}"
            f" | P(LONG) when CASH={cash_state_long:.4f}"
            f" | P(LONG) when LONG={long_state_long:.4f}"
            f" | difference={long_state_long - cash_state_long:+.4f}"
        )

    print("\nAccount-value sensitivity")
    print("Initial-capital reference stays fixed at $1,000.")

    for index in (0, len(records) // 2, len(records) - 1):
        record = records[index]

        print(f"\nFixed market date: {record.timestamp.date()}")

        for account_value in (500.0, 1000.0, 2000.0, 4000.0):
            cash_state_long, long_state_long = (
                compare_portfolio_state_probabilities(
                    model=model,
                    record=record,
                    initial_portfolio_value=1000.0,
                    account_value=account_value,
                    observation_version=observation_version,
                )
            )

            print(
                f"Account=${account_value:,.0f}"
                f" | P(LONG) when CASH={cash_state_long:.4f}"
                f" | P(LONG) when LONG={long_state_long:.4f}"
            )


if __name__ == "__main__":
    main()