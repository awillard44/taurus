from pathlib import Path
from datetime import datetime

from taurus.data.schemas import BarInterval
from taurus.data.sqlite_repository import SQLitePriceBarRepository
from taurus.environment.feature_state_builder import (
    build_feature_state_sequence,
)
from taurus.environment.trading_environment import (
    TaurusTradingEnvironment,
)
from taurus.evaluation.baseline_runner import (
    compare_baselines,
    run_agent_episodes,
)
from taurus.features.presets import DEFAULT_FEATURE_SET
from taurus.models.baselines.always_buy import AlwaysBuyAgent
from taurus.models.baselines.always_hold import AlwaysHoldAgent
from taurus.models.baselines.momentum import MomentumAgent
from taurus.models.baselines.random_agent import RandomAgent
from taurus.simulation.costs import ExecutionCosts
from taurus.simulation.portfolio import PortfolioState


database_path = Path("data/taurus.db")

repository = SQLitePriceBarRepository(database_path)

validation_windows = [
    ("2022", datetime(2022, 1, 1).date(), datetime(2022, 12, 31).date()),
    ("2023", datetime(2023, 1, 1).date(), datetime(2023, 12, 31).date()),
    ("2024", datetime(2024, 1, 1).date(), datetime(2024, 12, 31).date()),
    ("2025", datetime(2025, 1, 1).date(), datetime(2025, 12, 31).date()),
    ("2026", datetime(2026, 1, 1).date(), datetime(2026, 8, 21).date()),
]

symbols = [
    "NVDA",
    "AAPL",
    "MSFT",
    "AMZN",
]

spy_bars = repository.get_bars(
    "SPY",
    BarInterval.ONE_DAY,
)

for symbol in symbols:
    bars = repository.get_bars(
        symbol,
        BarInterval.ONE_DAY,
    )

    for window_name, start, end in validation_windows:
        warmup_start = datetime(
            start.year - 1,
            10,
            1,
        ).date()

        context_bars = [
            bar
            for bar in bars
            if warmup_start <= bar.timestamp.date() <= end
        ]

        context_spy_bars = [
            bar
            for bar in spy_bars
            if warmup_start <= bar.timestamp.date() <= end
        ]

        context_feature_states = build_feature_state_sequence(
            bars=context_bars,
            benchmark_bars=context_spy_bars,
            feature_set=DEFAULT_FEATURE_SET,
            minimum_history=50,
        )

        feature_states = [
            state
            for state in context_feature_states
            if start <= state.market.timestamp.date() <= end
        ]

        initial_portfolio = PortfolioState(
            cash=1_000.0,
            shares=0.0,
            asset_price=feature_states[0].market.close,
            portfolio_value=1_000.0,
        )

        execution_costs = ExecutionCosts(
            commission_rate=0.001,
            slippage_rate=0.001,
        )

        environment = TaurusTradingEnvironment(
            feature_states=feature_states,
            initial_portfolio=initial_portfolio,
            costs=execution_costs,
        )

        agents = {
            "always_hold": AlwaysHoldAgent(),
            "always_buy": AlwaysBuyAgent(),
            "momentum": MomentumAgent(),
        }

        results = compare_baselines(
            environment=environment,
            agents=agents,
        )

        print(f"\n{symbol} - {window_name}")

        for name, result in results.items():
            print(
                f"{name}: "
                f"${result.final_portfolio_value:.2f} "
                f"| return={result.total_return:.4f} "
                f"| volatility={result.portfolio_volatility} "
                f"| sharpe={result.sharpe_ratio} "
                f"| sortino={result.sortino_ratio} "
                f"| steps={result.steps}"
            )

        random_result = run_agent_episodes(
            environment=environment,
            agent=RandomAgent(seed=42),
            runs=100,
        )

        print(
            "random: "
            f"mean=${random_result.mean_final_portfolio_value:.2f} "
            f"| mean reward={random_result.mean_total_reward:.4f} "
            f"| reward stdev={random_result.reward_stdev:.4f} "
            f"| runs={random_result.runs}"
        )