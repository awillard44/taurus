from pathlib import Path

from taurus.data.schemas import BarInterval
from taurus.data.sqlite_repository import SQLitePriceBarRepository
from taurus.environment.trading_environment import TaurusTradingEnvironment
from taurus.evaluation.baseline_runner import compare_baselines, run_agent_episodes
from taurus.environment.feature_state_builder import (
    build_feature_state_sequence,
)
from taurus.features.presets import DEFAULT_FEATURE_SET
from taurus.models.baselines.always_buy import AlwaysBuyAgent
from taurus.models.baselines.always_hold import AlwaysHoldAgent
from taurus.models.baselines.momentum import MomentumAgent
from taurus.models.baselines.random_agent import RandomAgent
from taurus.simulation.portfolio import PortfolioState
from taurus.simulation.costs import ExecutionCosts


database_path = Path("data/taurus.db")

repository = SQLitePriceBarRepository(database_path)

nvda_bars = repository.get_bars(
    "NVDA",
    BarInterval.ONE_DAY,
)

spy_bars = repository.get_bars(
    "SPY",
    BarInterval.ONE_DAY,
)

feature_states = build_feature_state_sequence(
    bars=nvda_bars,
    benchmark_bars=spy_bars,
    feature_set=DEFAULT_FEATURE_SET,
    minimum_history=50,
)

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

for name, result in results.items():
    print(
        f"{name}: "
        f"${result.final_portfolio_value:.2f} "
        f"| reward={result.total_reward:.4f} "
        f"| steps={result.steps}"
    )

random_result = run_agent_episodes(
    environment=environment,
    agent=RandomAgent(),
    runs=100,
)

print(
    "random: "
    f"mean=${random_result.mean_final_portfolio_value:.2f} "
    f"| mean reward={random_result.mean_total_reward:.4f} "
    f"| reward stdev={random_result.reward_stdev:.4f} "
    f"| runs={random_result.runs}"
)