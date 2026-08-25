from datetime import datetime, timedelta

from taurus.data.schemas import BarInterval
from taurus.environment.trading_environment import TaurusTradingEnvironment
from taurus.features.market_state import MarketState
from taurus.models.baselines.always_buy import AlwaysBuyAgent
from taurus.simulation.portfolio import PortfolioState
from taurus.evaluation.baseline_runner import (
    compare_baselines,
    run_agent_episode,
    run_agent_episodes,
)
from taurus.models.baselines.always_hold import AlwaysHoldAgent


def build_market_state(
    *,
    timestamp: datetime,
    close: float,
) -> MarketState:
    return MarketState(
        symbol="NVDA",
        timestamp=timestamp,
        interval=BarInterval.ONE_DAY,
        close=close,
        volume=1_000_000,
        return_1=0.0,
        return_5=0.0,
        return_20=0.0,
        volatility=0.02,
        sma_20=close,
        sma_50=close,
        rsi_14=50.0,
        volume_ratio=1.0,
        relative_return=0.0,
    )


def test_run_agent_episode_completes_episode():
    start = datetime(2026, 8, 25)

    market_states = [
        build_market_state(
            timestamp=start,
            close=100.0,
        ),
        build_market_state(
            timestamp=start + timedelta(days=1),
            close=110.0,
        ),
        build_market_state(
            timestamp=start + timedelta(days=2),
            close=120.0,
        ),
    ]

    initial_portfolio = PortfolioState(
        cash=1_000.0,
        shares=0.0,
        asset_price=100.0,
        portfolio_value=1_000.0,
    )

    environment = TaurusTradingEnvironment(
        market_states=market_states,
        initial_portfolio=initial_portfolio,
    )

    agent = AlwaysBuyAgent()

    result = run_agent_episode(
        environment=environment,
        agent=agent,
    )

    assert result.steps == 2
    assert result.final_portfolio_value == 1_200.0
    assert result.total_reward > 0


def test_compare_baselines_runs_each_agent():
    start = datetime(2026, 8, 25)

    market_states = [
        build_market_state(
            timestamp=start,
            close=100.0,
        ),
        build_market_state(
            timestamp=start + timedelta(days=1),
            close=110.0,
        ),
        build_market_state(
            timestamp=start + timedelta(days=2),
            close=120.0,
        ),
    ]

    initial_portfolio = PortfolioState(
        cash=1_000.0,
        shares=0.0,
        asset_price=100.0,
        portfolio_value=1_000.0,
    )

    environment = TaurusTradingEnvironment(
        market_states=market_states,
        initial_portfolio=initial_portfolio,
    )

    agents = {
        "always_hold": AlwaysHoldAgent(),
        "always_buy": AlwaysBuyAgent(),
    }

    results = compare_baselines(
        environment=environment,
        agents=agents,
    )

    assert set(results.keys()) == {
        "always_hold",
        "always_buy",
    }

    assert results["always_hold"].final_portfolio_value == 1_000.0
    assert results["always_buy"].final_portfolio_value == 1_200.0

def test_run_agent_episodes_aggregates_results():
    start = datetime(2026, 8, 25)

    market_states = [
        build_market_state(
            timestamp=start,
            close=100.0,
        ),
        build_market_state(
            timestamp=start + timedelta(days=1),
            close=110.0,
        ),
        build_market_state(
            timestamp=start + timedelta(days=2),
            close=120.0,
        ),
    ]

    initial_portfolio = PortfolioState(
        cash=1_000.0,
        shares=0.0,
        asset_price=100.0,
        portfolio_value=1_000.0,
    )

    environment = TaurusTradingEnvironment(
        market_states=market_states,
        initial_portfolio=initial_portfolio,
    )

    agent = AlwaysBuyAgent()

    result = run_agent_episodes(
        environment=environment,
        agent=agent,
        runs=3,
    )

    assert result.runs == 3
    assert result.mean_final_portfolio_value == 1_200.0
    assert result.mean_total_reward > 0
    assert result.reward_stdev == 0.0