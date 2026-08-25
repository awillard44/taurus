from dataclasses import dataclass
from statistics import mean, stdev

from taurus.environment.trading_environment import TaurusTradingEnvironment
from taurus.evaluation.metrics import (
    calculate_max_drawdown,
    calculate_total_return,
)

@dataclass(frozen=True)
class EpisodeResult:
    initial_portfolio_value: float
    final_portfolio_value: float
    total_return: float
    max_drawdown: float
    total_reward: float
    steps: int
    portfolio_values: tuple[float, ...]


@dataclass(frozen=True)
class AggregateEpisodeResult:
    mean_final_portfolio_value: float
    mean_total_reward: float
    reward_stdev: float
    runs: int


def run_agent_episode(
    environment: TaurusTradingEnvironment,
    agent,
) -> EpisodeResult:
    # Run one complete trading episode with an agent

    environment.reset()

    initial_portfolio_value = (
        environment.state.portfolio.portfolio_value
    )

    portfolio_values = [initial_portfolio_value]

    total_reward = 0.0
    steps = 0
    terminated = False
    truncated = False

    while not terminated and not truncated:
        action = agent.predict(environment.state)

        _, reward, terminated, truncated, _ = environment.step(action)

        portfolio_values.append(
            environment.state.portfolio.portfolio_value
        )

        total_reward += reward
        steps += 1

    final_portfolio_value = (
        environment.state.portfolio.portfolio_value
    )

    return EpisodeResult(
        initial_portfolio_value=initial_portfolio_value,
        final_portfolio_value=final_portfolio_value,
        total_return=calculate_total_return(
            initial_portfolio_value,
            final_portfolio_value,
        ),
        max_drawdown=calculate_max_drawdown(
            portfolio_values
        ),
        total_reward=total_reward,
        steps=steps,
        portfolio_values=tuple(portfolio_values),
    )


def run_agent_episodes(
    environment: TaurusTradingEnvironment,
    agent,
    runs: int,
) -> AggregateEpisodeResult:
    # Run an agent across multiple episodes and aggregate the results

    results = [
        run_agent_episode(
            environment=environment,
            agent=agent,
        )
        for _ in range(runs)
    ]

    rewards = [result.total_reward for result in results]
    final_values = [
        result.final_portfolio_value
        for result in results
    ]

    return AggregateEpisodeResult(
        mean_final_portfolio_value=mean(final_values),
        mean_total_reward=mean(rewards),
        reward_stdev=stdev(rewards) if runs > 1 else 0.0,
        runs=runs,
    )


def compare_baselines(
    environment: TaurusTradingEnvironment,
    agents: dict[str, object],
) -> dict[str, EpisodeResult]:
    # Run multiple baseline agents against the same environment

    results = {}

    for name, agent in agents.items():
        results[name] = run_agent_episode(
            environment=environment,
            agent=agent,
        )

    return results