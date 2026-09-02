from dataclasses import dataclass
from statistics import mean, stdev

from taurus.environment.trading_environment import TaurusTradingEnvironment
from taurus.evaluation.metrics import (
    calculate_average_gain,
    calculate_average_loss,
    calculate_max_drawdown,
    calculate_portfolio_volatility,
    calculate_profit_factor,
    calculate_total_return,
    calculate_trade_count,
    calculate_win_rate,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)
from taurus.simulation.trade import Trade
from taurus.simulation.closed_trade import ClosedTrade
from taurus.simulation.trade_tracker import (
    OpenPosition,
    add_shares,
    close_shares,
)
from taurus.simulation.actions import TradingAction

@dataclass(frozen=True)
class EpisodeResult:
    initial_portfolio_value: float
    final_portfolio_value: float
    total_return: float
    max_drawdown: float
    total_reward: float
    steps: int
    portfolio_values: tuple[float, ...]
    portfolio_volatility: float | None
    sharpe_ratio: float | None
    sortino_ratio: float | None
    trades: tuple[Trade, ...]
    closed_trades: tuple[ClosedTrade, ...]
    trade_count: int
    win_rate: float | None
    average_gain: float
    average_loss: float
    profit_factor: float | None
    


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

    trades = []
    closed_trades = []
    open_position: OpenPosition | None = None

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

        _, reward, terminated, truncated, info = environment.step(action)

        trade = info.get("trade")

        if trade is not None:
            trades.append(trade)

            if trade.action == TradingAction.BUY:
                open_position = add_shares(
                    position=open_position,
                    timestamp=trade.timestamp,
                    price=trade.price,
                    shares=trade.shares,
                )

            elif trade.action == TradingAction.SELL:
                if open_position is None:
                    raise ValueError("Cannot realize a SELL trade without an open position.")

                open_position, closed_trade = close_shares(
                    position=open_position,
                    timestamp=trade.timestamp,
                    price=trade.price,
                    shares=trade.shares,
                )

                closed_trades.append(closed_trade)

        portfolio_values.append(
            environment.state.portfolio.portfolio_value
        )

        total_reward += reward
        steps += 1

    final_portfolio_value = (
        environment.state.portfolio.portfolio_value
    )

    portfolio_volatility = (
        calculate_portfolio_volatility(portfolio_values)
        if len(portfolio_values) >= 3 else None
    )

    portfolio_returns = [
        (
            portfolio_values[index] - portfolio_values[index -1]
        )
        / portfolio_values[index - 1]
        for index in range(1, len(portfolio_values))
    ]

    if len(portfolio_returns) >= 2:
        try:
            sharpe_ratio = calculate_sharpe_ratio(portfolio_returns)
        except Exception:
            sharpe_ratio = None
    else:
        sharpe_ratio = None

    if len(portfolio_returns) >= 2:
        try:
            sortino_ratio = calculate_sortino_ratio(
                portfolio_returns
            )
        except ValueError:
            sortino_ratio = None
    else:
        sortino_ratio = None

    trade_count = calculate_trade_count(closed_trades)

    win_rate = (
        calculate_win_rate(closed_trades)
        if closed_trades
        else None
    )

    average_gain = calculate_average_gain(closed_trades)
    average_loss = calculate_average_loss(closed_trades)

    has_realized_loss = any(
        trade.realized_pnl < 0
        for trade in closed_trades
    )

    profit_factor = (
        calculate_profit_factor(closed_trades)
        if has_realized_loss
        else None
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
        portfolio_volatility=portfolio_volatility,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        trades=tuple(trades),
        closed_trades=tuple(closed_trades),
        trade_count=trade_count,
        win_rate=win_rate,
        average_gain=average_gain,
        average_loss=average_loss,
        profit_factor=profit_factor,
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