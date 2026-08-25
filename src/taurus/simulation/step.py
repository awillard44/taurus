from dataclasses import dataclass
from datetime import datetime

from taurus.simulation.actions import TradingAction
from taurus.simulation.execution import execute_action
from taurus.simulation.portfolio import PortfolioState
from taurus.simulation.rewards import calculate_portfolio_return_reward
from taurus.simulation.trade import Trade
from taurus.simulation.valuation import revalue_portfolio


@dataclass(frozen=True)
class StepResult:
    portfolio: PortfolioState
    reward: float
    trade: Trade | None


def step_portfolio(
    state: PortfolioState,
    action: TradingAction,
    next_asset_price: float,
    timestamp: datetime,
) -> StepResult:
    # Apply an action, advance the market, and calculate reward

    execution_result = execute_action(
        state=state,
        action=action,
        timestamp=timestamp,
    )

    next_state = revalue_portfolio(
        state=execution_result.portfolio,
        new_asset_price=next_asset_price,
    )

    reward = calculate_portfolio_return_reward(
        previous_value=state.portfolio_value,
        current_value=next_state.portfolio_value,
    )

    return StepResult(
        portfolio=next_state,
        reward=reward,
        trade=execution_result.trade,
    )