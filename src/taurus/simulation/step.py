from dataclasses import dataclass

from taurus.simulation.actions import TradingAction
from taurus.simulation.execution import execute_action
from taurus.simulation.portfolio import PortfolioState
from taurus.simulation.rewards import calculate_portfolio_return_reward
from taurus.simulation.valuation import revalue_portfolio


@dataclass(frozen=True)
class StepResult:
    portfolio: PortfolioState
    reward: float

def step_portfolio(
    state: PortfolioState,
    action: TradingAction,
    next_asset_price: float,
) -> StepResult:
    # Apply an action and advance the portfolio one market step

    executed_state = execute_action(
        state=state,
        action=action,
    )

    next_state = revalue_portfolio(
        state=executed_state,
        new_asset_price=next_asset_price,
    )

    reward = calculate_portfolio_return_reward(
        previous_value=state.portfolio_value,
        current_value=next_state.portfolio_value,
    )

    return StepResult(
        portfolio=next_state,
        reward=reward,
    )