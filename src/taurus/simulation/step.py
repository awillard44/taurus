from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from taurus.simulation.actions import TradingAction
from taurus.simulation.execution import execute_action
from taurus.simulation.portfolio import PortfolioState
from taurus.simulation.rewards import calculate_portfolio_return_reward
from taurus.simulation.trade import Trade
from taurus.simulation.valuation import revalue_portfolio
from taurus.simulation.costs import ExecutionCosts


ExecutionVersion = Literal[
    "same-close-v1",
    "next-open-v2",
]

@dataclass(frozen=True)
class StepResult:
    portfolio: PortfolioState
    reward: float
    trade: Trade | None


def step_portfolio(
    state: PortfolioState,
    action: TradingAction,
    next_asset_price: float,
    timestamp,
    costs: ExecutionCosts = ExecutionCosts(),
    execution_version: ExecutionVersion = "same-close-v1",
    next_open_price: float | None = None,
) -> StepResult:
    if execution_version == "same-close-v1":
        execution_state = state

    elif execution_version == "next-open-v2":
        if next_open_price is None:
            raise ValueError(
                "Next-open execution requires next_open_price."
            )

        execution_state = revalue_portfolio(
            state=state,
            new_asset_price=next_open_price,
        )

    else:
        raise ValueError(
            f"Unsupported execution version: {execution_version}"
        )

    execution_result = execute_action(
        state=execution_state,
        action=action,
        timestamp=timestamp,
        costs=costs,
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