import gymnasium as gym
import numpy as np
from gymnasium import spaces

from taurus.environment.observation import build_observation
from taurus.environment.state import EnvironmentState
from taurus.features.market_state import MarketState
from taurus.simulation.actions import TradingAction
from taurus.simulation.portfolio import PortfolioState
from taurus.simulation.step import step_portfolio


class TaurusTradingEnvironment(gym.Env):
    # Gymnasium-compatible trading environment for Taurus

    metadata = {"render_modes": []}

    def __init__(
        self,
        market_states: list[MarketState],
        initial_portfolio: PortfolioState,
    ):
        super().__init__()

        if len(market_states) < 2:
            raise ValueError(
                "At least two market states are required."
            )

        self.market_states = market_states
        self.initial_portfolio = initial_portfolio

        self.current_index = 0

        self.state = EnvironmentState(
            market=self.market_states[0],
            portfolio=self.initial_portfolio,
            step_index=0,
        )

        self.action_space = spaces.Discrete(
            len(TradingAction)
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(13,),
            dtype=np.float32,
        )

    def _get_observation(self) -> np.ndarray:
        observation = build_observation(self.state)

        return np.array(
            observation,
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        super().reset(seed=seed)

        self.current_index = 0

        self.state = EnvironmentState(
            market=self.market_states[0],
            portfolio=self.initial_portfolio,
            step_index=0,
        )

        return self._get_observation(), {}

    def step(self, action: int):
        trading_action = TradingAction(action)

        current_market = self.market_states[
            self.current_index
        ]

        next_market = self.market_states[
            self.current_index + 1
        ]

        step_result = step_portfolio(
            state=self.state.portfolio,
            action=trading_action,
            next_asset_price=next_market.close,
            timestamp=current_market.timestamp,
        )

        self.current_index += 1

        self.state = EnvironmentState(
            market=next_market,
            portfolio=step_result.portfolio,
            step_index=self.current_index,
        )

        terminated = (
            self.current_index
            == len(self.market_states) - 1
        )

        truncated = False

        info = {
            "trade": step_result.trade,
        }

        return (
            self._get_observation(),
            step_result.reward,
            terminated,
            truncated,
            info,
        )