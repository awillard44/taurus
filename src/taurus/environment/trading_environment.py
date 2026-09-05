import gymnasium as gym
import numpy as np

from gymnasium import spaces

from taurus.environment.feature_state import FeatureEnvironmentState
from taurus.environment.observation import (
    ObservationVersion,
    build_feature_observation,
)
from taurus.environment.state import EnvironmentState
from taurus.simulation.actions import TradingAction
from taurus.simulation.portfolio import PortfolioState
from taurus.simulation.step import step_portfolio
from taurus.simulation.costs import ExecutionCosts


class TaurusTradingEnvironment(gym.Env):
    # Gymnasium-compatible trading environment for Taurus

    metadata = {"render_modes": []}

    def __init__(
        self,
        feature_states: list[FeatureEnvironmentState],
        initial_portfolio: PortfolioState,
        costs: ExecutionCosts = ExecutionCosts(),
        observation_version: ObservationVersion = "initial-capital-v1",
    ):
        self.costs = costs
        super().__init__()

        if observation_version not in (
            "initial-capital-v1",
            "allocation-v2",
        ):
            raise ValueError(
                f"Unsupported observation version: {observation_version}"
            )

        self.observation_version = observation_version

        if len(feature_states) < 2:
            raise ValueError(
                "At least two feature states are required."
            )

        self.feature_states = feature_states
        self.initial_portfolio = initial_portfolio

        self.current_index = 0

        self.state = EnvironmentState(
            market=self.feature_states[0].market,
            portfolio=self.initial_portfolio,
            step_index=0,
        )

        self.action_space = spaces.Discrete(
            len(TradingAction)
        )

        portfolio_input_count = (
            3
            if self.observation_version == "initial-capital-v1"
            else 2
        )

        observation_size = (
            len(self.feature_states[0].features)
            + portfolio_input_count
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size,),
            dtype=np.float32,
        )

    def _get_observation(self) -> np.ndarray:
        return build_feature_observation(
            features=self.feature_states[
                self.current_index
            ].features,
            portfolio=self.state.portfolio,
            current_price=self.state.market.close,
            initial_portfolio_value=(
                self.initial_portfolio.portfolio_value
            ),
            observation_version=self.observation_version,
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
            market=self.feature_states[0].market,
            portfolio=self.initial_portfolio,
            step_index=0,
        )

        return self._get_observation(), {}

    def step(self, action: int):
        trading_action = TradingAction(action)

        current_feature_state = self.feature_states[
            self.current_index
        ]

        next_feature_state = self.feature_states[
            self.current_index + 1
        ]

        current_market = (
            current_feature_state.market
        )

        next_market = (
            next_feature_state.market
        )

        step_result = step_portfolio(
            state=self.state.portfolio,
            action=trading_action,
            next_asset_price=next_market.close,
            timestamp=current_market.timestamp,
            costs=self.costs,
        )

        self.current_index += 1

        self.state = EnvironmentState(
            market=next_market,
            portfolio=step_result.portfolio,
            step_index=self.current_index,
        )

        terminated = (
            self.current_index
            == len(self.feature_states) - 1
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