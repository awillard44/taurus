import gymnasium as gym

from taurus.environment.trading_environment import (
    TaurusTradingEnvironment,
)
from taurus.simulation.actions import TradingAction


class TargetPositionEnvironment(gym.Wrapper):
    def __init__(
        self,
        environment: TaurusTradingEnvironment,
    ):
        super().__init__(environment)

        self._taurus_environment = environment

        self.action_space = gym.spaces.Discrete(2)

    @property
    def taurus_environment(
        self,
    ) -> TaurusTradingEnvironment:
        return self._taurus_environment

    def step(self, action):
        portfolio = self._taurus_environment.state.portfolio

        if action == 0:
            if portfolio.shares > 0:
                trading_action = TradingAction.SELL
            else:
                trading_action = TradingAction.HOLD

        elif action == 1:
            if portfolio.shares <= 0:
                trading_action = TradingAction.BUY
            else:
                trading_action = TradingAction.HOLD

        else:
            raise ValueError(
                f"Unsupported target position: {action}"
            )

        return self._taurus_environment.step(
            trading_action
        )