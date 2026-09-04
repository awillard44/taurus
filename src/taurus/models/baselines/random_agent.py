import random

from taurus.environment.state import EnvironmentState
from taurus.simulation.actions import TradingAction


class RandomAgent:
    def __init__(
        self,
        seed: int | None = None,
    ):
        self._random = random.Random(seed)

    def predict(
        self,
        state: EnvironmentState,
    ) -> TradingAction:
        return self._random.choice(
            list(TradingAction)
        )