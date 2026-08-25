import random

from taurus.environment.state import EnvironmentState
from taurus.simulation.actions import TradingAction


class RandomAgent:
    # Baseline agent that selects a random trading action

    def predict(
            self,
            state: EnvironmentState,
        ) -> TradingAction:
            return random.choice(list(TradingAction))