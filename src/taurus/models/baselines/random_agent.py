import random

from taurus.simulation.actions import TradingAction


class RandomAgent:
    # Baseline agent that selects a random trading action

    def predict(self) -> TradingAction:
        return random.choice(list(TradingAction))