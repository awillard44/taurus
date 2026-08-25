from taurus.environment.state import EnvironmentState
from taurus.simulation.actions import TradingAction


class AlwaysHoldAgent:
    # Baseline agent that always chooses HOLD

    def predict(
            self,
            state: EnvironmentState,
        ) -> TradingAction:
            return TradingAction.HOLD