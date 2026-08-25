from taurus.environment.state import EnvironmentState
from taurus.simulation.actions import TradingAction


class AlwaysBuyAgent:
    # Baseline agent that buys whenever cash is available

    def predict(
        self,
        state: EnvironmentState,
    ) -> TradingAction:
        if state.portfolio.cash > 0:
            return TradingAction.BUY

        return TradingAction.HOLD