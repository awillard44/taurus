from taurus.simulation.actions import TradingAction


class AlwaysHoldAgent:
    # Baseline agent that always chooses HOLD

    def predict(self) -> TradingAction:
        return TradingAction.HOLD