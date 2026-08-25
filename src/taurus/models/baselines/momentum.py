from taurus.environment.state import EnvironmentState
from taurus.simulation.actions import TradingAction


class MomentumAgent:
    # Baseline agent using simple recent-return momentum

    def predict(
        self,
        state: EnvironmentState,
    ) -> TradingAction:
        if state.market.return_5 > 0 and state.portfolio.cash > 0:
            return TradingAction.BUY

        if state.market.return_5 < 0 and state.portfolio.shares > 0:
            return TradingAction.SELL

        return TradingAction.HOLD