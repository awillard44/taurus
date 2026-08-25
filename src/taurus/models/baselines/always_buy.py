from taurus.simulation.actions import TradingAction
from taurus.simulation.portfolio import PortfolioState


class AlwaysBuyAgent:
    # Baseline agent that buys whenever cash is available

    def predict(
        self,
        portfolio: PortfolioState,
    ) -> TradingAction:
        if portfolio.cash > 0:
            return TradingAction.BUY

        return TradingAction.HOLD