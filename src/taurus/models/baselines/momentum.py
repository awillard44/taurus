from taurus.features.market_state import MarketState
from taurus.simulation.actions import TradingAction
from taurus.simulation.portfolio import PortfolioState


class MomentumAgent:
    # Baseline agent using simple recent-return momentum

    def predict(
        self,
        market: MarketState,
        portfolio: PortfolioState,
    ) -> TradingAction:
        if market.return_5 > 0 and portfolio.cash > 0:
            return TradingAction.BUY

        if market.return_5 < 0 and portfolio.shares > 0:
            return TradingAction.SELL

        return TradingAction.HOLD