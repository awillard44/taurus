from dataclasses import dataclass

from taurus.features.market_state import MarketState
from taurus.simulation.portfolio import PortfolioState


@dataclass(frozen=True)
class EnvironmentState:
    market: MarketState
    portfolio: PortfolioState
    step_index: int