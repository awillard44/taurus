from dataclasses import dataclass
from datetime import datetime

from taurus.simulation.actions import TradingAction


@dataclass(frozen=True)
class Trade:
    timestamp: datetime
    action: TradingAction
    price: float
    shares: float
    value: float