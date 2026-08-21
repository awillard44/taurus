from dataclasses import dataclass
from datetime import datetime

from taurus.data.schemas import BarInterval


@dataclass(frozen=True)
class MarketState:
    symbol: str
    timestamp: datetime
    interval: BarInterval

    close: float
    volume: float

    return_1: float
    return_5: float
    return_20: float

    volatility: float
    sma_20: float
    sma_50: float
    rsi_14: float
    volume_ratio: float
    relative_return: float