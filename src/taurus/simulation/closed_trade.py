from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ClosedTrade:
    entry_timestamp: datetime
    exit_timestamp: datetime
    entry_price: float
    exit_price: float
    shares: float
    entry_value: float
    exit_value: float
    realized_pnl: float
    return_pct: float