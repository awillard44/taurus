from datetime import datetime
from typing import Protocol

from taurus.data.schemas import PriceBar


class MarketDataProvider(Protocol):
    """Interface for market-data providers used by Taurus."""

    def get_daily_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[PriceBar]:
        """Return daily OHLCV bars for a symbol within a date range."""
        ...