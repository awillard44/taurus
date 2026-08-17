from datetime import datetime

from taurus.data.provider import MarketDataProvider
from taurus.data.repository import PriceBarRepository


def ingest_daily_bars(
    provider: MarketDataProvider,
    repository: PriceBarRepository,
    symbol: str,
    start: datetime,
    end: datetime,
) -> int:
    bars = provider.get_daily_bars(
        symbol=symbol,
        start=start,
        end=end,
    )

    repository.save_bars(bars)

    return len(bars)