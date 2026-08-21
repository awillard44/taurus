from datetime import datetime
from pathlib import Path

from taurus.data.ingestion import ingest_daily_bars
from taurus.data.providers.yahoo import YahooFinanceProvider
from taurus.data.sqlite_repository import SQLitePriceBarRepository


database_path = Path("data/taurus.db")

database_path.parent.mkdir(exist_ok=True)

provider = YahooFinanceProvider()
repository = SQLitePriceBarRepository(database_path)

symbols = ["NVDA", "SPY"]

for symbol in symbols:
    count = ingest_daily_bars(
        provider=provider,
        repository=repository,
        symbol=symbol,
        start=datetime(2026, 5, 1),
        end=datetime(2026, 8, 21),
    )

    print(f"Ingested {count} {symbol} price bars.")