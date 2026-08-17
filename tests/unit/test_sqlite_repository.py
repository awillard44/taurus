from datetime import datetime

from taurus.data.schemas import PriceBar
from taurus.data.sqlite_repository import SQLitePriceBarRepository


def test_save_and_get_price_bar(tmp_path):
    database_path = tmp_path / "test.db"

    repository = SQLitePriceBarRepository(database_path)

    bar = PriceBar(
        symbol="NVDA",
        timestamp=datetime(2026, 8, 17),
        open=180.0,
        high=184.0,
        low=178.5,
        close=182.75,
        volume=42_000_000,
        source="test",
    )

    repository.save_bars([bar])

    bars = repository.get_bars("NVDA")

    assert len(bars) == 1
    assert bars[0] == bar