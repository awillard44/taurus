from datetime import datetime

from taurus.data.schemas import BarInterval, PriceBar
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
        interval=BarInterval.ONE_DAY,
    )

    repository.save_bars([bar])

    bars = repository.get_bars(
        "NVDA",
        BarInterval.ONE_DAY,
    )

    assert len(bars) == 1
    assert bars[0] == bar

def test_get_bars_filters_by_interval(tmp_path):
    database_path = tmp_path / "test.db"

    repository = SQLitePriceBarRepository(database_path)

    daily_bar = PriceBar(
        symbol="NVDA",
        timestamp=datetime(2026, 8, 20),
        open=180.0,
        high=184.0,
        low=178.0,
        close=182.0,
        volume=42_000_000,
        source="test",
        interval=BarInterval.ONE_DAY,
    )

    five_minute_bar = PriceBar(
        symbol="NVDA",
        timestamp=datetime(2026, 8, 20, 10, 0),
        open=181.0,
        high=182.0,
        low=180.5,
        close=181.5,
        volume=500_000,
        source="test",
        interval=BarInterval.FIVE_MINUTES,
    )

    repository.save_bars([
        daily_bar,
        five_minute_bar,
    ])

    bars = repository.get_bars(
        "NVDA",
        BarInterval.ONE_DAY,
    )

    assert bars == [daily_bar]