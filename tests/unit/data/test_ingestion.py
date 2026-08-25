from datetime import datetime

from taurus.data.ingestion import ingest_daily_bars
from taurus.data.schemas import BarInterval, PriceBar


def test_ingest_daily_bars_fetches_and_saves_data():
    expected_bars = [
        PriceBar(
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
    ]

    class FakeProvider:
        def get_daily_bars(self, symbol, start, end):
            return expected_bars

    class FakeRepository:
        def __init__(self):
            self.saved_bars = []

        def save_bars(self, bars):
            self.saved_bars = bars

        def get_bars(self, symbol):
            return self.saved_bars

    provider = FakeProvider()
    repository = FakeRepository()

    count = ingest_daily_bars(
        provider=provider,
        repository=repository,
        symbol="NVDA",
        start=datetime(2026, 8, 17),
        end=datetime(2026, 8, 18),
    )

    assert count == 1
    assert repository.saved_bars == expected_bars