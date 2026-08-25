from datetime import datetime

import pytest

from taurus.data.schemas import BarInterval, PriceBar


def test_price_bar_stores_market_data():
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

    assert bar.symbol == "NVDA"
    assert bar.close == 182.75
    assert bar.volume == 42_000_000
    assert bar.interval == BarInterval.ONE_DAY



def test_price_bar_is_immutable():
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

    with pytest.raises(Exception):
        bar.close = 200.0