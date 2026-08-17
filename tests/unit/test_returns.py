from datetime import datetime

from taurus.data.schemas import PriceBar
from taurus.features.returns import (
    calculate_return_1d,
    calculate_return_5d,
    calculate_return_20d,
)

def test_calculate_return_1d():
    previous_bar = PriceBar(
        symbol="NVDA",
        timestamp=datetime(2026, 8, 17),
        open=100.0,
        high=102.0,
        low=99.0,
        close=100.0,
        volume=1_000_000,
        source="test",
    )

    current_bar = PriceBar(
        symbol="NVDA",
        timestamp=datetime(2026, 8, 18),
        open=101.0,
        high=104.0,
        low=100.0,
        close=103.0,
        volume=1_100_000,
        source="test",
    )

    result = calculate_return_1d(previous_bar, current_bar)

    assert result == 0.03

def test_calculate_negative_return_1d():
    previous_bar = PriceBar(
        symbol="NVDA",
        timestamp=datetime(2026, 8, 17),
        open=100.0,
        high=102.0,
        low=99.0,
        close=100.0,
        volume=1_000_000,
        source="test",
    )

    current_bar = PriceBar(
        symbol="NVDA",
        timestamp=datetime(2026, 8, 18),
        open=99.0,
        high=100.0,
        low=97.0,
        close=98.0,
        volume=1_100_000,
        source="test",
    )

    result = calculate_return_1d(previous_bar, current_bar)

    assert result == -0.02


def test_calculate_return_5d():
    five_days_ago_bar = PriceBar(
        symbol="NVDA",
        timestamp=datetime(2026, 8, 10),
        open=100.0,
        high=102.0,
        low=99.0,
        close=100.0,
        volume=1_000_000,
        source="test",
    )

    current_bar = PriceBar(
        symbol="NVDA",
        timestamp=datetime(2026, 8, 17),
        open=108.0,
        high=111.0,
        low=107.0,
        close=110.0,
        volume=1_200_000,
        source="test",
    )

    result = calculate_return_5d(five_days_ago_bar, current_bar)

    assert result == 0.10

def test_calculate_return_20d():
    twenty_days_ago_bar = PriceBar(
        symbol="NVDA",
        timestamp=datetime(2026, 7, 20),
        open=100.0,
        high=102.0,
        low=99.0,
        close=100.0,
        volume=1_000_000,
        source="test",
    )

    current_bar = PriceBar(
        symbol="NVDA",
        timestamp=datetime(2026, 8, 17),
        open=118.0,
        high=121.0,
        low=117.0,
        close=120.0,
        volume=1_200_000,
        source="test",
    )

    result = calculate_return_20d(twenty_days_ago_bar, current_bar)

    assert result == 0.20