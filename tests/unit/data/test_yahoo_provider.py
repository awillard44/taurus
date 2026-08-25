from datetime import datetime

import pandas as pd

from taurus.data.providers.yahoo import YahooFinanceProvider
from taurus.data.schemas import PriceBar

def test_yahoo_provider_converts_history_to_price_bars(monkeypatch):
    fake_history = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [105.0],
            "Low": [99.0],
            "Close": [103.0],
            "Volume": [1_000_000],
        },
        index=pd.DatetimeIndex(["2026-08-10"]),
    )

    class FakeTicker:
        def history(self, **kwargs):
            return fake_history

    monkeypatch.setattr(
        "taurus.data.providers.yahoo.yf.Ticker",
        lambda symbol: FakeTicker(),
    )

    provider = YahooFinanceProvider()

    bars = provider.get_daily_bars(
        "NVDA",
        datetime(2026, 8, 10),
        datetime(2026, 8, 11),
    )

    assert len(bars) == 1
    assert isinstance(bars[0], PriceBar)
    assert bars[0].symbol == "NVDA"
    assert bars[0].close == 103.0
    assert bars[0].volume == 1_000_000
    assert bars[0].source == "yahoo"