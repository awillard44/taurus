from datetime import datetime

import yfinance as yf

from taurus.data.schemas import PriceBar

class YahooFinanceProvider:
    # Market-data provider backed by Yahoo Finance

    def get_daily_bars(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> list[PriceBar]:
        ticker = yf.Ticker(symbol)

        history = ticker.history(
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
        )

        bars: list[PriceBar] = []

        for timestamp, row in history.iterrows():
            bar = PriceBar(
                symbol=symbol,
                timestamp=timestamp.to_pydatetime(),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=int(row["Volume"]),
                source="yahoo",
            )
            bars.append(bar
            )
        return bars