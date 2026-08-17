from typing import Protocol

from taurus.data.schemas import PriceBar


class PriceBarRepository(Protocol):
    # Interface for storing and retrieving market price bars

    def save_bars(self, bars: list[PriceBar]) -> None:
        ...

    def get_bars(self, symbol: str) -> list[PriceBar]:
        ...