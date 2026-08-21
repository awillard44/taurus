from pathlib import Path

from taurus.data.schemas import BarInterval
from taurus.data.sqlite_repository import SQLitePriceBarRepository
from taurus.features.state_builder import build_market_state


database_path = Path("data/taurus.db")

repository = SQLitePriceBarRepository(database_path)

nvda_bars = repository.get_bars(
    "NVDA",
    BarInterval.ONE_DAY,
)

spy_bars = repository.get_bars(
    "SPY",
    BarInterval.ONE_DAY,
)

state = build_market_state(
    bars=nvda_bars,
    benchmark_bars=spy_bars,
)

print(state)