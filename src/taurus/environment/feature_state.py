from dataclasses import dataclass

from taurus.features.market_state import MarketState


@dataclass(frozen=True)
class FeatureEnvironmentState:
    market: MarketState
    features: dict[str, float]

@dataclass(frozen=True)
class FeatureEnvironmentState:
    market: MarketState
    features: dict[str, float]
    open_price: float | None = None