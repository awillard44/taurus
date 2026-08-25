from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioState:
    cash: float
    shares: float
    asset_price: float
    portfolio_value: float