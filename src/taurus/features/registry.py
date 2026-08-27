from collections.abc import Callable

from taurus.features.moving_average import calculate_sma
from taurus.features.returns import calculate_return
from taurus.features.rsi import calculate_rsi
from taurus.features.volatility import calculate_volatility
from taurus.features.volume import calculate_volume_ratio
from taurus.features.relative_return import calculate_relative_return
from taurus.features.atr import calculate_atr
from taurus.features.bollinger_bands import calculate_bollinger_bands
from taurus.features.ema import calculate_ema
from taurus.features.macd import calculate_macd
from taurus.features.stochastic import calculate_stochastic
from taurus.features.vwap import calculate_vwap
from taurus.features.obv import calculate_obv
from taurus.features.adx import calculate_adx


IndicatorValue = float | dict[str, float]
IndicatorFunction = Callable[..., IndicatorValue]


INDICATOR_REGISTRY: dict[str, IndicatorFunction] = {
    "return": calculate_return,
    "sma": calculate_sma,
    "rsi": calculate_rsi,
    "volatility": calculate_volatility,
    "volume_ratio": calculate_volume_ratio,
    "relative_return": calculate_relative_return,
    "atr": calculate_atr,
    "bollinger_bands": calculate_bollinger_bands,
    "ema": calculate_ema,
    "macd": calculate_macd,
    "stochastic": calculate_stochastic,
    "vwap": calculate_vwap,
    "obv": calculate_obv,
    "adx": calculate_adx,
}


def get_indicator_function(name: str) -> IndicatorFunction:
    # Return the calculation function registered for an indicator
    try:
        return INDICATOR_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown indicator: {name}"
        ) from exc