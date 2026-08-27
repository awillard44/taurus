from taurus.features.config import IndicatorConfig
from taurus.features.registry import (
    IndicatorValue,
    get_indicator_function,
)


def execute_indicator(
    config: IndicatorConfig,
    *args,
) -> IndicatorValue:
    """Execute a configured indicator against supplied input data."""

    indicator_function = get_indicator_function(
        config.indicator
    )

    return indicator_function(
        *args,
        **config.parameters,
    )