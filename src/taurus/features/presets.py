from taurus.features.config import (
    FeatureSetConfig,
    IndicatorConfig,
)


DEFAULT_FEATURE_SET = FeatureSetConfig(
    name="default",
    indicators=(
        IndicatorConfig(
            key="return_1",
            indicator="return",
            inputs=("close",),
            parameters={"periods": 1},
        ),
        IndicatorConfig(
            key="return_5",
            indicator="return",
            inputs=("close",),
            parameters={"periods": 5},
        ),
        IndicatorConfig(
            key="sma_20",
            indicator="sma",
            inputs=("close",),
            parameters={"period": 20},
        ),
        IndicatorConfig(
            key="rsi_14",
            indicator="rsi",
            inputs=("close",),
            parameters={"period": 14},
        ),
        IndicatorConfig(
            key="atr_14",
            indicator="atr",
            inputs=("high", "low", "close"),
            parameters={"period": 14},
        ),
        IndicatorConfig(
            key="volume_ratio_20",
            indicator="volume_ratio",
            inputs=("volume",),
            parameters={"period": 20},
        ),
        IndicatorConfig(
            key="relative_return_20",
            indicator="relative_return",
            inputs=("close", "benchmark_close"),
            parameters={"periods": 20},
        ),
        IndicatorConfig(
            key="adx_14",
            indicator="adx",
            inputs=("high", "low", "close"),
            parameters={"period": 14},
        ),
    ),
)