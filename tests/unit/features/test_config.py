from taurus.features.config import (
    FeatureSetConfig,
    IndicatorConfig,
)


def test_indicator_config_stores_values():
    config = IndicatorConfig(
        key="rsi_14",
        indicator="rsi",
        inputs=("close",),
        parameters={"period": 14},
    )

    assert config.key == "rsi_14"
    assert config.indicator == "rsi"
    assert config.inputs == ("close",)
    assert config.parameters == {"period": 14}


def test_feature_set_config_stores_indicators():
    rsi = IndicatorConfig(
        key="rsi_14",
        indicator="rsi",
        inputs=("close",),
        parameters={"period": 14},
    )

    sma = IndicatorConfig(
        key="sma_20",
        indicator="sma",
        inputs=("close",),
        parameters={"period": 20},
    )

    feature_set = FeatureSetConfig(
        name="default_v1",
        indicators=(rsi, sma),
    )

    assert feature_set.name == "default_v1"
    assert feature_set.indicators == (rsi, sma)