from taurus.features.presets import DEFAULT_FEATURE_SET
from taurus.features.feature_set_executor import execute_feature_set

def test_default_feature_set_name():
    assert DEFAULT_FEATURE_SET.name == "default"


def test_default_feature_set_contains_expected_indicators():
    indicator_keys = {
        indicator.key
        for indicator in DEFAULT_FEATURE_SET.indicators
    }

    assert indicator_keys == {
        "return_1",
        "return_5",
        "sma_20",
        "rsi_14",
        "atr_14",
        "volume_ratio_20",
        "relative_return_20",
        "adx_14",
    }


def test_default_feature_set_executes():
    values = [
        float(value)
        for value in range(100, 160)
    ]

    results = execute_feature_set(
        config=DEFAULT_FEATURE_SET,
        input_data={
            "close": values,
            "high": [
                value + 1.0
                for value in values
            ],
            "low": [
                value - 1.0
                for value in values
            ],
            "volume": [
                1_000_000.0 + index * 10_000.0
                for index in range(len(values))
            ],
            "benchmark_close": [
                value * 0.9
                for value in values
            ],
        },
    )

    assert set(results) == {
        "return_1",
        "return_5",
        "sma_20",
        "rsi_14",
        "atr_14",
        "volume_ratio_20",
        "relative_return_20",
        "adx_14_adx",
        "adx_14_plus_di",
        "adx_14_minus_di",
    }