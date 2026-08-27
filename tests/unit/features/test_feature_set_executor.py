import pytest

from statistics import mean, pstdev
from taurus.features.config import (
    FeatureSetConfig,
    IndicatorConfig,
)
from taurus.features.feature_set_executor import execute_feature_set
from taurus.features.registry import INDICATOR_REGISTRY


def test_execute_feature_set_runs_multiple_indicators():
    feature_set = FeatureSetConfig(
        name="test_features",
        indicators=(
            IndicatorConfig(
                key="rsi_3",
                indicator="rsi",
                inputs=("close",),
                parameters={"period": 3},
            ),
            IndicatorConfig(
                key="sma_3",
                indicator="sma",
                inputs=("close",),
                parameters={"period": 3},
            ),
        ),
    )

    closing_prices = [
        100.0,
        103.0,
        101.0,
        105.0,
    ]

    input_data = {
        "close": closing_prices,
    }

    results = execute_feature_set(
        config=feature_set,
        input_data=input_data,
    )

    assert results["rsi_3"] == 77.77777777777777
    assert results["sma_3"] == 103.0


def test_execute_feature_set_uses_feature_keys():
    feature_set = FeatureSetConfig(
        name="multiple_smas",
        indicators=(
            IndicatorConfig(
                key="short_sma",
                indicator="sma",
                inputs=("close",),
                parameters={"period": 2},
            ),
            IndicatorConfig(
                key="long_sma",
                indicator="sma",
                inputs=("close",),
                parameters={"period": 4},
            ),
        ),
    )

    closing_prices = [
        100.0,
        110.0,
        120.0,
        130.0,
    ]

    results = execute_feature_set(
        config=feature_set,
        input_data={
            "close": closing_prices,
        },
    )

    assert results == {
        "short_sma": 125.0,
        "long_sma": 115.0,
    }


def test_execute_feature_set_supports_multi_input_indicator():
    feature_set = FeatureSetConfig(
        name="relative_return_test",
        indicators=(
            IndicatorConfig(
                key="relative_return_2",
                indicator="relative_return",
                inputs=("close", "benchmark_close"),
                parameters={"periods": 2},
            ),
        ),
    )

    asset_prices = [
        100.0,
        105.0,
        110.0,
    ]

    benchmark_prices = [
        100.0,
        102.0,
        104.0,
    ]

    results = execute_feature_set(
        config=feature_set,
        input_data={
            "close": asset_prices,
            "benchmark_close": benchmark_prices,
        },
    )

    assert results["relative_return_2"] == pytest.approx(0.06)


def test_execute_feature_set_flattens_multi_output_indicator(monkeypatch):
    def fake_indicator(values):
        return {
            "lower": 90.0,
            "middle": 100.0,
            "upper": 110.0,
        }

    monkeypatch.setitem(
        INDICATOR_REGISTRY,
        "fake_bands",
        fake_indicator,
    )

    feature_set = FeatureSetConfig(
        name="multi_output_test",
        indicators=(
            IndicatorConfig(
                key="bands",
                indicator="fake_bands",
                inputs=("close",),
            ),
        ),
    )

    results = execute_feature_set(
        config=feature_set,
        input_data={
            "close": [
                90.0,
                100.0,
                110.0,
            ],
        },
    )

    assert results == {
        "bands_lower": 90.0,
        "bands_middle": 100.0,
        "bands_upper": 110.0,
    }


def test_execute_feature_set_runs_bollinger_bands():
    feature_set = FeatureSetConfig(
        name="bollinger_test",
        indicators=(
            IndicatorConfig(
                key="bollinger_5",
                indicator="bollinger_bands",
                inputs=("close",),
                parameters={
                    "period": 5,
                    "stddev_multiplier": 2.0,
                },
            ),
        ),
    )

    closing_prices = [
        100.0,
        102.0,
        104.0,
        106.0,
        108.0,
    ]

    results = execute_feature_set(
        config=feature_set,
        input_data={
            "close": closing_prices,
        },
    )

    middle = mean(closing_prices)
    deviation = pstdev(closing_prices)

    assert results["bollinger_5_middle"] == pytest.approx(
        middle
    )

    assert results["bollinger_5_lower"] == pytest.approx(
        middle - (2.0 * deviation)
    )

    assert results["bollinger_5_upper"] == pytest.approx(
        middle + (2.0 * deviation)
    )


def test_execute_feature_set_runs_macd():
    feature_set = FeatureSetConfig(
        name="macd_test",
        indicators=(
            IndicatorConfig(
                key="macd_test",
                indicator="macd",
                inputs=("close",),
                parameters={
                    "fast_period": 3,
                    "slow_period": 5,
                    "signal_period": 2,
                },
            ),
        ),
    )

    closing_prices = [
        float(value)
        for value in range(100, 140)
    ]

    results = execute_feature_set(
        config=feature_set,
        input_data={
            "close": closing_prices,
        },
    )

    assert "macd_test_macd" in results
    assert "macd_test_signal" in results
    assert "macd_test_histogram" in results


def test_execute_feature_set_runs_stochastic():
    feature_set = FeatureSetConfig(
        name="stochastic_test",
        indicators=(
            IndicatorConfig(
                key="stochastic_3_2",
                indicator="stochastic",
                inputs=("high", "low", "close"),
                parameters={
                    "k_period": 3,
                    "d_period": 2,
                },
            ),
        ),
    )

    highs = [
        10.0,
        12.0,
        14.0,
        15.0,
        16.0,
    ]

    lows = [
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
    ]

    closes = [
        9.0,
        11.0,
        13.0,
        14.0,
        15.0,
    ]

    results = execute_feature_set(
        config=feature_set,
        input_data={
            "high": highs,
            "low": lows,
            "close": closes,
        },
    )

    assert "stochastic_3_2_k" in results
    assert "stochastic_3_2_d" in results


def test_execute_feature_set_runs_vwap():
    feature_set = FeatureSetConfig(
        name="vwap_test",
        indicators=(
            IndicatorConfig(
                key="vwap_2",
                indicator="vwap",
                inputs=("high", "low", "close", "volume"),
                parameters={
                    "period": 2,
                },
            ),
        ),
    )

    results = execute_feature_set(
        config=feature_set,
        input_data={
            "high": [101.0, 111.0],
            "low": [99.0, 109.0],
            "close": [100.0, 110.0],
            "volume": [1_000_000.0, 9_000_000.0],
        },
    )

    assert results["vwap_2"] == pytest.approx(109.0)


def test_execute_feature_set_runs_obv():
    feature_set = FeatureSetConfig(
        name="obv_test",
        indicators=(
            IndicatorConfig(
                key="obv_3",
                indicator="obv",
                inputs=("close", "volume"),
                parameters={
                    "period": 3,
                },
            ),
        ),
    )

    results = execute_feature_set(
        config=feature_set,
        input_data={
            "close": [
                100.0,
                105.0,
                103.0,
                110.0,
            ],
            "volume": [
                500_000.0,
                1_000_000.0,
                2_000_000.0,
                3_000_000.0,
            ],
        },
    )

    assert results["obv_3"] == pytest.approx(
        2_000_000.0
    )


def test_execute_feature_set_runs_adx():
    feature_set = FeatureSetConfig(
        name="adx_test",
        indicators=(
            IndicatorConfig(
                key="adx_5",
                indicator="adx",
                inputs=("high", "low", "close"),
                parameters={
                    "period": 5,
                },
            ),
        ),
    )

    results = execute_feature_set(
        config=feature_set,
        input_data={
            "high": [
                float(value)
                for value in range(101, 131)
            ],
            "low": [
                float(value)
                for value in range(99, 129)
            ],
            "close": [
                float(value)
                for value in range(100, 130)
            ],
        },
    )

    assert "adx_5_adx" in results
    assert "adx_5_plus_di" in results
    assert "adx_5_minus_di" in results

    assert (
        results["adx_5_plus_di"]
        > results["adx_5_minus_di"]
    )

    assert results["adx_5_adx"] > 0.0