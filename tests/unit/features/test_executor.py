from taurus.features.config import IndicatorConfig
from taurus.features.executor import execute_indicator


def test_execute_indicator_runs_configured_rsi():
    config = IndicatorConfig(
        key="rsi_3",
        indicator="rsi",
        inputs=("close",),
        parameters={"period": 3},
    )

    closing_prices = [
        100.0,
        103.0,
        101.0,
        105.0,
    ]

    result = execute_indicator(
        config,
        closing_prices,
    )

    assert result == 77.77777777777777


def test_execute_indicator_runs_configured_sma():
    config = IndicatorConfig(
        key="sma_3",
        indicator="sma",
        inputs=("close",),
        parameters={"period": 3},
    )

    closing_prices = [
        100.0,
        110.0,
        120.0,
    ]

    result = execute_indicator(
        config,
        closing_prices,
    )

    assert result == 110.0


def test_execute_indicator_uses_configured_parameters():
    closing_prices = [
        100.0,
        110.0,
        120.0,
        130.0,
    ]

    sma_2 = IndicatorConfig(
        key="sma_2",
        indicator="sma",
        inputs=("close",),
        parameters={"period": 2},
    )

    sma_4 = IndicatorConfig(
        key="sma_4",
        indicator="sma",
        inputs=("close",),
        parameters={"period": 4},
    )

    result_2 = execute_indicator(
        sma_2,
        closing_prices,
    )

    result_4 = execute_indicator(
        sma_4,
        closing_prices,
    )

    assert result_2 == 125.0
    assert result_4 == 115.0