from taurus.features.config import FeatureSetConfig
from taurus.features.executor import execute_indicator


def execute_feature_set(
    config: FeatureSetConfig,
    input_data: dict[str, list[float]],
) -> dict[str, float]:
    # Execute configured indicators against named input series

    results: dict[str, float] = {}

    for indicator_config in config.indicators:
        inputs = [
            input_data[input_name]
            for input_name in indicator_config.inputs
        ]

        value = execute_indicator(
            indicator_config,
            *inputs,
        )

        if isinstance(value, dict):
            for output_name, output_value in value.items():
                results[
                    f"{indicator_config.key}_{output_name}"
                ] = output_value
        else:
            results[indicator_config.key] = value

    return results