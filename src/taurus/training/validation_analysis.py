from dataclasses import dataclass
from math import sqrt

from taurus.training.ppo_evaluator import PPOStepRecord


@dataclass(frozen=True)
class FeaturePolicyAssociation:
    feature: str
    correlation_with_long_probability: float


def _pearson_correlation(
    x_values: list[float],
    y_values: list[float],
) -> float:
    if len(x_values) != len(y_values):
        raise ValueError(
            "Correlation inputs must have equal lengths."
        )

    if len(x_values) < 2:
        raise ValueError(
            "Correlation requires at least two observations."
        )

    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)

    covariance = sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(
            x_values,
            y_values,
            strict=True,
        )
    )

    x_variance = sum(
        (x - x_mean) ** 2
        for x in x_values
    )

    y_variance = sum(
        (y - y_mean) ** 2
        for y in y_values
    )

    denominator = sqrt(
        x_variance * y_variance
    )

    if denominator == 0.0:
        return 0.0

    return covariance / denominator


def calculate_feature_policy_associations(
    step_records: tuple[PPOStepRecord, ...],
) -> tuple[FeaturePolicyAssociation, ...]:
    if not step_records:
        raise ValueError(
            "Step records must not be empty."
        )

    feature_names = [
        key
        for key, _ in (
            step_records[0]
            .normalized_feature_values
        )
    ]

    long_probabilities = [
        record.long_probability
        for record in step_records
    ]

    associations = []

    for feature_name in feature_names:
        feature_values = []

        for record in step_records:
            normalized_features = dict(
                record.normalized_feature_values
            )

            feature_values.append(
                normalized_features[feature_name]
            )

        correlation = _pearson_correlation(
            x_values=feature_values,
            y_values=long_probabilities,
        )

        associations.append(
            FeaturePolicyAssociation(
                feature=feature_name,
                correlation_with_long_probability=(
                    correlation
                ),
            )
        )

    return tuple(
        sorted(
            associations,
            key=lambda association: abs(
                association
                .correlation_with_long_probability
            ),
            reverse=True,
        )
    )