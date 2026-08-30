import pytest

from taurus.environment.normalization import (
    normalize_feature,
    normalize_market_features,
)


def test_normalize_return_leaves_value_unchanged():
    assert normalize_feature(
        "return_5",
        0.025,
    ) == pytest.approx(0.025)


def test_normalize_relative_return_leaves_value_unchanged():
    assert normalize_feature(
        "relative_return_20",
        -0.03,
    ) == pytest.approx(-0.03)


def test_normalize_rsi_scales_to_zero_to_one():
    assert normalize_feature(
        "rsi_14",
        55.0,
    ) == pytest.approx(0.55)


def test_normalize_adx_scales_to_zero_to_one():
    assert normalize_feature(
        "adx_14_adx",
        30.0,
    ) == pytest.approx(0.30)


def test_normalize_plus_di_scales_to_zero_to_one():
    assert normalize_feature(
        "adx_14_plus_di",
        25.0,
    ) == pytest.approx(0.25)


def test_normalize_minus_di_scales_to_zero_to_one():
    assert normalize_feature(
        "adx_14_minus_di",
        15.0,
    ) == pytest.approx(0.15)


def test_normalize_volume_ratio_leaves_value_unchanged():
    assert normalize_feature(
        "volume_ratio_20",
        1.75,
    ) == pytest.approx(1.75)


def test_normalize_unknown_feature_leaves_value_unchanged():
    assert normalize_feature(
        "some_future_feature",
        42.0,
    ) == pytest.approx(42.0)

def test_normalize_market_features_scales_sma_relative_to_price():
    result = normalize_market_features(
        features={
            "sma_20": 95.0,
        },
        current_price=100.0,
    )

    assert result["sma_20"] == pytest.approx(
        0.05
    )


def test_normalize_market_features_scales_atr_relative_to_price():
    result = normalize_market_features(
        features={
            "atr_14": 2.5,
        },
        current_price=100.0,
    )

    assert result["atr_14"] == pytest.approx(
        0.025
    )


def test_normalize_market_features_applies_basic_normalization():
    result = normalize_market_features(
        features={
            "return_5": 0.03,
            "rsi_14": 60.0,
            "adx_14_adx": 25.0,
        },
        current_price=100.0,
    )

    assert result["return_5"] == pytest.approx(
        0.03
    )

    assert result["rsi_14"] == pytest.approx(
        0.60
    )

    assert result["adx_14_adx"] == pytest.approx(
        0.25
    )


def test_normalize_market_features_requires_positive_price():
    with pytest.raises(ValueError):
        normalize_market_features(
            features={
                "sma_20": 100.0,
            },
            current_price=0.0,
        )