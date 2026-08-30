def normalize_feature(
    key: str,
    value: float,
) -> float:
    """Normalize a single feature into a more neural-network-friendly scale."""

    if key.startswith("return_"):
        return value

    if key.startswith("relative_return_"):
        return value

    if key.startswith("rsi_"):
        return value / 100.0

    if key.endswith("_adx"):
        return value / 100.0

    if key.endswith("_plus_di"):
        return value / 100.0

    if key.endswith("_minus_di"):
        return value / 100.0

    if key.startswith("volume_ratio_"):
        return value

    return value

def normalize_market_features(
    features: dict[str, float],
    current_price: float,
) -> dict[str, float]:
    """Normalize a complete market feature set."""

    if current_price <= 0:
        raise ValueError(
            "Current price must be greater than zero."
        )

    normalized = {}

    for key, value in features.items():
        if key.startswith("sma_"):
            normalized[key] = (
                current_price - value
            ) / current_price

        elif key.startswith("ema_"):
            normalized[key] = (
                current_price - value
            ) / current_price

        elif key.startswith("atr_"):
            normalized[key] = (
                value / current_price
            )

        elif key.startswith("vwap_"):
            normalized[key] = (
                current_price - value
            ) / current_price

        else:
            normalized[key] = normalize_feature(
                key,
                value,
            )

    return normalized