def calculate_portfolio_return_reward(
    previous_value: float,
    current_value: float,
) -> float:
    """Calculate reward as portfolio percentage return."""

    if previous_value <= 0:
        raise ValueError("Previous portfolio value must be greater than zero.")

    return (current_value - previous_value) / previous_value