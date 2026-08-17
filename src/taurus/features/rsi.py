def calculate_rsi_14(closing_prices: list[float]) -> float:
    # Calculate the 14-period Relative Strength Index

    if len(closing_prices) != 15:
        raise ValueError("Expected exactly 15 closing prices.")