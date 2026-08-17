def calculate_sma_20(closing_prices: list[float]) -> float:
    # Calculate the simple moving average of 20 closing prices

    if len(closing_prices) != 20:
        raise ValueError("Expected exactly 20 closing prices.")

    return sum(closing_prices) / len(closing_prices)

def calculate_sma_50(closing_prices: list[float]) -> float:
    # Calculate the simple moving average of 50 closing prices

    if len(closing_prices) != 50:
        raise ValueError("Expected exactly 50 closing prices")

    return sum(closing_prices) / len(closing_prices)