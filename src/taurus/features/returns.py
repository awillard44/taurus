from taurus.data.schemas import PriceBar


def calculate_return_1d(
    previous_bar: PriceBar,
    current_bar: PriceBar,
) -> float:
    """Calculate the return between two consecutive daily price bars."""

    daily_return = (
        current_bar.close - previous_bar.close
    ) / previous_bar.close

    return daily_return

def calculate_return_5d(
    five_days_ago_bar: PriceBar,
    current_bar: PriceBar,
) -> float:
    """Calculate the return over five trading days."""

    five_day_return = (
        current_bar.close - five_days_ago_bar.close
    ) / five_days_ago_bar.close

    return five_day_return

def calculate_return_20d(
    twenty_days_ago_bar: PriceBar,
    current_bar: PriceBar,
) -> float:
    """Calculate the return over twenty trading days."""

    twenty_day_return = (
        current_bar.close - twenty_days_ago_bar.close
    ) / twenty_days_ago_bar.close

    return twenty_day_return