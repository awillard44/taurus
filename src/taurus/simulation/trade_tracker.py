from dataclasses import dataclass
from datetime import datetime

from taurus.simulation.closed_trade import ClosedTrade


@dataclass
class OpenPosition:
    entry_timestamp: datetime
    average_entry_price: float
    shares: float


def add_shares(
    position: OpenPosition | None,
    timestamp: datetime,
    price: float,
    shares: float,
) -> OpenPosition:
    # Add shares to an open position using weighted average cost

    if price <= 0:
        raise ValueError("Price must be greater than zero.")

    if shares <= 0:
        raise ValueError("Shares must be greater than zero.")

    if position is None:
        return OpenPosition(
            entry_timestamp=timestamp,
            average_entry_price=price,
            shares=shares,
        )

    total_existing_value = (
        position.average_entry_price * position.shares
    )

    new_purchase_value = price * shares

    total_shares = position.shares + shares

    average_entry_price = (total_existing_value + new_purchase_value) / total_shares

    return OpenPosition(
        entry_timestamp=position.entry_timestamp,
        average_entry_price=average_entry_price,
        shares=total_shares,
    )


def close_shares(
    position: OpenPosition,
    timestamp: datetime,
    price: float,
    shares: float,
) -> tuple[OpenPosition | None, ClosedTrade]:
    # Close some or all shares from an open position

    if price <= 0:
        raise ValueError("Price must be greater than zero.")

    if shares <= 0:
        raise ValueError("Shares must be greater than zero.")

    if shares > position.shares:
        raise ValueError("Cannot close more shares than are currently open.")

    entry_value = position.average_entry_price * shares
    exit_value = price * shares
    realized_pnl = exit_value - entry_value
    return_pct = realized_pnl / entry_value

    closed_trade = ClosedTrade(
        entry_timestamp=position.entry_timestamp,
        exit_timestamp=timestamp,
        entry_price=position.average_entry_price,
        exit_price=price,
        shares=shares,
        entry_value=entry_value,
        exit_value=exit_value,
        realized_pnl=realized_pnl,
        return_pct=return_pct,
    )

    remaining_shares = position.shares - shares

    if remaining_shares == 0:
        remaining_position = None
    else:
        remaining_position = OpenPosition(
            entry_timestamp=position.entry_timestamp,
            average_entry_price=position.average_entry_price,
            shares=remaining_shares,
        )

    return remaining_position, closed_trade