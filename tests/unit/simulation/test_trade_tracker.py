import pytest

from datetime import datetime

from taurus.simulation.trade_tracker import(
    OpenPosition,
    add_shares,
    close_shares,
)


def test_add_shares_creates_new_position():
    timestamp = datetime(2026, 8, 25)

    position = add_shares(
        position=None,
        timestamp=timestamp,
        price=100.0,
        shares=5.0,
    )

    assert position.entry_timestamp == timestamp
    assert position.average_entry_price == 100.0
    assert position.shares == 5.0


def test_add_shares_updates_weighted_average_price():
    position = OpenPosition(
        entry_timestamp=datetime(2026, 8, 25),
        average_entry_price=100.0,
        shares=5.0,
    )

    updated = add_shares(
        position=position,
        timestamp=datetime(2026, 8, 26),
        price=120.0,
        shares=5.0,
    )

    assert updated.entry_timestamp == position.entry_timestamp
    assert updated.average_entry_price == 110.0
    assert updated.shares == 10.0


def test_add_shares_rejects_invalid_price():
    with pytest.raises(ValueError):
        add_shares(
            position=None,
            timestamp=datetime(2026, 8, 25),
            price=0.0,
            shares=5.0,
        )


def test_add_shares_rejects_invalid_share_count():
    with pytest.raises(ValueError):
        add_shares(
            position=None,
            timestamp=datetime(2026, 8, 25),
            price=100.0,
            shares=0.0,
        )

def test_close_shares_partial_position_with_profit():
    position = OpenPosition(
        entry_timestamp=datetime(2026, 8, 25),
        average_entry_price=100.0,
        shares=10.0,
    )

    remaining, closed = close_shares(
        position=position,
        timestamp=datetime(2026, 8, 26),
        price=120.0,
        shares=3.0,
    )

    assert remaining is not None
    assert remaining.shares == 7.0
    assert remaining.average_entry_price == 100.0

    assert closed.shares == 3.0
    assert closed.entry_value == 300.0
    assert closed.exit_value == 360.0
    assert closed.realized_pnl == 60.0
    assert closed.return_pct == pytest.approx(0.20)


def test_close_shares_full_position():
    position = OpenPosition(
        entry_timestamp=datetime(2026, 8, 25),
        average_entry_price=100.0,
        shares=10.0,
    )

    remaining, closed = close_shares(
        position=position,
        timestamp=datetime(2026, 8, 26),
        price=120.0,
        shares=10.0,
    )

    assert remaining is None
    assert closed.realized_pnl == 200.0
    assert closed.return_pct == pytest.approx(0.20)


def test_close_shares_with_loss():
    position = OpenPosition(
        entry_timestamp=datetime(2026, 8, 25),
        average_entry_price=100.0,
        shares=5.0,
    )

    remaining, closed = close_shares(
        position=position,
        timestamp=datetime(2026, 8, 26),
        price=80.0,
        shares=5.0,
    )

    assert remaining is None
    assert closed.entry_value == 500.0
    assert closed.exit_value == 400.0
    assert closed.realized_pnl == -100.0
    assert closed.return_pct == pytest.approx(-0.20)


def test_close_shares_rejects_more_than_open_position():
    position = OpenPosition(
        entry_timestamp=datetime(2026, 8, 25),
        average_entry_price=100.0,
        shares=5.0,
    )

    with pytest.raises(ValueError):
        close_shares(
            position=position,
            timestamp=datetime(2026, 8, 26),
            price=120.0,
            shares=6.0,
        )


def test_close_shares_rejects_invalid_price():
    position = OpenPosition(
        entry_timestamp=datetime(2026, 8, 25),
        average_entry_price=100.0,
        shares=5.0,
    )

    with pytest.raises(ValueError):
        close_shares(
            position=position,
            timestamp=datetime(2026, 8, 26),
            price=0.0,
            shares=5.0,
        )


def test_close_shares_rejects_invalid_share_count():
    position = OpenPosition(
        entry_timestamp=datetime(2026, 8, 25),
        average_entry_price=100.0,
        shares=5.0,
    )

    with pytest.raises(ValueError):
        close_shares(
            position=position,
            timestamp=datetime(2026, 8, 26),
            price=120.0,
            shares=0.0,
        )