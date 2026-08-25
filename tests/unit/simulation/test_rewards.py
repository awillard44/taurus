import pytest

from taurus.simulation.rewards import calculate_portfolio_return_reward


def test_positive_portfolio_return_reward():
    result = calculate_portfolio_return_reward(
        previous_value=1_000.0,
        current_value=1_050.0,
    )

    assert result == pytest.approx(0.05)


def test_negative_portfolio_return_reward():
    result = calculate_portfolio_return_reward(
        previous_value=1_000.0,
        current_value=950.0,
    )

    assert result == pytest.approx(-0.05)


def test_zero_portfolio_return_reward():
    result = calculate_portfolio_return_reward(
        previous_value=1_000.0,
        current_value=1_000.0,
    )

    assert result == 0.0


def test_reward_rejects_nonpositive_previous_value():
    with pytest.raises(ValueError):
        calculate_portfolio_return_reward(
            previous_value=0.0,
            current_value=1_000.0,
        )