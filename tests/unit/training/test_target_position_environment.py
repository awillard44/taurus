from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from taurus.simulation.actions import TradingAction
from taurus.training.target_position_environment import (
    TargetPositionEnvironment,
)


class StubTradingEnvironment(gym.Env):
    metadata = {}

    def __init__(
        self,
        shares: float,
    ):
        super().__init__()

        self.action_space = gym.spaces.Discrete(3)

        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        self.state = SimpleNamespace(
            portfolio=SimpleNamespace(
                shares=shares,
            )
        )

        self.last_action = None

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        super().reset(seed=seed)

        return (
            np.asarray(
                [0.0],
                dtype=np.float32,
            ),
            {},
        )

    def step(
        self,
        action,
    ):
        self.last_action = action

        return (
            np.asarray(
                [0.0],
                dtype=np.float32,
            ),
            0.0,
            False,
            False,
            {},
        )


def test_target_position_environment_has_two_actions():
    environment = StubTradingEnvironment(
        shares=0.0,
    )

    wrapper = TargetPositionEnvironment(
        environment,  # type: ignore[arg-type]
    )

    assert wrapper.action_space.n == 2


def test_cash_target_while_flat_translates_to_hold():
    environment = StubTradingEnvironment(
        shares=0.0,
    )

    wrapper = TargetPositionEnvironment(
        environment,  # type: ignore[arg-type]
    )

    wrapper.step(0)

    assert environment.last_action == TradingAction.HOLD


def test_long_target_while_flat_translates_to_buy():
    environment = StubTradingEnvironment(
        shares=0.0,
    )

    wrapper = TargetPositionEnvironment(
        environment,  # type: ignore[arg-type]
    )

    wrapper.step(1)

    assert environment.last_action == TradingAction.BUY


def test_long_target_while_invested_translates_to_hold():
    environment = StubTradingEnvironment(
        shares=10.0,
    )

    wrapper = TargetPositionEnvironment(
        environment,  # type: ignore[arg-type]
    )

    wrapper.step(1)

    assert environment.last_action == TradingAction.HOLD


def test_cash_target_while_invested_translates_to_sell():
    environment = StubTradingEnvironment(
        shares=10.0,
    )

    wrapper = TargetPositionEnvironment(
        environment,  # type: ignore[arg-type]
    )

    wrapper.step(0)

    assert environment.last_action == TradingAction.SELL


def test_target_position_environment_rejects_invalid_action():
    environment = StubTradingEnvironment(
        shares=0.0,
    )

    wrapper = TargetPositionEnvironment(
        environment,  # type: ignore[arg-type]
    )

    with pytest.raises(
        ValueError,
        match="Unsupported target position",
    ):
        wrapper.step(2)