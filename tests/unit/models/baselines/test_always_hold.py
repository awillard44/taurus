from taurus.models.baselines.always_hold import AlwaysHoldAgent
from taurus.simulation.actions import TradingAction


def test_always_hold_agent_returns_hold():
    agent = AlwaysHoldAgent()

    action = agent.predict()

    assert action == TradingAction.HOLD