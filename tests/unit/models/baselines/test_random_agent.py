from taurus.models.baselines.random_agent import RandomAgent
from taurus.simulation.actions import TradingAction


def test_random_agent_returns_valid_action():
    agent = RandomAgent()

    action = agent.predict()

    assert action in TradingAction