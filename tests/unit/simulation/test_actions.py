from taurus.simulation.actions import TradingAction


def test_trading_action_values():
    assert TradingAction.HOLD == 0
    assert TradingAction.BUY == 1
    assert TradingAction.SELL == 2