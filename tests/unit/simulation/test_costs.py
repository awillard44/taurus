from taurus.simulation.costs import ExecutionCosts


def test_execution_costs_defaults_to_zero():
    costs = ExecutionCosts()

    assert costs.commission_rate == 0.0
    assert costs.slippage_rate == 0.0


def test_execution_costs_stores_rates():
    costs = ExecutionCosts(
        commission_rate=0.001,
        slippage_rate=0.002,
    )

    assert costs.commission_rate == 0.001
    assert costs.slippage_rate == 0.002