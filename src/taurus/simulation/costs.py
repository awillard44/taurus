from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionCosts:
    commission_rate: float = 0.0
    slippage_rate: float = 0.0