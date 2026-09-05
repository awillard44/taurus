from datetime import date, datetime, timezone

import pytest

from taurus.data.schemas import BarInterval, PriceBar
from taurus.environment.feature_state import FeatureEnvironmentState
from taurus.environment.trading_environment import TaurusTradingEnvironment
from taurus.features.presets import DEFAULT_FEATURE_SET
from taurus.simulation.costs import ExecutionCosts
from taurus.training.environment_builder import (
    build_experiment_environment,
)
from taurus.training.experiment_config import (
    DateRange,
    TrainingExperimentConfig,
)


class FakeRepository:
    def __init__(
        self,
        bars_by_symbol: dict[str, list[PriceBar]],
    ):
        self._bars_by_symbol = bars_by_symbol

    def get_bars(
        self,
        symbol: str,
        interval: BarInterval,
    ) -> list[PriceBar]:
        assert interval == BarInterval.ONE_DAY
        return self._bars_by_symbol[symbol]


def make_bars(
    symbol: str,
    start_year: int = 2021,
    end_year: int = 2026,
) -> list[PriceBar]:
    bars = []

    index = 0

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            for day in range(1, 21):
                timestamp = datetime(
                    year,
                    month,
                    day,
                    tzinfo=timezone.utc,
                )

                price = 100.0 + index

                bars.append(
                    PriceBar(
                        symbol=symbol,
                        timestamp=timestamp,
                        open=price,
                        high=price + 1.0,
                        low=price - 1.0,
                        close=price,
                        volume=1_000_000.0,
                        source="test",
                        interval=BarInterval.ONE_DAY,
                    )
                )

                index += 1

    return bars


@pytest.fixture
def experiment_config() -> TrainingExperimentConfig:
    return TrainingExperimentConfig(
        name="nvda_initial",
        symbol="NVDA",
        benchmark_symbol="SPY",
        training=DateRange(
            start=date(2022, 1, 1),
            end=date(2024, 12, 31),
        ),
        validation=DateRange(
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
        ),
        test=DateRange(
            start=date(2026, 1, 1),
            end=date(2026, 8, 21),
        ),
        seed=42,
    )


@pytest.fixture
def repository() -> FakeRepository:
    return FakeRepository(
        {
            "NVDA": make_bars("NVDA"),
            "SPY": make_bars("SPY"),
        }
    )


def test_build_training_environment_uses_only_training_split(
    repository,
    experiment_config,
):
    environment = build_experiment_environment(
        repository=repository,
        config=experiment_config,
        split="training",
    )

    timestamps = [
        state.market.timestamp.date()
        for state in environment.feature_states
    ]

    assert timestamps
    assert min(timestamps) >= experiment_config.training.start
    assert max(timestamps) <= experiment_config.training.end


def test_build_validation_environment_uses_only_validation_split(
    repository,
    experiment_config,
):
    environment = build_experiment_environment(
        repository=repository,
        config=experiment_config,
        split="validation",
    )

    timestamps = [
        state.market.timestamp.date()
        for state in environment.feature_states
    ]

    assert timestamps
    assert min(timestamps) >= experiment_config.validation.start
    assert max(timestamps) <= experiment_config.validation.end


def test_build_test_environment_uses_only_test_split(
    repository,
    experiment_config,
):
    environment = build_experiment_environment(
        repository=repository,
        config=experiment_config,
        split="test",
    )

    timestamps = [
        state.market.timestamp.date()
        for state in environment.feature_states
    ]

    assert timestamps
    assert min(timestamps) >= experiment_config.test.start
    assert max(timestamps) <= experiment_config.test.end


def test_build_environment_preserves_pre_split_history_for_warmup(
    repository,
    experiment_config,
):
    environment = build_experiment_environment(
        repository=repository,
        config=experiment_config,
        split="validation",
    )

    first_state = environment.feature_states[0]

    assert first_state.market.timestamp.date() >= (
        experiment_config.validation.start
    )

    assert len(environment.feature_states) > 0


def test_build_environment_rejects_non_positive_minimum_history(
    repository,
    experiment_config,
):
    with pytest.raises(
        ValueError,
        match="Minimum history must be positive",
    ):
        build_experiment_environment(
            repository=repository,
            config=experiment_config,
            split="training",
            minimum_history=0,
        )


def test_build_environment_rejects_non_positive_initial_cash(
    repository,
    experiment_config,
):
    with pytest.raises(
        ValueError,
        match="Initial cash must be positive",
    ):
        build_experiment_environment(
            repository=repository,
            config=experiment_config,
            split="training",
            initial_cash=0.0,
        )


def test_build_environment_uses_configured_execution_costs(
    repository,
    experiment_config,
):
    costs = ExecutionCosts(
        commission_rate=0.002,
        slippage_rate=0.003,
    )

    environment = build_experiment_environment(
        repository=repository,
        config=experiment_config,
        split="training",
        costs=costs,
    )

    assert environment.costs == costs


def test_build_environment_returns_trading_environment(
    repository,
    experiment_config,
):
    environment = build_experiment_environment(
        repository=repository,
        config=experiment_config,
        split="training",
    )

    assert isinstance(
        environment,
        TaurusTradingEnvironment,
    )

def test_build_environment_rejects_invalid_split(
    repository,
    experiment_config,
):
    with pytest.raises(
        ValueError,
        match="Unsupported experiment split",
    ):
        build_experiment_environment(
            repository=repository,
            config=experiment_config,
            split="invalid",  # type: ignore[arg-type]
        )