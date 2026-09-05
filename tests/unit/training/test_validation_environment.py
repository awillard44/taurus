import numpy as np
import pytest

from datetime import date, datetime, timezone

from taurus.data.schemas import BarInterval, PriceBar
from taurus.training.experiment_config import (
    DateRange,
    TrainingExperimentConfig,
)
from taurus.training.validation_environment import (
    ValidationEnvironment,
    build_validation_environment,
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
) -> list[PriceBar]:
    bars = []
    index = 0

    for year in range(2021, 2027):
        for month in range(1, 13):
            for day in range(1, 21):
                price = 100.0 + index

                bars.append(
                    PriceBar(
                        symbol=symbol,
                        timestamp=datetime(
                            year,
                            month,
                            day,
                            tzinfo=timezone.utc,
                        ),
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


def make_config() -> TrainingExperimentConfig:
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


def make_repository() -> FakeRepository:
    return FakeRepository(
        {
            "NVDA": make_bars("NVDA"),
            "SPY": make_bars("SPY"),
        }
    )


def test_build_validation_environment_returns_validation_wrapper():
    result = build_validation_environment(
        repository=make_repository(),
        config=make_config(),
    )

    assert isinstance(
        result,
        ValidationEnvironment,
    )


@pytest.mark.parametrize(
    "observation_version, portfolio_input_count",
    [
        ("initial-capital-v1", 3),
        ("allocation-v2", 2),
    ],
)
def test_validation_environment_contains_only_validation_dates(
    observation_version,
    portfolio_input_count,
):
    config = make_config()

    result = build_validation_environment(
        repository=make_repository(),
        config=config,
        observation_version=observation_version,
    )

    timestamps = [
        state.market.timestamp.date()
        for state in (
            result.environment
            .taurus_environment
            .feature_states
        )
    ]

    assert timestamps
    assert min(timestamps) >= config.validation.start
    assert max(timestamps) <= config.validation.end

    environment = result.environment
    underlying = environment.taurus_environment

    assert underlying.observation_version == observation_version

    expected_size = (
        len(underlying.feature_states[0].features)
        + portfolio_input_count
    )

    assert environment.observation_space.shape == (expected_size,)

    observation, _ = environment.reset()

    assert observation.shape == (expected_size,)
    assert np.isfinite(observation).all()
    assert environment.observation_space.contains(observation)

    # Target-position action 1 means LONG
    observation, _, _, _, _ = environment.step(1)

    assert observation.shape == (expected_size,)
    assert np.isfinite(observation).all()
    assert environment.observation_space.contains(observation)