import numpy as np
import pytest

from datetime import date, datetime, timezone

from taurus.data.schemas import BarInterval, PriceBar
from taurus.training.experiment_config import (
    DateRange,
    TrainingExperimentConfig,
)
from taurus.training.training_environment import (
    TrainingEnvironment,
    build_training_environment,
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


def test_build_training_environment_returns_training_wrapper():
    repository = FakeRepository(
        {
            "NVDA": make_bars("NVDA"),
            "SPY": make_bars("SPY"),
        }
    )

    config = TrainingExperimentConfig(
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

    result = build_training_environment(
        repository=repository,
        config=config,
    )

    assert isinstance(
        result,
        TrainingEnvironment,
    )


@pytest.mark.parametrize(
    "execution_version",
    ["same-close-v1", "next-open-v2"],
)
@pytest.mark.parametrize(
    "observation_version, portfolio_input_count",
    [
        ("initial-capital-v1", 3),
        ("allocation-v2", 2)
    ],
)
def test_training_environment_contains_only_training_dates(
    observation_version,
    portfolio_input_count,
    execution_version,
):
    repository = FakeRepository(
        {
            "NVDA": make_bars("NVDA"),
            "SPY": make_bars("SPY"),
        }
    )

    config = TrainingExperimentConfig(
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

    result = build_training_environment(
        repository=repository,
        config=config,
        observation_version=observation_version,
        execution_version=execution_version,
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
    assert min(timestamps) >= config.training.start
    assert max(timestamps) <= config.training.end

    environment = result.environment
    underlying = environment.taurus_environment

    assert underlying.observation_version == observation_version
    assert underlying.execution_version == execution_version

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
    observation, _, _, _, info = environment.step(1)

    assert observation.shape == (expected_size,)
    assert np.isfinite(observation).all()
    assert environment.observation_space.contains(observation)

    trade = info["trade"]
    assert trade is not None

    if execution_version == "next-open-v2":
        execution_state = underlying.feature_states[1]
        reference_price = execution_state.open_price
        assert reference_price is not None
    else:
        execution_state = underlying.feature_states[0]
        reference_price = execution_state.market.close

    assert trade.timestamp == execution_state.market.timestamp
    assert trade.price == pytest.approx(
        reference_price * (1.0 + underlying.costs.slippage_rate)
    )