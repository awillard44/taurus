import pytest
import torch
import numpy as np

import taurus.training.ppo_evaluator as ppo_evaluator

from datetime import date, datetime, timezone
from types import SimpleNamespace

from taurus.data.schemas import BarInterval, PriceBar
from taurus.training.experiment_config import (
    DateRange,
    TrainingExperimentConfig,
)
from taurus.training.ppo_evaluator import (
    PPOEvaluationResult,
    evaluate_ppo,
    check_model_observation_compatibility,
)
from taurus.training.validation_environment import (
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


class FakePolicy:
    def obs_to_tensor(
            self,
            observation,
    ):
        tensor = torch.as_tensor(
            observation,
            dtype=torch.float32,
        ).unsqueeze(0)

        return tensor, False

    def get_distribution(
            self,
            _observation_tensor,
    ):
        probabilities = torch.tensor(
            [[0.75, 0.25]],
            dtype=torch.float32,
        )

        return SimpleNamespace(
            distribution=SimpleNamespace(
                probs=probabilities,
            )
        )

class FakePPOModel:
    def __init__(self):
        self.predict_calls = 0
        self.deterministic_values = []
        self.policy = FakePolicy()

    def predict(
        self,
        _observation,
        deterministic=False,
    ):
        self.predict_calls += 1
        self.deterministic_values.append(
            deterministic
        )

        return 0, None

    def learn(self, *args, **kwargs):
        raise AssertionError(
            "Validation must never call learn()."
        )


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


def make_validation_environment():
    repository = FakeRepository(
        {
            "NVDA": make_bars("NVDA"),
            "SPY": make_bars("SPY"),
        }
    )

    return build_validation_environment(
        repository=repository,
        config=make_config(),
    )


def test_evaluate_ppo_returns_evaluation_result():
    model = FakePPOModel()
    environment = make_validation_environment()

    result = evaluate_ppo(
        model=model,
        validation_environment=environment,
    )

    assert isinstance(
        result,
        PPOEvaluationResult,
    )


def test_evaluate_ppo_uses_deterministic_predictions():
    model = FakePPOModel()
    environment = make_validation_environment()

    evaluate_ppo(
        model=model,
        validation_environment=environment,
    )

    assert model.predict_calls > 0
    assert all(model.deterministic_values)


def test_evaluate_ppo_cash_policy_preserves_portfolio_value():
    model = FakePPOModel()
    environment = make_validation_environment()

    result = evaluate_ppo(
        model=model,
        validation_environment=environment,
    )

    assert result.initial_portfolio_value == 1_000.0
    assert result.final_portfolio_value == 1_000.0
    assert result.total_return == 0.0
    assert result.total_reward == 0.0


def test_evaluate_ppo_runs_until_environment_terminates():
    model = FakePPOModel()
    environment = make_validation_environment()

    expected_steps = (
        len(
            environment.environment
            .taurus_environment
            .feature_states
        ) 
        - 1
    )

    result = evaluate_ppo(
        model=model,
        validation_environment=environment,
    )

    assert result.steps == expected_steps
    assert model.predict_calls == expected_steps


def test_evaluate_ppo_counts_target_positions():
    model = FakePPOModel()
    environment = make_validation_environment()

    result = evaluate_ppo(
        model=model,
        validation_environment=environment,
    )

    assert result.cash_count == result.steps
    assert result.long_count == 0

def test_evaluate_ppo_records_no_transitions_for_constant_target():
    model = FakePPOModel()
    environment = make_validation_environment()

    result = evaluate_ppo(
        model=model,
        validation_environment=environment,
    )

    assert result.transitions == ()

def test_evaluate_ppo_records_every_step():
    model = FakePPOModel()
    environment = make_validation_environment()

    result = evaluate_ppo(
        model=model,
        validation_environment=environment,
    )

    assert len(result.step_records) == result.steps


def test_evaluate_ppo_records_policy_probabilities():
    model = FakePPOModel()
    environment = make_validation_environment()

    result = evaluate_ppo(
        model=model,
        validation_environment=environment,
    )

    first_record = result.step_records[0]

    assert first_record.cash_probability == pytest.approx(
        0.75
    )

    assert first_record.long_probability == pytest.approx(
        0.25
    )

    assert first_record.confidence_margin == pytest.approx(
        0.50
    )


def test_evaluate_ppo_records_feature_state():
    model = FakePPOModel()
    environment = make_validation_environment()

    result = evaluate_ppo(
        model=model,
        validation_environment=environment,
    )

    first_record = result.step_records[0]

    assert first_record.feature_values
    assert first_record.normalized_feature_values

    raw_keys = {
        key
        for key, _ in first_record.feature_values
    }

    normalized_keys = {
        key
        for key, _ in first_record.normalized_feature_values
    }

    assert raw_keys == normalized_keys


def test_evaluate_ppo_records_constant_cash_target():
    model = FakePPOModel()
    environment = make_validation_environment()

    result = evaluate_ppo(
        model=model,
        validation_environment=environment,
    )

    assert all(
        record.target == 0
        for record in result.step_records
    )

    assert all(
        not record.target_changed
        for record in result.step_records
    )

@pytest.mark.parametrize("input_count", [12, 13])
def test_observation_compatibility_accepts_matching_shapes(
    input_count,
):
    model = SimpleNamespace(
        observation_space=SimpleNamespace(shape=(input_count,))
    )
    environment = SimpleNamespace(
        observation_space=SimpleNamespace(shape=(input_count,))
    )

    check_model_observation_compatibility(
        model=model,
        environment=environment,
    )


@pytest.mark.parametrize(
    "model_shape, environment_shape, observation_version",
    [
        ((13,), (12,), "allocation-v2"),
        ((12,), (13,), "initial-capital-v1"),
    ],
)
def test_observation_compatibility_rejects_mismatched_shapes(
    model_shape,
    environment_shape,
    observation_version,
):
    model = SimpleNamespace(
        observation_space=SimpleNamespace(shape=model_shape)
    )
    environment = SimpleNamespace(
        observation_space=SimpleNamespace(shape=environment_shape),
        taurus_environment=SimpleNamespace(
            observation_version=observation_version
        ),
    )

    with pytest.raises(
        ValueError,
        match="Model expects observation shape",
    ):
        check_model_observation_compatibility(
            model=model,
            environment=environment,
        )

@pytest.mark.parametrize(
    "observation_version, expected_cash, expected_long",
    [
        (
            "initial-capital-v1",
            [0.01, 0.60, 4.0, 0.0, 4.0],
            [0.01, 0.60, 0.0, 4.0, 4.0],
        ),
        (
            "allocation-v2",
            [0.01, 0.60, 1.0, 0.0],
            [0.01, 0.60, 0.0, 1.0],
        ),
    ],
)
def test_portfolio_sensitivity_uses_selected_observation_version(
    monkeypatch,
    observation_version,
    expected_cash,
    expected_long,
):
    observations = []

    def capture_probabilities(model, observation):
        observations.append(observation.copy())

        if len(observations) == 1:
            return 0.80, 0.20

        return 0.30, 0.70

    monkeypatch.setattr(
        ppo_evaluator,
        "_get_target_probabilities",
        capture_probabilities,
    )

    record = SimpleNamespace(
        portfolio_value=1000.0,
        asset_price=100.0,
        feature_values=(
            ("return_1", 0.01),
            ("rsi_14", 60.0),
        ),
    )

    result = ppo_evaluator.compare_portfolio_state_probabilities(
        model=object(),
        record=record,
        initial_portfolio_value=1000.0,
        account_value=4000.0,
        observation_version=observation_version,
    )

    assert result == pytest.approx((0.20, 0.70))
    assert len(observations) == 2

    np.testing.assert_allclose(observations[0], expected_cash)
    np.testing.assert_allclose(observations[1], expected_long)

@pytest.mark.parametrize("account_value", [0.0, -100.0])
def test_portfolio_sensitivity_rejects_nonpositive_account_value(
    account_value,
):
    record = SimpleNamespace(
        portfolio_value=1000.0,
        asset_price=100.0,
        feature_values=(("return_1", 0.01),),
    )

    with pytest.raises(
        ValueError,
        match="Account value must be positive",
    ):
        ppo_evaluator.compare_portfolio_state_probabilities(
            model=object(),
            record=record,
            initial_portfolio_value=1000.0,
            account_value=account_value,
            observation_version="allocation-v2",
        )