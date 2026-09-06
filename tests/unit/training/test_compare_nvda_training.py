import importlib.util
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def comparison_script(monkeypatch):
    scripts_directory = (
        Path(__file__).resolve().parents[3] / "scripts"
    )
    monkeypatch.syspath_prepend(str(scripts_directory))

    spec = importlib.util.spec_from_file_location(
        "compare_nvda_training",
        scripts_directory / "compare_nvda_training.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "observation_version",
    ["initial-capital-v1", "allocation-v2"],
)
def test_main_forwards_observation_version(
    comparison_script,
    monkeypatch,
    observation_version,
):
    script = comparison_script
    model_path = Path("artifacts/training/example/model.zip")
    model = object()
    repository = object()
    target_environment = object()
    baseline_environment = object()

    monkeypatch.setattr(
        script,
        "resolve_model_path",
        Mock(return_value=model_path),
    )

    metadata_loader = Mock(return_value=observation_version)
    monkeypatch.setattr(
        script,
        "load_observation_version",
        metadata_loader,
    )

    monkeypatch.setattr(
        script,
        "SQLitePriceBarRepository",
        Mock(return_value=repository),
    )
    monkeypatch.setattr(
        script,
        "PPO",
        SimpleNamespace(load=Mock(return_value=model)),
    )

    training_builder = Mock(
        return_value=SimpleNamespace(
            environment=target_environment
        )
    )
    baseline_builder = Mock(return_value=baseline_environment)

    monkeypatch.setattr(
        script,
        "build_training_environment",
        training_builder,
    )
    monkeypatch.setattr(
        script,
        "build_experiment_environment",
        baseline_builder,
    )

    compatibility_check = Mock()
    monkeypatch.setattr(
        script,
        "check_model_observation_compatibility",
        compatibility_check,
    )

    record = SimpleNamespace(
        timestamp=datetime(2023, 7, 5),
        asset_price=100.0,
        portfolio_value=1000.0,
        long_probability=0.60,
        policy_entropy=0.67,
        confidence_margin=0.20,
        exposure_ratio=0.0,
    )

    result = SimpleNamespace(
        initial_portfolio_value=1000.0,
        final_portfolio_value=1000.0,
        total_return=0.0,
        total_reward=0.0,
        steps=1,
        cash_count=1,
        long_count=0,
        step_records=(record,),
        transitions=(),
    )

    evaluator = Mock(return_value=result)
    monkeypatch.setattr(script, "evaluate_ppo_environment", evaluator)

    monkeypatch.setattr(
        script,
        "run_agent_episode",
        Mock(
            return_value=SimpleNamespace(
                final_portfolio_value=1000.0,
                total_return=0.0,
                total_reward=0.0,
                steps=1,
                max_drawdown=0.0,
            )
        ),
    )

    sensitivity = Mock(return_value=(0.60, 0.70))
    monkeypatch.setattr(
        script,
        "compare_portfolio_state_probabilities",
        sensitivity,
    )

    script.main(["--run-name", "example"])

    metadata_loader.assert_called_once_with(
        model_path.with_name("metadata.json")
    )

    training_builder.assert_called_once()
    assert (
        training_builder.call_args.kwargs["observation_version"]
        == observation_version
    )

    baseline_builder.assert_called_once()
    assert (
        baseline_builder.call_args.kwargs["observation_version"]
        == observation_version
    )
    assert baseline_builder.call_args.kwargs["split"] == "training"

    compatibility_check.assert_called_once_with(
        model=model,
        environment=target_environment,
    )
    evaluator.assert_called_once_with(
        model=model,
        environment=target_environment,
    )

    calls = sensitivity.call_args_list
    assert calls

    # Confirm that both sensitivity sections actually ran.
    assert any("account_value" not in call.kwargs for call in calls)
    assert {
        call.kwargs["account_value"]
        for call in calls
        if "account_value" in call.kwargs
    } == {500.0, 1000.0, 2000.0, 4000.0}

    for call in calls:
        assert call.kwargs["model"] is model
        assert (
            call.kwargs["observation_version"]
            == observation_version
        )