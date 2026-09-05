import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def report_script(monkeypatch):
    scripts_directory = (
        Path(__file__).resolve().parents[3] / "scripts"
    )
    monkeypatch.syspath_prepend(str(scripts_directory))

    spec = importlib.util.spec_from_file_location(
        "compare_nvda_runs",
        scripts_directory / "compare_nvda_runs.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_accepts_multiple_run_names(report_script):
    args = report_script.parse_args(
        ["--run-names", "smoke_run", "longer_run"]
    )

    assert args.run_names == ["smoke_run", "longer_run"]


def test_report_requires_run_names(report_script):
    with pytest.raises(SystemExit) as error:
        report_script.parse_args([])

    assert error.value.code == 2


def test_summary_includes_final_value_in_drawdown(report_script):
    result = SimpleNamespace(
        final_portfolio_value=600.0,
        total_return=-0.40,
        transitions=("first", "second"),
        step_records=(
            SimpleNamespace(
                portfolio_value=1000.0,
                long_probability=0.20,
                policy_entropy=0.60,
                exposure_ratio=0.0,
            ),
            SimpleNamespace(
                portfolio_value=1200.0,
                long_probability=0.60,
                policy_entropy=0.30,
                exposure_ratio=0.50,
            ),
            SimpleNamespace(
                portfolio_value=900.0,
                long_probability=0.70,
                policy_entropy=0.0,
                exposure_ratio=1.0,
            ),
        ),
    )

    summary = report_script.summarize_ppo(result)

    assert summary == pytest.approx({
        "final_value": 600.0,
        "return": -0.40,
        "drawdown": 0.50,
        "long_probability": 0.50,
        "entropy": 0.30,
        "exposure": 0.50,
        "transitions": 2,
    })


def test_summary_rejects_empty_evaluation(report_script):
    result = SimpleNamespace(step_records=())

    with pytest.raises(
        ValueError,
        match="Cannot summarize an empty evaluation",
    ):
        report_script.summarize_ppo(result)