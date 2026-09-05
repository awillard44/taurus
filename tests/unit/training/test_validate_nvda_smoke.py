import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "validate_nvda_smoke.py"
)

spec = importlib.util.spec_from_file_location(
    "validate_nvda_smoke",
    SCRIPT_PATH,
)
validation_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation_script)


def test_parse_args_selects_named_run():
    args = validation_script.parse_args(
        ["--run-name", "nvda_ppo_example"]
    )

    assert args.run_name == "nvda_ppo_example"


def test_parse_args_requires_run_name():
    with pytest.raises(SystemExit) as error:
        validation_script.parse_args([])

    assert error.value.code == 2


def test_resolve_model_path_selects_requested_run(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    for run_name in ("run_a", "run_b"):
        directory = tmp_path / "artifacts" / "training" / run_name
        directory.mkdir(parents=True)
        (directory / "model.zip").touch()

    result = validation_script.resolve_model_path("run_b")

    assert result == Path("artifacts/training/run_b/model.zip")


@pytest.mark.parametrize(
    "run_name",
    ["", " ", ".", "..", "../other", "nested/run", r"nested\run"],
)
def test_resolve_model_path_rejects_invalid_names(run_name):
    with pytest.raises(
        ValueError,
        match="single directory name",
    ):
        validation_script.resolve_model_path(run_name)


def test_resolve_model_path_rejects_missing_model(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)

    directory = tmp_path / "artifacts" / "training" / "empty_run"
    directory.mkdir(parents=True)

    with pytest.raises(
        FileNotFoundError,
        match="Model artifact not found",
    ):
        validation_script.resolve_model_path("empty_run")