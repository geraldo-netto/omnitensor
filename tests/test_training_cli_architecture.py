from __future__ import annotations

import ast
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import omnitensor.training.cli as facade
import omnitensor.training.runner as runner
from omnitensor.training import (
    forecast_cli,
    forecast_client,
    forecast_contracts,
    training_install_cli,
    training_record_cli,
    training_train_cli,
)

ROOT = Path(__file__).parents[1]
LEGACY_RUNNER_MODULE = "omnitensor.training.runner"


def test_runner_facade_preserves_extracted_contract_and_client_identity():
    assert runner.ForecastRunError is forecast_contracts.ForecastRunError
    assert runner.ForecastClient is forecast_contracts.ForecastClient
    assert runner.SocketForecastClient is forecast_client.SocketForecastClient

    for owner in (
        forecast_contracts.ForecastRunError,
        forecast_contracts.ForecastClient,
        forecast_client.SocketForecastClient,
    ):
        assert owner.__module__ == LEGACY_RUNNER_MODULE
        assert pickle.loads(pickle.dumps(owner)) is owner

    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        pickle.loads(pickle.dumps(forecast_contracts.ForecastRunError("code", "detail")))


def test_cli_facade_preserves_private_helpers_and_live_dependencies(monkeypatch):
    assert facade._feature_values is training_record_cli.feature_values
    assert facade._features is training_train_cli.features
    assert facade._targets is training_install_cli.targets
    for helper in (facade._feature_values, facade._features, facade._targets):
        assert helper.__module__ == "omnitensor.training.cli"
        assert pickle.loads(pickle.dumps(helper)) is helper

    trainer = object()
    recorder = object()
    loader = object()
    feature_parser = object()
    monkeypatch.setattr(facade, "ForecastTrainer", trainer)
    monkeypatch.setattr(facade, "TelemetryRecorder", recorder)
    monkeypatch.setattr(facade, "load_forecast_binding", loader)
    monkeypatch.setattr(facade, "_features", feature_parser)

    assert facade._training_dependencies().forecast_trainer is trainer
    assert facade._training_dependencies().feature_parser is feature_parser
    assert facade._record_dependencies().recorder_type is recorder
    assert facade.forecast_main.__module__ == "omnitensor.training.cli"
    dependencies = forecast_cli.ForecastCliDependencies(binding_loader=loader)
    assert dependencies.binding_loader is loader


def test_forecast_training_resolves_the_live_facade_feature_parser(monkeypatch, tmp_path, capsys):
    observed = []

    class Trainer:
        def train(self, spec, _records, _output):
            observed.append(spec.feature_names)
            return SimpleNamespace(samples=1, quality={})

    monkeypatch.setattr(facade, "ForecastTrainer", Trainer)
    monkeypatch.setattr(facade, "_features", lambda _text: ("patched",))

    assert (
        facade.train_main(
            [
                "--profile",
                "resource-scheduler",
                "--id",
                "local-resource-forecast",
                "--version",
                "1.0.0",
                "--features",
                "ignored",
                "--target",
                "patched",
                "--window",
                "1",
                "--output-dir",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert observed == [("patched",)]
    assert "omnitensor-install-trained-model" in capsys.readouterr().out


def test_trainer_dispatch_is_closed_ordered_and_read_only():
    assert tuple(training_train_cli.TRAINER_DISPATCH) == (
        "forecast",
        "storage",
        "network",
        "build",
        "hardware",
        "desktop",
    )
    with pytest.raises(TypeError):
        training_train_cli.TRAINER_DISPATCH["unknown"] = lambda *_args: 0
    with pytest.raises(training_train_cli.TrainingError) as failure:
        training_train_cli.dispatch_trainer("unknown")
    assert failure.value.code == "trainer-unknown"
    assert failure.value.detail == "no trainer command is named unknown"

    assert training_train_cli._default_outputs() == {
        "forecast": training_train_cli.DEFAULT_OUTPUT_ROOT,
        "storage": training_train_cli.DEFAULT_STORAGE_OUTPUT,
        "network": training_train_cli.DEFAULT_NETWORK_OUTPUT,
        "build": training_train_cli.DEFAULT_BUILD_OUTPUT,
        "hardware": training_train_cli.DEFAULT_HARDWARE_OUTPUT,
        "desktop": training_train_cli.DEFAULT_DESKTOP_OUTPUT,
    }


def test_cli_leaves_import_before_facade_without_backedges():
    modules = (
        "forecast_contracts",
        "forecast_client",
        "training_record_cli",
        "training_train_cli",
        "training_install_cli",
        "forecast_cli",
    )
    script = "\n".join(
        [
            "import importlib, sys",
            *(f"importlib.import_module('omnitensor.training.{name}')" for name in modules),
            "assert 'omnitensor.training.cli' not in sys.modules",
        ]
    )
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src")},
        check=True,
    )

    for name in modules:
        tree = ast.parse((ROOT / f"src/omnitensor/training/{name}.py").read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, ast.ImportFrom) and node.module == "cli" for node in ast.walk(tree)
        )


def test_console_targets_stay_on_facade_and_numeric_promotion_stays_separate():
    # The trainer entry points ship with the trainers, in their own distribution.
    project = (ROOT / "packaging" / "omnitensor-training" / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    targets = (
        "record_main",
        "snapshot_record_main",
        "train_main",
        "storage_train_main",
        "network_train_main",
        "build_train_main",
        "hardware_train_main",
        "desktop_train_main",
        "desktop_revoke_main",
        "install_main",
        "forecast_main",
    )
    for target in targets:
        assert f'"omnitensor.training.cli:{target}"' in project

    assert not hasattr(facade, "promote_main")
    assert not hasattr(training_install_cli, "promote_main")
    install_source = Path(training_install_cli.__file__).read_text(encoding="utf-8")
    assert "numeric_promotion" not in install_source


def test_the_training_sdist_carries_the_sources_the_wheel_target_expects():
    """OMNI-0447: an sdist of two files built a wheel with zero modules.

    Both build targets must name paths that exist in a checkout *and* inside
    an unpacked sdist, and neither may escape the distribution root.
    """
    import tomllib

    packaging_root = ROOT / "packaging" / "omnitensor-training"
    project = tomllib.loads((packaging_root / "pyproject.toml").read_text(encoding="utf-8"))
    build = project["tool"]["hatch"]["build"]["targets"]

    wheel_sources = build["wheel"]["force-include"]
    assert wheel_sources == {
        "src/omnitensor/training": "omnitensor/training",
        "model-recipes": "omnitensor/model-recipes",
    }
    for source in wheel_sources:
        # In the checkout these are symlinks onto the repository's own copies;
        # in an sdist they are the real directories the sdist target put there.
        resolved = (packaging_root / source).resolve()
        assert resolved.is_dir()
        assert resolved == (ROOT / source).resolve()

    sdist = build["sdist"]["force-include"]
    assert sdist == {
        "../../src/omnitensor/training": "src/omnitensor/training",
        "../../model-recipes": "model-recipes",
    }
    for destination in (*wheel_sources.values(), *sdist.values()):
        assert not destination.startswith("..")

    # The distribution ships its own licence rather than reaching out of its
    # root for one, which put a `../../LICENSE` member inside the sdist.
    assert project["project"]["license-files"] == ["LICENSE"]
    assert (packaging_root / "LICENSE").is_file()
    assert not (packaging_root / "LICENSE").is_symlink()
    assert (packaging_root / "LICENSE").read_text(encoding="utf-8") == (
        (ROOT / "LICENSE").read_text(encoding="utf-8")
    )
