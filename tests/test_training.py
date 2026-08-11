from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from omnitensor.plugins.recorder import FeatureRow, TelemetryRecorder
from omnitensor.preparation import file_digest
from omnitensor.training.cli import _feature_values, install_main, record_main, train_main
from omnitensor.training.contracts import TrainingError, TrainingReport, TrainingSpec
from omnitensor.training.forecast import ForecastTrainer, forecast_dataset
from omnitensor.training.installation import (
    _validated_targets,
    available_targets,
    install_training,
)


class FakeExporter:
    def export(self, model, destination):
        self.model = model
        Path(destination).write_bytes(b"portable model")


def spec(**changes):
    values = {
        "profile_id": "resource-scheduler",
        "artifact_id": "local-resource-forecast",
        "artifact_version": "1.0.0",
        "feature_names": ("load", "queue"),
        "target_feature": "load",
        "window": 3,
        "horizon": 1,
    }
    values.update(changes)
    return TrainingSpec(**values)


def rows(count=40):
    return tuple(
        FeatureRow(
            "resource-scheduler",
            index,
            {"load": float(index), "queue": float(index % 4)},
        )
        for index in range(count)
    )


def record_history(root, count=40):
    recorder = TelemetryRecorder(root)
    for row in rows(count):
        recorder.record(row.profile_id, row.features, row.observed_at_ms)


def test_dataset_uses_explicit_time_order_window_and_future_target():
    recorded = rows(12)
    dataset = forecast_dataset(spec(), recorded)
    corpus = hashlib.sha256()
    for row in recorded:
        corpus.update(
            json.dumps(row.document(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        corpus.update(b"\n")

    assert dataset.inputs[0] == (0.0, 0.0, 1.0, 1.0, 2.0, 2.0)
    assert dataset.targets[0] == 3.0
    assert len(dataset.inputs) == 9
    assert dataset.corpus_sha256 == corpus.hexdigest()


def test_dataset_accepts_exactly_one_window():
    dataset = forecast_dataset(spec(), rows(4))

    assert dataset.inputs == ((0.0, 0.0, 1.0, 1.0, 2.0, 2.0),)
    assert dataset.targets == (3.0,)


def test_dataset_refuses_missing_features_instead_of_training_on_implicit_zero():
    broken = (*rows(4), FeatureRow("resource-scheduler", 5, {"load": 5.0}), *rows(12)[6:])

    with pytest.raises(TrainingError) as captured:
        forecast_dataset(spec(), broken)

    assert captured.value.code == "features-missing"
    assert captured.value.detail == "recorded row 5 has no queue"


def test_dataset_refuses_too_little_history():
    with pytest.raises(TrainingError) as captured:
        forecast_dataset(spec(), rows(3))

    assert captured.value.code == "insufficient-history"
    assert captured.value.detail == "at least 4 recorded rows are required, got 3"


@pytest.mark.parametrize(
    ("targets", "detail"),
    [
        ((), "at least one target is required"),
        ("gpu", "at least one target is required"),
        (b"gpu", "at least one target is required"),
        (("gpu", "gpu"), "targets must be unique"),
        (
            ("tpu",),
            "TPU export needs a matching fully-int8 TFLite graph and Edge TPU compiler",
        ),
        (("cpu", "remote"), "unsupported target: cpu"),
    ],
)
def test_target_validation_has_stable_failures(targets, detail):
    with pytest.raises(TrainingError) as captured:
        _validated_targets(targets)

    assert captured.value.code == "targets-invalid"
    assert captured.value.detail == detail


def test_target_validation_returns_canonical_lane_order():
    assert _validated_targets(("gpu", "npu")) == ("npu", "gpu")


@given(
    count=st.integers(min_value=3, max_value=80),
    window=st.integers(min_value=1, max_value=8),
    horizon=st.integers(min_value=1, max_value=8),
)
def test_window_count_and_target_offset_hold_for_bounded_history(count, window, horizon):
    assume(count >= window + horizon)
    dataset = forecast_dataset(spec(window=window, horizon=horizon), rows(count))

    assert len(dataset.inputs) == count - window - horizon + 1
    assert dataset.targets[0] == float(window + horizon - 1)
    assert len(dataset.inputs[0]) == window * 2


@given(
    pairs=st.lists(
        st.tuples(
            st.from_regex(r"[a-z][a-z0-9_]{0,10}", fullmatch=True),
            st.floats(allow_nan=False, allow_infinity=False, width=32),
        ),
        min_size=1,
        max_size=20,
        unique_by=lambda pair: pair[0],
    )
)
def test_feature_argument_parser_preserves_unique_finite_numbers(pairs):
    parsed = _feature_values([f"{name}={value!r}" for name, value in pairs])

    assert tuple(parsed) == tuple(name for name, _value in pairs)
    assert parsed == {name: float(value) for name, value in pairs}


@pytest.mark.parametrize(
    "changes",
    [
        {"artifact_id": "bad id"},
        {"artifact_id": "a" * 117},
        {"artifact_version": "latest"},
        {"profile_id": "../../escaped"},
        {"profile_id": "UPPERCASE"},
        {"feature_names": ()},
        {"feature_names": ["load"]},
        {"feature_names": ("load", "load")},
        {"target_feature": "queue"},
        {"window": 0},
        {"horizon": 129},
    ],
)
def test_training_spec_refuses_ambiguous_or_unbounded_contracts(changes):
    with pytest.raises(TrainingError):
        spec(**changes)


def test_training_freezes_model_report_and_quality(tmp_path):
    record_history(tmp_path / "records")
    exporter = FakeExporter()

    report = ForecastTrainer(exporter).train(spec(), tmp_path / "records", tmp_path / "fit")

    assert report.samples == 37
    assert report.quality["useful"] is True
    assert exporter.model.input_width == 6
    frozen = json.loads((tmp_path / "fit/training-report.json").read_text())
    assert frozen["model"]["sha256"] == file_digest(tmp_path / "fit/model.onnx")
    assert frozen["tensorContract"] == {
        "inputs": [{"shape": [1, 6], "dtype": "float32", "layout": "NC"}]
    }


def test_forecast_recipe_refuses_profiles_needing_different_models(tmp_path):
    other = spec(profile_id="hardware-health")

    with pytest.raises(TrainingError, match="recipe-unsupported"):
        ForecastTrainer(FakeExporter()).train(other, tmp_path, tmp_path / "fit")


def test_report_load_rechecks_portable_bytes(tmp_path):
    record_history(tmp_path / "records")
    ForecastTrainer(FakeExporter()).train(spec(), tmp_path / "records", tmp_path / "fit")
    report_path = tmp_path / "fit/training-report.json"

    assert TrainingReport.load(report_path).spec == spec()
    (tmp_path / "fit/model.onnx").write_bytes(b"changed")
    with pytest.raises(TrainingError, match="model-invalid"):
        TrainingReport.load(report_path)


def test_report_load_refuses_contract_drift(tmp_path):
    record_history(tmp_path / "records")
    ForecastTrainer(FakeExporter()).train(spec(), tmp_path / "records", tmp_path / "fit")
    report_path = tmp_path / "fit/training-report.json"
    document = json.loads(report_path.read_text())
    document["tensorContract"]["inputs"][0]["shape"] = [1, 999]
    report_path.write_text(json.dumps(document))

    with pytest.raises(TrainingError, match="tensor contract disagrees"):
        TrainingReport.load(report_path)


def test_real_onnx_export_is_valid_and_reproducible(tmp_path):
    onnx = pytest.importorskip("onnx")
    record_history(tmp_path / "records")

    first = ForecastTrainer().train(spec(), tmp_path / "records", tmp_path / "first")
    second = ForecastTrainer().train(spec(), tmp_path / "records", tmp_path / "second")

    onnx.checker.check_model(onnx.load(tmp_path / "first/model.onnx"))
    assert first.model_sha256 == second.model_sha256


def fake_prepared_training(tmp_path, monkeypatch):
    record_history(tmp_path / "records")
    report = ForecastTrainer(FakeExporter()).train(spec(), tmp_path / "records", tmp_path / "fit")
    compiled = tmp_path / "compiled/model.param"
    compiled.parent.mkdir()
    compiled.write_bytes(b"graph")
    compiled.with_suffix(".bin").write_bytes(b"weights")

    class Converted:
        param = compiled

    monkeypatch.setattr(
        "omnitensor.training.installation.convert_to_ncnn", lambda *a, **k: Converted()
    )
    monkeypatch.setattr(
        "omnitensor.training.installation._tool", lambda name: Path(f"/tools/{name}")
    )
    return report


def test_install_compiles_installs_and_binds_one_logical_model(tmp_path, monkeypatch):
    fake_prepared_training(tmp_path, monkeypatch)

    installed = install_training(
        tmp_path / "fit/training-report.json",
        targets=("gpu",),
        artifact_root=tmp_path / "artifacts",
        bindings_root=tmp_path / "bindings",
    )

    assert installed.variants[0].artifact_id == "local-resource-forecast-gpu"
    manifest = json.loads(installed.binding_path.read_text())
    assert manifest["requirements"]["accelerator"] == "gpu"
    assert manifest["requirements"]["acceleratorPreference"] == ["gpu"]
    [bound] = manifest["requirements"]["models"]
    assert bound["format"] == "ncnn"
    assert bound["tensorContract"] == spec().tensor_contract
    assert (tmp_path / "artifacts/local-resource-forecast-gpu/1.0.0/model.param").is_file()


@pytest.mark.skipif(
    bool(os.environ.get("MUTANT_UNDER_TEST")),
    reason="native compiler and Vulkan smoke runs once outside mutation workers",
)
def test_real_portable_fit_compiles_and_installs_for_present_toolchains(tmp_path):
    targets = available_targets()
    if not targets:
        pytest.skip("no producer compiler extra is installed")
    record_history(tmp_path / "records")
    ForecastTrainer().train(spec(), tmp_path / "records", tmp_path / "fit")

    installed = install_training(
        tmp_path / "fit/training-report.json",
        targets=targets,
        artifact_root=tmp_path / "artifacts",
        bindings_root=tmp_path / "bindings",
    )

    assert {variant.accelerator for variant in installed.variants} == set(targets)
    assert all(variant.installed_at.is_file() for variant in installed.variants)
    assert installed.binding_path.is_file()
    gpu = next(
        (variant for variant in installed.variants if variant.accelerator == "gpu"),
        None,
    )
    if gpu is not None:
        from omnitensor.executors.vulkan import VulkanGpuExecutor

        executor = VulkanGpuExecutor(True)
        if executor.availability().available:
            result = executor.run(
                str(gpu.installed_at),
                [[[10.0, 2.0, 11.0, 3.0, 12.0, 0.0]]],
            )
            assert math.isfinite(result.outputs[0][0][0])


def test_install_refuses_tpu_without_claiming_portability_it_cannot_build(tmp_path):
    record_history(tmp_path / "records")
    ForecastTrainer(FakeExporter()).train(spec(), tmp_path / "records", tmp_path / "fit")
    with pytest.raises(TrainingError, match="fully-int8"):
        install_training(
            tmp_path / "fit/training-report.json",
            targets=("tpu",),
            artifact_root=tmp_path / "artifacts",
            bindings_root=tmp_path / "bindings",
        )


def test_train_cli_reports_next_install_command(tmp_path, capsys, monkeypatch):
    report = TrainingReport(spec(), 10, "0" * 64, "1" * 64, {"useful": True})

    def train(_self, _spec, _records, output):
        Path(output).mkdir(parents=True)
        return report

    monkeypatch.setattr("omnitensor.training.cli.ForecastTrainer.train", train)

    code = train_main(
        [
            "--profile",
            "resource-scheduler",
            "--id",
            "local-resource-forecast",
            "--version",
            "1.0.0",
            "--features",
            "load,queue",
            "--target",
            "load",
            "--window",
            "3",
            "--output-dir",
            str(tmp_path / "fit"),
        ]
    )

    assert code == 0
    assert "omnitensor-install-trained-model" in json.loads(capsys.readouterr().out)["next"]


def test_record_cli_appends_explicit_numeric_features(tmp_path, capsys):
    code = record_main(
        [
            "--profile",
            "resource-scheduler",
            "--feature",
            "load=0.5",
            "--feature",
            "queue=3",
            "--observed-at-ms",
            "42",
            "--records-root",
            str(tmp_path / "records"),
        ]
    )

    assert code == 0
    assert json.loads(capsys.readouterr().out)["f"] == {"load": 0.5, "queue": 3.0}
    [row] = TelemetryRecorder(tmp_path / "records").rows("resource-scheduler")
    assert row.observed_at_ms == 42


def test_record_cli_refuses_duplicate_or_non_numeric_features(tmp_path, capsys):
    code = record_main(
        [
            "--profile",
            "resource-scheduler",
            "--feature",
            "load=1",
            "--feature",
            "load=two",
            "--records-root",
            str(tmp_path),
        ]
    )

    assert code == 1
    assert "recording failed" in capsys.readouterr().err


def test_install_cli_reports_stable_failure_without_traceback(tmp_path, capsys):
    code = install_main([str(tmp_path / "missing.json"), "--targets", "gpu"])

    assert code == 1
    assert "installation failed" in capsys.readouterr().err
