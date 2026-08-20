from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import sample_manifest
from hypothesis import assume, given
from hypothesis import strategies as st

from omnitensor.plugins.recorder import FeatureRow, TelemetryRecorder
from omnitensor.preparation import file_digest
from omnitensor.registry import Workload
from omnitensor.training.cli import _feature_values, install_main, record_main, train_main
from omnitensor.training.compilers import (
    CompilationRequest,
    CompiledTarget,
    CompilerCapability,
    CompilerError,
)
from omnitensor.training.contracts import TrainingError, TrainingReport, TrainingSpec
from omnitensor.training.forecast import ForecastTrainer, forecast_dataset
from omnitensor.training.installation import (
    _binding_manifest,
    available_targets,
    install_training,
    validated_targets,
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


@pytest.mark.parametrize("timestamp", [0, 1])
def test_dataset_refuses_duplicate_or_decreasing_observation_time(timestamp):
    recorded = list(rows(4))
    recorded[2] = FeatureRow(
        "resource-scheduler",
        timestamp,
        {"load": 2.0, "queue": 2.0},
    )

    with pytest.raises(TrainingError) as captured:
        forecast_dataset(spec(), tuple(recorded))

    assert captured.value.code == "observations-unordered"
    assert captured.value.detail == "recorded row timestamps must increase strictly"


@pytest.mark.parametrize(
    ("targets", "detail"),
    [
        ((), "at least one target is required"),
        ("gpu", "at least one target is required"),
        (b"gpu", "at least one target is required"),
        (("gpu", "gpu"), "targets must be unique"),
        (("cpu", "remote"), "unsupported target: cpu"),
    ],
)
def test_target_validation_has_stable_failures(targets, detail):
    with pytest.raises(TrainingError) as captured:
        validated_targets(targets)

    assert captured.value.code == "targets-invalid"
    assert captured.value.detail == detail


def test_target_validation_returns_canonical_lane_order():
    assert validated_targets(("tpu", "gpu", "npu")) == ("gpu", "npu", "tpu")


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


def test_training_spec_refuses_unhashable_features_with_the_stable_contract():
    with pytest.raises(TrainingError) as caught:
        spec(feature_names=(["load"],))

    assert (caught.value.code, caught.value.detail) == (
        "features-invalid",
        "feature names must be bounded strings",
    )


@pytest.mark.parametrize("feature_names", [("", ""), (1, 1)])
def test_duplicate_invalid_features_keep_the_uniqueness_precedence(feature_names):
    with pytest.raises(TrainingError) as caught:
        spec(feature_names=feature_names)

    assert (caught.value.code, caught.value.detail) == (
        "features-invalid",
        "feature names must be unique",
    )


@pytest.mark.parametrize(
    ("document", "detail"),
    [
        (None, "training spec must be an object"),
        ({"featureNames": []}, "training spec fields are invalid"),
    ],
)
def test_training_spec_document_shape_refusals_are_exact(document, detail):
    with pytest.raises(TrainingError) as caught:
        TrainingSpec.from_document(document)

    assert (caught.value.code, caught.value.detail) == ("report-invalid", detail)


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


def test_training_off_the_event_loop_produces_the_same_report(tmp_path):
    """The solve and the export both block; an async caller gets a thread."""
    record_history(tmp_path / "records")

    report = asyncio.run(
        ForecastTrainer(FakeExporter()).train_async(spec(), tmp_path / "records", tmp_path / "fit")
    )

    assert report.samples == 37
    assert (tmp_path / "fit/model.onnx").is_file()


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

    class FakeCompiler:
        def __init__(self, accelerator):
            self.accelerator = accelerator
            self.observed = []

        def capability(self):
            return CompilerCapability(self.accelerator, True, "test compiler")

        def incompatibility(self, _source_format, _fully_quantized):
            return None

        def compile(self, _request, output_dir):
            self.observed.append((_request, output_dir))
            output_dir.mkdir(parents=True, exist_ok=True)
            suffix = ".param" if self.accelerator == "gpu" else ".xml"
            primary = output_dir / f"model{suffix}"
            primary.write_bytes(b"graph")
            primary.with_suffix(".bin").write_bytes(b"weights")
            model_format = "ncnn" if self.accelerator == "gpu" else "openvino"
            return CompiledTarget(self.accelerator, primary, model_format, False)

    compilers = (FakeCompiler("npu"), FakeCompiler("gpu"))
    monkeypatch.setattr(
        "omnitensor.training.installation.default_target_compilers",
        lambda _resolve: compilers,
    )
    return report, compilers


def test_install_compiles_installs_and_binds_one_logical_model(tmp_path, monkeypatch):
    _report, compilers = fake_prepared_training(tmp_path, monkeypatch)

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
    assert tuple(bound)[-6:] == (
        "fullyQuantized",
        "minimumCompilerVersion",
        "minimumRuntimeVersion",
        "tensorContract",
        "featureContract",
        "outputContract",
    )
    assert bound["format"] == "ncnn"
    assert bound["fullyQuantized"] is False
    assert bound["minimumCompilerVersion"] == "0.0.0"
    assert bound["minimumRuntimeVersion"] == "0.0.0"
    assert bound["tensorContract"] == spec().tensor_contract
    assert bound["featureContract"] == {
        "version": 1,
        "recipe": "forecast-v1",
        "featureNames": ["load", "queue"],
        "targetFeature": "load",
        "window": 3,
        "horizon": 1,
        "observationOrder": "oldest-first",
        "flattenOrder": "observations-then-features",
    }
    assert (tmp_path / "artifacts/local-resource-forecast-gpu/1.0.0/model.param").is_file()
    assert compilers[0].observed == []
    assert compilers[1].observed == [
        (
            CompilationRequest(tmp_path / "fit/model.onnx", "onnx", (1, 6), False),
            tmp_path / "fit/compiled/gpu",
        )
    ]


def test_binding_refuses_unknown_profile_with_stable_failure(tmp_path, monkeypatch):
    monkeypatch.setattr("omnitensor.training.installation.load_workloads", lambda _root: {})
    report = TrainingReport(spec(), 1, "0" * 64, "1" * 64, {})

    with pytest.raises(TrainingError) as captured:
        _binding_manifest(report, {"gpu": {}}, bundled_root=tmp_path)

    assert captured.value.code == "profile-unknown"
    assert captured.value.detail == "no bundled profile is named resource-scheduler"


def test_binding_replaces_legacy_models_without_aliasing_inputs(tmp_path, monkeypatch):
    legacy_model = {"id": "legacy"}
    base = sample_manifest(
        "resource-scheduler",
        model=legacy_model,
        models=[legacy_model],
    )
    monkeypatch.setattr(
        "omnitensor.training.installation.load_workloads",
        lambda _root: {"resource-scheduler": Workload("resource-scheduler", base)},
    )
    monkeypatch.setattr(
        "omnitensor.training.installation.validate_workload_document", lambda _manifest: []
    )
    report = TrainingReport(spec(), 1, "0" * 64, "1" * 64, {})
    variant = {"featureContract": {"featureNames": ["load", "queue"]}}

    binding = _binding_manifest(report, {"gpu": variant}, bundled_root=tmp_path)
    variant["featureContract"]["featureNames"].append("memory")
    base["requirements"]["models"][0]["id"] = "changed"

    assert base["requirements"]["model"] is legacy_model
    assert "model" not in binding["requirements"]
    assert binding["requirements"]["models"] == [
        {"featureContract": {"featureNames": ["load", "queue"]}}
    ]


def test_binding_reports_every_semantic_violation(tmp_path, monkeypatch):
    base = sample_manifest("resource-scheduler")
    monkeypatch.setattr(
        "omnitensor.training.installation.load_workloads",
        lambda _root: {"resource-scheduler": Workload("resource-scheduler", base)},
    )
    monkeypatch.setattr(
        "omnitensor.training.installation.validate_workload_document",
        lambda _manifest: ["first violation", "second violation"],
    )
    report = TrainingReport(spec(), 1, "0" * 64, "1" * 64, {})

    with pytest.raises(TrainingError) as captured:
        _binding_manifest(report, {"gpu": {}}, bundled_root=tmp_path)

    assert captured.value.code == "binding-invalid"
    assert captured.value.detail == "first violation; second violation"


def test_install_reloads_identical_feature_semantics_for_every_native_lane(tmp_path, monkeypatch):
    fake_prepared_training(tmp_path, monkeypatch)

    installed = install_training(
        tmp_path / "fit/training-report.json",
        targets=("gpu", "npu"),
        artifact_root=tmp_path / "artifacts",
        bindings_root=tmp_path / "bindings",
    )

    from omnitensor.registry import load_workloads

    reloaded = load_workloads(tmp_path / "bindings")["resource-scheduler"]
    assert [variant.accelerator for variant in installed.variants] == ["gpu", "npu"]
    assert [model["format"] for model in reloaded.models] == ["ncnn", "openvino"]
    assert all(model["featureContract"] == spec().feature_contract for model in reloaded.models)


def test_install_translates_compiler_catalog_and_compilation_failures(tmp_path):
    record_history(tmp_path / "records")
    ForecastTrainer(FakeExporter()).train(spec(), tmp_path / "records", tmp_path / "fit")

    class Compiler:
        accelerator = "gpu"

        def capability(self):
            return CompilerCapability("gpu", True, "available")

        def incompatibility(self, _source_format, _fully_quantized):
            return None

        def compile(self, _request, _output):
            raise CompilerError("compilation-failed", "compiler diagnostic")

    with pytest.raises(TrainingError) as duplicate:
        install_training(
            tmp_path / "fit/training-report.json",
            targets=("gpu",),
            artifact_root=tmp_path / "artifacts",
            bindings_root=tmp_path / "bindings",
            compilers=(Compiler(), Compiler()),
        )
    assert duplicate.value.code == "compiler-invalid"
    assert duplicate.value.detail == "duplicate compiler target: gpu"

    with pytest.raises(TrainingError) as failed:
        install_training(
            tmp_path / "fit/training-report.json",
            targets=("gpu",),
            artifact_root=tmp_path / "artifacts",
            bindings_root=tmp_path / "bindings",
            compilers=(Compiler(),),
        )
    assert failed.value.code == "compilation-failed"
    assert failed.value.detail == "compiler diagnostic"


def test_install_refuses_a_catalog_missing_the_requested_provider(tmp_path):
    record_history(tmp_path / "records")
    ForecastTrainer(FakeExporter()).train(spec(), tmp_path / "records", tmp_path / "fit")

    with pytest.raises(TrainingError) as captured:
        install_training(
            tmp_path / "fit/training-report.json",
            targets=("gpu",),
            artifact_root=tmp_path / "artifacts",
            bindings_root=tmp_path / "bindings",
            compilers=(),
        )

    assert captured.value.code == "compiler-missing"
    assert captured.value.detail == "no compiler provider for gpu"


def test_install_wires_explicit_roots_and_variant_identity_exactly(tmp_path, monkeypatch):
    record_history(tmp_path / "records")
    ForecastTrainer(FakeExporter()).train(spec(), tmp_path / "records", tmp_path / "fit")
    compiler = SimpleNamespace(accelerator="gpu")
    artifact = SimpleNamespace(reference=SimpleNamespace(id="compiled-id", format="ncnn"))
    compiled = CompiledTarget("gpu", tmp_path / "native/model.param", "ncnn", False)
    observed = []

    def compile_and_prepare(report, request, provider, output):
        observed.append(("compile", report, request, provider, output))
        return compiled, artifact

    def install(artifact_value, root):
        observed.append(("install", artifact_value, root))
        return tmp_path / "landed/model.param"

    def fragment(report, artifact_value, fully_quantized):
        observed.append(("fragment", report, artifact_value, fully_quantized))
        return {"native": True}

    def binding(report, variants, *, bundled_root):
        observed.append(("binding", report, variants, bundled_root))
        return {"binding": True}

    def write(path, document, *, prefix):
        observed.append(("write", path, document, prefix))

    monkeypatch.setattr(
        "omnitensor.training.installation._compile_and_prepare", compile_and_prepare
    )
    monkeypatch.setattr("omnitensor.training.installation.install_prepared", install)
    monkeypatch.setattr("omnitensor.training.installation._model_fragment", fragment)
    monkeypatch.setattr("omnitensor.training.installation._binding_manifest", binding)
    monkeypatch.setattr("omnitensor.training.installation.write_json_atomic", write)

    installed = install_training(
        tmp_path / "fit/training-report.json",
        targets=("gpu",),
        artifact_root=tmp_path / "artifact-root",
        bindings_root=tmp_path / "binding-root",
        build_root=tmp_path / "build-root",
        bundled_root=tmp_path / "bundled-root",
        compilers=(compiler,),
    )

    report = TrainingReport.load(tmp_path / "fit/training-report.json")
    assert observed == [
        (
            "compile",
            report,
            CompilationRequest(tmp_path / "fit/model.onnx", "onnx", (1, 6), False),
            compiler,
            tmp_path / "build-root",
        ),
        ("install", artifact, tmp_path / "artifact-root"),
        ("fragment", report, artifact, False),
        (
            "binding",
            report,
            {"gpu": {"native": True}},
            tmp_path / "bundled-root",
        ),
        (
            "write",
            tmp_path / "binding-root/resource-scheduler/manifest.json",
            {"binding": True},
            ".model-binding-",
        ),
    ]
    assert installed == type(installed)(
        "resource-scheduler",
        tmp_path / "binding-root/resource-scheduler/manifest.json",
        (
            type(installed.variants[0])(
                "gpu",
                "compiled-id",
                "ncnn",
                tmp_path / "landed/model.param",
                {"native": True},
            ),
        ),
    )


@given(
    feature_names=st.lists(
        st.from_regex(r"[a-z][a-z0-9-]{0,15}", fullmatch=True),
        min_size=1,
        max_size=8,
        unique=True,
    ),
    window=st.integers(min_value=1, max_value=16),
    horizon=st.integers(min_value=1, max_value=128),
)
def test_feature_contract_preserves_bounded_declared_order(feature_names, window, horizon):
    assume(len(feature_names) * window <= 512)
    declared = spec(
        feature_names=tuple(feature_names),
        target_feature=feature_names[0],
        window=window,
        horizon=horizon,
    ).feature_contract

    assert declared["featureNames"] == feature_names
    assert declared["targetFeature"] == feature_names[0]
    assert declared["window"] == window
    assert declared["horizon"] == horizon


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
