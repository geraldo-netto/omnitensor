from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from hypothesis import given
from hypothesis import strategies as st

import omnitensor.training.production_pipeline as pipeline
from omnitensor.training.production_pipeline import (
    export_torch_onnx_atomic,
    production_report,
    run_production_pipeline,
    sha256_file,
    validate_onnx_io,
    width_checked_dot,
)
from omnitensor.training.recipes import FetchedModelSource, ModelRecipeError, load_model_recipe


@dataclass(frozen=True)
class _Evidence:
    accepted: bool


_ACCEPTED = _Evidence(True)


def _run(
    model_path: Path,
    report_path: Path,
    *,
    evidence: _Evidence = _ACCEPTED,
    writer=None,
):
    def export() -> None:
        model_path.write_bytes(b"portable")

    def default_writer(path, payload, *, prefix):
        assert prefix == ".report-"
        path.write_text(payload["state"])

    return run_production_pipeline(
        model_path,
        report_path,
        conflict_detail="output exists",
        export=export,
        evaluate=lambda: evidence,
        accepted=lambda value: value.accepted,
        rejection_code="gate-failed",
        rejection_detail="gate refused output",
        report=lambda value: {"state": "accepted" if value.accepted else "rejected"},
        report_prefix=".report-",
        result=lambda model, report, value: (model, report, value),
        writer=writer or default_writer,
    )


def test_pipeline_runs_in_order_and_returns_family_result(tmp_path):
    model = tmp_path / "model.onnx"
    report = tmp_path / "report.json"
    evidence = _Evidence(True)
    events = []

    def export():
        events.append("export")
        model.write_bytes(b"portable")

    def evaluate():
        events.append("evaluate")
        return evidence

    def accepted(value):
        events.append("accepted")
        return value.accepted

    def build_report(value):
        events.append("report")
        return {"state": "accepted"}

    def writer(path, payload, *, prefix):
        events.append("writer")
        assert (payload, prefix) == ({"state": "accepted"}, ".report-")
        path.write_text(payload["state"])

    def result(model_path, report_path, value):
        events.append("result")
        return model_path, report_path, value

    produced = run_production_pipeline(
        model,
        report,
        conflict_detail="output exists",
        export=export,
        evaluate=evaluate,
        accepted=accepted,
        rejection_code="gate-failed",
        rejection_detail="gate refused output",
        report=build_report,
        report_prefix=".report-",
        result=result,
        writer=writer,
    )

    assert produced == (model, report, evidence)
    assert events == ["export", "evaluate", "accepted", "report", "writer", "result"]
    assert model.read_bytes() == b"portable"
    assert report.read_text() == "accepted"


def test_pipeline_prepares_export_after_conflict_check_and_before_cleanup(tmp_path):
    model = tmp_path / "model.onnx"
    report = tmp_path / "report.json"
    model.write_bytes(b"existing")
    calls = []

    def prepare_export():
        calls.append("prepare")
        raise KeyboardInterrupt

    with pytest.raises(ModelRecipeError):
        run_production_pipeline(
            model,
            report,
            conflict_detail="output exists",
            export=None,
            evaluate=lambda: _ACCEPTED,
            accepted=lambda value: value.accepted,
            rejection_code="gate-failed",
            rejection_detail="gate refused output",
            report=lambda value: {},
            report_prefix=".report-",
            result=lambda model, report, value: None,
            writer=lambda path, payload, prefix: None,
            prepare_export=prepare_export,
        )
    assert calls == []
    assert model.read_bytes() == b"existing"

    model.unlink()
    with pytest.raises(KeyboardInterrupt):
        run_production_pipeline(
            model,
            report,
            conflict_detail="output exists",
            export=None,
            evaluate=lambda: _ACCEPTED,
            accepted=lambda value: value.accepted,
            rejection_code="gate-failed",
            rejection_detail="gate refused output",
            report=lambda value: {},
            report_prefix=".report-",
            result=lambda model, report, value: None,
            writer=lambda path, payload, prefix: None,
            prepare_export=prepare_export,
        )
    assert calls == ["prepare"]


@pytest.mark.parametrize("occupied_name", ["model.onnx", "report.json"])
def test_pipeline_conflict_never_deletes_existing_output(tmp_path, occupied_name):
    model = tmp_path / "model.onnx"
    report = tmp_path / "report.json"
    occupied = tmp_path / occupied_name
    occupied.write_bytes(b"existing")

    with pytest.raises(ModelRecipeError) as caught:
        _run(model, report)

    assert (caught.value.code, caught.value.detail) == (
        "producer-conflict",
        "output exists",
    )
    assert occupied.read_bytes() == b"existing"


def test_pipeline_removes_model_after_gate_refusal(tmp_path):
    model = tmp_path / "model.onnx"
    report = tmp_path / "report.json"

    with pytest.raises(ModelRecipeError) as caught:
        _run(model, report, evidence=_Evidence(False))

    assert (caught.value.code, caught.value.detail) == (
        "gate-failed",
        "gate refused output",
    )
    assert not model.exists()
    assert not report.exists()


def test_pipeline_removes_both_outputs_after_base_exception(tmp_path):
    model = tmp_path / "model.onnx"
    report = tmp_path / "report.json"

    def interrupted_writer(path, payload, *, prefix):
        path.write_bytes(b"partial")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        _run(model, report, writer=interrupted_writer)

    assert not model.exists()
    assert not report.exists()


def test_pipeline_cleanup_is_model_first_and_first_unlink_error_wins():
    events = []

    class TrackedPath:
        def __init__(self, name, *, fail=False):
            self.name = name
            self.fail = fail

        def exists(self):
            return False

        def unlink(self, *, missing_ok):
            events.append((self.name, missing_ok))
            if self.fail:
                raise OSError(f"cannot unlink {self.name}")

    def interrupted():
        raise KeyboardInterrupt

    model = TrackedPath("model")
    report = TrackedPath("report")
    with pytest.raises(KeyboardInterrupt):
        run_production_pipeline(
            model,
            report,
            conflict_detail="output exists",
            export=interrupted,
            evaluate=lambda: _ACCEPTED,
            accepted=lambda value: value.accepted,
            rejection_code="gate-failed",
            rejection_detail="gate refused output",
            report=lambda value: {},
            report_prefix=".report-",
            result=lambda model, report, value: None,
            writer=lambda path, payload, prefix: None,
        )
    assert events == [("model", True), ("report", True)]

    events.clear()
    model = TrackedPath("model", fail=True)
    with pytest.raises(OSError, match="^cannot unlink model$"):
        run_production_pipeline(
            model,
            report,
            conflict_detail="output exists",
            export=interrupted,
            evaluate=lambda: _ACCEPTED,
            accepted=lambda value: value.accepted,
            rejection_code="gate-failed",
            rejection_detail="gate refused output",
            report=lambda value: {},
            report_prefix=".report-",
            result=lambda model, report, value: None,
            writer=lambda path, payload, prefix: None,
        )
    assert events == [("model", True)]


def test_report_envelope_preserves_exact_order_and_digests(tmp_path):
    recipe = load_model_recipe("model-recipes/clip-vit-b-32-image.json")
    source_root = tmp_path / "source"
    source_root.mkdir()
    receipt = source_root / "source-receipt.json"
    receipt.write_bytes(b'{"verified":true}')
    model = tmp_path / "portable.onnx"
    model.write_bytes(b"portable-clip")
    source = FetchedModelSource(recipe, source_root, receipt)
    native = {
        "gpu": {"status": "custom"},
        "npu": {"status": "custom"},
        "tpu": {"status": "custom"},
    }

    report = production_report(
        source,
        model,
        kind="clip-image-source-production",
        source_identity={"sourceModelSha256": recipe.model_source.sha256},
        holdout={"imageCount": 10},
        portable_source_gate={"accepted": True},
        native_targets=native,
    )

    assert tuple(report) == (
        "reportVersion",
        "kind",
        "recipeId",
        "recipeVersion",
        "recipeSha256",
        "sourceReceiptSha256",
        "sourceModelSha256",
        "portableModel",
        "holdout",
        "portableSourceGate",
        "nativeTargets",
        "cpuFallback",
    )
    assert tuple(report["portableModel"]) == (
        "format",
        "sha256",
        "tensorContract",
        "outputContract",
        "producer",
    )
    assert tuple(report["nativeTargets"]) == ("gpu", "npu", "tpu")
    assert report["sourceReceiptSha256"] == hashlib.sha256(receipt.read_bytes()).hexdigest()
    assert report["portableModel"]["sha256"] == hashlib.sha256(b"portable-clip").hexdigest()
    assert report["cpuFallback"] == "forbidden"


def test_report_envelope_supplies_generic_native_refusals(tmp_path):
    recipe = load_model_recipe("model-recipes/clip-vit-b-32-image.json")
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(b"receipt")
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    report = production_report(
        FetchedModelSource(recipe, tmp_path, receipt),
        model,
        kind="kind",
        source_identity={},
        holdout={},
        portable_source_gate={},
    )

    assert report["nativeTargets"] == {
        target: {
            "status": "unqualified",
            "reason": "no target compiler, native parity, or named-device evidence was run",
        }
        for target in ("gpu", "npu", "tpu")
    }


@pytest.mark.parametrize("payload", [b"", b"abc", bytes(range(256)) * 4097])
def test_sha256_file_matches_canonical_digest(tmp_path, payload):
    path = tmp_path / "payload"
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


@given(
    left=st.lists(st.integers(-100, 100), min_size=0, max_size=32),
    right=st.lists(st.integers(-100, 100), min_size=0, max_size=32),
)
def test_width_checked_dot_never_truncates(left, right):
    if len(left) != len(right):
        with pytest.raises(ValueError, match="^width mismatch$"):
            width_checked_dot(left, right, mismatch_detail="width mismatch")
    else:
        assert width_checked_dot(left, right, mismatch_detail="width mismatch") == sum(
            a * b for a, b in zip(left, right, strict=True)
        )


def _value(name: str, shape: tuple[int, ...]):
    dimensions = [SimpleNamespace(dim_value=value) for value in shape]
    tensor_type = SimpleNamespace(shape=SimpleNamespace(dim=dimensions))
    return SimpleNamespace(name=name, type=SimpleNamespace(tensor_type=tensor_type))


def _model(
    input_name: str = "image",
    input_shape: tuple[int, ...] = (1, 3),
    output_name: str = "embedding",
    output_shape: tuple[int, ...] = (1, 2),
):
    return SimpleNamespace(
        graph=SimpleNamespace(
            input=[_value(input_name, input_shape)],
            output=[_value(output_name, output_shape)],
        )
    )


@pytest.mark.parametrize(
    ("model", "detail"),
    [
        (_model(input_name="wrong"), "bad input"),
        (_model(output_name="wrong"), "bad output"),
        (_model(output_shape=(1, 3)), "bad shape"),
    ],
)
def test_onnx_io_validation_preserves_failure_precedence(model, detail):
    with pytest.raises(ModelRecipeError) as caught:
        validate_onnx_io(
            model,
            input_name="image",
            input_shape=(1, 3),
            input_detail="bad input",
            output_name="embedding",
            output_shape=(1, 2),
            output_detail="bad output",
            shape_detail="bad shape",
        )
    assert (caught.value.code, caught.value.detail) == ("producer-invalid", detail)


def test_onnx_io_validation_accepts_exact_contract():
    validate_onnx_io(
        _model(),
        input_name="image",
        input_shape=(1, 3),
        input_detail="bad input",
        output_name="embedding",
        output_shape=(1, 2),
        output_detail="bad output",
        shape_detail="bad shape",
    )


def test_torch_export_is_atomic_and_preserves_exact_arguments(tmp_path, monkeypatch):
    destination = tmp_path / "nested" / "model.onnx"
    calls = {}
    staged_paths = []
    real_mkstemp = tempfile.mkstemp

    def tracked_mkstemp(*, prefix, suffix, dir):
        assert (prefix, suffix, dir) == (".torch-", ".onnx", destination.parent)
        descriptor, name = real_mkstemp(prefix=prefix, suffix=suffix, dir=dir)
        staged_paths.append(Path(name))
        return descriptor, name

    def export(exportable, dummy, staged, **kwargs):
        calls["export"] = (exportable, dummy, staged, kwargs)
        staged.write_bytes(b"golden-onnx")

    graph = object()
    torch = SimpleNamespace(onnx=SimpleNamespace(export=export))
    onnx = SimpleNamespace(
        load=lambda path, *, load_external_data: calls.setdefault(
            "load", (path, load_external_data)
        )
        and graph,
        checker=SimpleNamespace(check_model=lambda value: calls.setdefault("checked", value)),
    )
    monkeypatch.setattr(pipeline.tempfile, "mkstemp", tracked_mkstemp)

    export_torch_onnx_atomic(
        destination,
        prefix=".torch-",
        build_arguments=lambda: ("module", "dummy"),
        input_names=("image",),
        output_names=("embedding",),
        validate=lambda value: calls.setdefault("validated", value),
        torch=torch,
        onnx=onnx,
    )

    assert destination.read_bytes() == b"golden-onnx"
    assert calls["export"][0:2] == ("module", "dummy")
    assert calls["export"][2].parent == destination.parent
    assert calls["export"][3] == {
        "input_names": ["image"],
        "output_names": ["embedding"],
        "opset_version": 17,
        "do_constant_folding": True,
        "dynamo": False,
    }
    assert calls["load"] == (calls["export"][2], False)
    assert calls["checked"] is graph
    assert calls["validated"] is graph
    assert staged_paths and not any(path.exists() for path in staged_paths)


@pytest.mark.parametrize("failure", ["export", "checker", "validator"])
def test_torch_export_removes_stage_after_base_exception(tmp_path, monkeypatch, failure):
    destination = tmp_path / "model.onnx"
    staged_paths = []
    real_mkstemp = tempfile.mkstemp

    def tracked_mkstemp(*, prefix, suffix, dir):
        descriptor, name = real_mkstemp(prefix=prefix, suffix=suffix, dir=dir)
        staged_paths.append(Path(name))
        return descriptor, name

    def export(exportable, dummy, staged, **kwargs):
        staged.write_bytes(b"partial")
        if failure == "export":
            raise KeyboardInterrupt

    def check_model(graph):
        if failure == "checker":
            raise KeyboardInterrupt

    def validate(graph):
        if failure == "validator":
            raise KeyboardInterrupt

    monkeypatch.setattr(pipeline.tempfile, "mkstemp", tracked_mkstemp)
    with pytest.raises(KeyboardInterrupt):
        export_torch_onnx_atomic(
            destination,
            prefix=".torch-",
            build_arguments=lambda: ("module", "dummy"),
            input_names=("image",),
            output_names=("embedding",),
            validate=validate,
            torch=SimpleNamespace(onnx=SimpleNamespace(export=export)),
            onnx=SimpleNamespace(
                load=lambda path, *, load_external_data: object(),
                checker=SimpleNamespace(check_model=check_model),
            ),
        )

    assert not destination.exists()
    assert staged_paths and not any(path.exists() for path in staged_paths)
