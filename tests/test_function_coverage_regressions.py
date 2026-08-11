from __future__ import annotations

import asyncio
import builtins
import hashlib
import io
import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, call

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor import acceptance
from omnitensor.acceptance import Check
from omnitensor.control import _integral_weight
from omnitensor.jobs import _request_id_from_text
from omnitensor.plugins.artifact_installation import (
    ArtifactInstallationError,
    _copy_digested,
)
from omnitensor.plugins.artifacts import ArtifactReference
from omnitensor.plugins.build_ingestion import BuildMetadataIngestor
from omnitensor.plugins.cancellation import JobCancellationToken
from omnitensor.plugins.collection import (
    MAX_COLLECTED_ITEMS,
    BoundedCollector,
    CollectionError,
    SourceSnapshot,
)
from omnitensor.plugins.document_ingestion import DocumentIngestor
from omnitensor.plugins.ingestion import OptedInRootScanner
from omnitensor.plugins.ipc import (
    FRAME_FORMAT_VERSION,
    IPCFrame,
    IPCProtocolError,
    WorkerMessageType,
    parse_ready,
)
from omnitensor.plugins.supervisor import _cancel_monitor, _startup_failure
from omnitensor.plugins.triggers import SourceStatus
from omnitensor.service import OmniTensorInterface, OmniTensorService
from omnitensor.tensorref import _digest_of
from omnitensor.training.cli import _features, _targets, install_main
from omnitensor.training.contracts import (
    TrainingError,
    TrainingReport,
    TrainingSpec,
    _quality_is_finite,
)
from omnitensor.training.forecast import OnnxLinearExporter
from omnitensor.training.installation import (
    InstalledTraining,
    InstalledVariant,
    _tool,
)


def training_spec() -> TrainingSpec:
    return TrainingSpec(
        "resource-scheduler",
        "local-forecast",
        "1.0.0",
        ("load",),
        "load",
        2,
    )


def test_default_acceptance_report_runs_every_host_probe(monkeypatch, tmp_path):
    calls = []

    def passing(name):
        def probe(*args, **kwargs):
            calls.append(call(name, *args, **kwargs))
            if name == "check_plugin_discovery":
                args[0]()
            return Check(name, True, name)

        return probe

    for name in (
        "check_executable",
        "check_service",
        "check_schemas",
        "check_workload_catalog",
        "check_plugin_discovery",
        "check_isolation",
        "check_bus",
        "check_snapshot",
        "check_backends",
        "check_confinement",
        "check_applet_contract",
        "check_applet",
    ):
        monkeypatch.setattr(acceptance, name, passing(name))
    monkeypatch.setattr(acceptance, "bundled_workloads_path", lambda: tmp_path / "bundled")

    from omnitensor.plugins import discovery, identity, loading

    metadata = (object(),)
    identities = SimpleNamespace(plugins=(object(),))
    workers = (object(),)

    def discover(**kwargs):
        calls.append(call("discover_plugin_metadata", **kwargs))
        return metadata

    def resolve(plugins):
        calls.append(call("resolve_plugin_identities", plugins))
        return identities

    def external(plugins):
        calls.append(call("external_worker_specs", plugins))
        return workers

    monkeypatch.setattr(discovery, "discover_plugin_metadata", discover)
    monkeypatch.setattr(
        identity,
        "resolve_plugin_identities",
        resolve,
    )
    monkeypatch.setattr(loading, "external_worker_specs", external)
    applet = tmp_path / "applet"
    applet.mkdir()
    checksums = tmp_path / "SHA256SUMS"
    checksums.write_text(f"{'0' * 64}  applet.js\n", encoding="utf-8")

    report = acceptance.build_default_report(
        snapshot_path=tmp_path / "state.json",
        now_ms=1,
        applet_root=applet,
        applet_checksums=checksums,
        service=object(),
        bus=object(),
    )

    assert report.ok is True
    assert {check.name for check in report.checks} == {
        "check_executable",
        "check_service",
        "check_schemas",
        "check_workload_catalog",
        "check_plugin_discovery",
        "check_isolation",
        "check_bus",
        "check_snapshot",
        "check_backends",
        "check_confinement",
        "check_applet_contract",
        "check_applet",
    }
    bundled = tmp_path / "bundled"
    service = report.checks[1]
    assert service.name == "check_service"
    assert calls == [
        call("check_executable"),
        call("check_service", ANY),
        call("check_schemas"),
        call("check_workload_catalog", bundled),
        call("check_plugin_discovery", ANY),
        call("discover_plugin_metadata", bundled_root=bundled),
        call("discover_plugin_metadata", bundled_root=bundled),
        call("resolve_plugin_identities", metadata),
        call("external_worker_specs", identities.plugins),
        call("check_isolation", workers),
        call("check_bus", ANY),
        call("check_snapshot", tmp_path / "state.json", now_ms=1),
        call("check_backends"),
        call("check_confinement"),
        call("check_applet_contract", applet, tmp_path / "state.json"),
        call("check_applet", applet, {"applet.js": "0" * 64}),
    ]


def test_confinement_probe_reports_available_and_blocked_hosts(monkeypatch):
    from omnitensor.plugins import seccomp

    monkeypatch.setattr(seccomp, "confinement_error", lambda: "")
    assert acceptance.check_confinement() == Check(
        "confinement", True, "plugin workers can be confined on this host"
    )
    monkeypatch.setattr(seccomp, "confinement_error", lambda: "unsupported architecture")
    assert acceptance.check_confinement() == Check(
        "confinement",
        False,
        "unsupported architecture; external plugin workers will not start, and "
        "--no-seccomp is the deliberate override",
    )


def test_ingestor_root_ports_expose_normalized_opt_in_paths(tmp_path):
    expected = (tmp_path.resolve(),)
    assert BuildMetadataIngestor([tmp_path]).roots == expected
    assert DocumentIngestor([tmp_path]).roots == expected


def test_cancellation_token_exposes_its_stable_job_identity():
    assert JobCancellationToken("job-1").job_id == "job-1"


class SampleCollector(BoundedCollector):
    label = "sample"

    def identity_of(self, item):
        return item.identity

    def document_of(self, item):
        return {"id": item.identity}


def test_base_collector_hooks_fail_explicitly_and_default_churn_is_stable():
    subject = object.__new__(BoundedCollector)
    with pytest.raises(NotImplementedError):
        subject.identity_of(object())
    with pytest.raises(NotImplementedError):
        subject.document_of(object())
    assert subject.changed_fields("same", "same") == []
    assert subject.changed_fields("before", "after") == ["state"]


@pytest.mark.parametrize(
    "snapshot",
    [
        object(),
        SourceSnapshot("ready", 1, ()),
        SourceSnapshot(SourceStatus.READY, -1, ()),
        SourceSnapshot(SourceStatus.READY, 1, tuple(range(MAX_COLLECTED_ITEMS + 1))),
        SourceSnapshot(SourceStatus.READY, 1, (object(),)),
        SourceSnapshot(SourceStatus.READY, 1, (SimpleNamespace(identity="bad id"),)),
    ],
)
def test_base_collector_rejects_every_invalid_source_boundary(snapshot):
    subject = object.__new__(SampleCollector)
    with pytest.raises(CollectionError) as caught:
        subject._validate_snapshot(snapshot)
    assert caught.value.code == "source-invalid"


def test_base_collector_accepts_the_exact_item_limit():
    subject = object.__new__(SampleCollector)
    items = tuple(SimpleNamespace(identity=f"item-{index}") for index in range(MAX_COLLECTED_ITEMS))
    subject._validate_snapshot(SourceSnapshot(SourceStatus.READY, 0, items))


def test_native_collector_clocks_return_epoch_milliseconds():
    from omnitensor.plugins.network_collection import _now_ms as network_now
    from omnitensor.plugins.peripheral_collection import _now_ms as peripheral_now

    before = time.time_ns() // 1_000_000
    observed = (network_now(), peripheral_now())
    after = time.time_ns() // 1_000_000
    assert all(before <= value <= after for value in observed)


def test_worker_claims_stdout_before_plugin_code(monkeypatch):
    from omnitensor.plugins import worker

    events = []
    channel = io.BytesIO()

    class Stream:
        def __init__(self, descriptor):
            self.descriptor = descriptor

        def flush(self):
            events.append("flush")

        def fileno(self):
            return self.descriptor

    monkeypatch.setattr(worker, "sys", SimpleNamespace(stdout=Stream(10), stderr=Stream(20)))
    monkeypatch.setattr(
        worker,
        "os",
        SimpleNamespace(
            dup=lambda descriptor: events.append(("dup", descriptor)) or 11,
            fdopen=lambda descriptor, mode: events.append(("fdopen", descriptor, mode)) or channel,
            dup2=lambda source, target: events.append(("dup2", source, target)),
        ),
    )

    assert worker.claim_frame_channel() is channel
    assert events == ["flush", ("dup", 10), ("fdopen", 11, "wb"), ("dup2", 20, 10)]


def test_dbus_result_method_delegates_to_runtime():
    class Runtime:
        async def job_result_text(self, request):
            return f"result:{request}"

    interface = OmniTensorInterface(Runtime())
    method = OmniTensorInterface.GetJobResult.__wrapped__
    assert asyncio.run(method(interface, "job")) == "result:job"


def test_service_progress_and_plugin_artifact_lookup_use_runtime_ports():
    progress = []
    service = object.__new__(OmniTensorService)
    service.jobs = SimpleNamespace(note_progress=lambda *arguments: progress.append(arguments))
    service._workloads = {}
    entry = {
        "id": "plugin-model",
        "version": "1.0.0",
        "format": "ncnn",
        "sha256": "a" * 64,
    }
    plugin = SimpleNamespace(manifest={"plugin": {"artifacts": [entry]}})
    service._plugin_runtime = SimpleNamespace(
        snapshot=SimpleNamespace(catalog=SimpleNamespace(plugins=(plugin,)))
    )

    service._note_job_progress("job", "infer", 0.5, "running")
    reference = service._declared_reference("plugin-model")

    assert progress == [("job", "infer", 0.5, "running")]
    assert reference == ArtifactReference("plugin-model", "1.0.0", "ncnn", "a" * 64)
    assert service._declared_reference("missing") is None


def test_service_artifact_lookup_continues_after_nonmatching_workload():
    expected = ArtifactReference("wanted", "1.0.0", "ncnn", "b" * 64)
    service = object.__new__(OmniTensorService)
    service._workloads = {
        "first": SimpleNamespace(models=({"id": "other", "version": "1.0.0", "format": "ncnn"},)),
        "second": SimpleNamespace(
            models=(
                {
                    "id": "wanted",
                    "version": "1.0.0",
                    "format": "ncnn",
                    "sha256": "b" * 64,
                },
            ),
            manifest={},
        ),
    }
    service._plugin_runtime = SimpleNamespace(
        snapshot=SimpleNamespace(catalog=SimpleNamespace(plugins=()))
    )

    assert service._declared_reference("wanted") == expected


def test_tensor_digest_uses_the_same_bounded_reader_as_tensor_loading(tmp_path):
    path = tmp_path / "tensor.bin"
    path.write_bytes(b"tensor")
    assert _digest_of(path, 6) == hashlib.sha256(b"tensor").hexdigest()


def test_target_companion_copy_removes_partial_output_on_overflow(tmp_path):
    source = tmp_path / "weights.bin"
    destination = tmp_path / "installed.bin"
    source.write_bytes(b"too large")

    with pytest.raises(ArtifactInstallationError, match="companion-too-large"):
        _copy_digested(source, destination, 2)

    assert not destination.exists()


def test_target_companion_copy_accepts_the_exact_byte_limit(tmp_path):
    source = tmp_path / "weights.bin"
    destination = tmp_path / "installed.bin"
    payload = b"exact"
    source.write_bytes(payload)

    assert _copy_digested(source, destination, len(payload)) == hashlib.sha256(payload).hexdigest()
    assert destination.read_bytes() == payload


@pytest.mark.parametrize(
    ("text", "limit", "expected"),
    [
        (None, 100, "invalid"),
        ('{"requestId":"ok"}', 2, "invalid"),
        ("not-json", 100, "invalid"),
        ('{"requestId":"ok"}', 100, "ok"),
    ],
)
def test_request_identity_recovery_is_bounded_and_never_raises(text, limit, expected):
    assert _request_id_from_text(text, limit) == expected


def test_request_identity_recovery_accepts_exact_bytes_and_rejects_json_nan():
    valid = '{"requestId":"ok"}'
    assert _request_id_from_text(valid, len(valid.encode())) == "ok"
    assert _request_id_from_text('{"requestId":NaN}', 100) == "invalid"


@given(
    text=st.one_of(st.none(), st.integers(), st.text(max_size=300)),
    limit=st.integers(min_value=0, max_value=256),
)
def test_request_identity_recovery_fuzz_stays_closed_and_bounded(text, limit):
    result = _request_id_from_text(text, limit)
    assert result == "invalid" or (
        1 <= len(result) <= 120
        and all(
            character.isascii() and (character.isalnum() or character in "._-")
            for character in result
        )
    )


@given(
    request_id=st.from_regex(r"[A-Za-z0-9._-]{1,120}", fullmatch=True),
)
def test_request_identity_recovery_fuzz_round_trips_valid_identifiers(request_id):
    text = json.dumps({"requestId": request_id})
    assert _request_id_from_text(text, len(text.encode("utf-8"))) == request_id


@pytest.mark.parametrize(
    ("frame", "code", "detail"),
    [
        (
            IPCFrame(FRAME_FORMAT_VERSION + 1, WorkerMessageType.READY, None, {"pluginId": "p"}),
            "frame-version-incompatible",
            f"expected {FRAME_FORMAT_VERSION}; received {FRAME_FORMAT_VERSION + 1}",
        ),
        (
            IPCFrame(FRAME_FORMAT_VERSION, WorkerMessageType.HELLO, None, {"pluginId": "p"}),
            "invalid-ready",
            "expected an uncorrelated ready frame",
        ),
        (
            IPCFrame(FRAME_FORMAT_VERSION, WorkerMessageType.READY, "request", {"pluginId": "p"}),
            "invalid-ready",
            "expected an uncorrelated ready frame",
        ),
        (
            IPCFrame(
                FRAME_FORMAT_VERSION,
                WorkerMessageType.READY,
                None,
                {"pluginId": "p", "x": 1},
            ),
            "invalid-ready",
            "ready fields do not match the contract",
        ),
        (
            IPCFrame(FRAME_FORMAT_VERSION, WorkerMessageType.READY, None, {"pluginId": "other"}),
            "plugin-identity-mismatch",
            "expected p; received 'other'",
        ),
    ],
)
def test_ready_frame_rejects_each_protocol_mismatch(frame, code, detail):
    with pytest.raises(IPCProtocolError) as caught:
        parse_ready(frame, "p")
    assert caught.value.code == code
    assert caught.value.detail == detail


def test_opted_in_root_check_fails_closed_when_resolution_fails(tmp_path):
    scanner = OptedInRootScanner([tmp_path])

    class Unresolvable:
        def resolve(self):
            raise OSError("gone")

    assert scanner._within_roots(Unresolvable()) is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [(True, None), (2, 2), (2.0, 2), (2.5, None), ("2", None)],
)
def test_policy_weights_coerce_only_schema_integers(value, expected):
    assert _integral_weight(value) == expected


def test_worker_startup_diagnostics_preserve_protocol_and_runtime_causes():
    rejected = _startup_failure(IPCProtocolError("invalid-ready", "bad"))
    failed = _startup_failure(RuntimeError("boom"))
    assert rejected == "worker startup rejected: invalid-ready"
    assert failed == "worker startup failed: RuntimeError"


def test_cancelling_an_absent_monitor_is_a_noop():
    assert asyncio.run(_cancel_monitor(None)) is None


@pytest.mark.parametrize(
    "document",
    [None, {}, {**training_spec().document(), "featureNames": "load"}],
)
def test_training_spec_documents_reject_wrong_shapes(document):
    with pytest.raises(TrainingError, match="report-invalid"):
        TrainingSpec.from_document(document)


@pytest.mark.parametrize(
    "changes",
    [
        {"samples": 0},
        {"corpus_sha256": "A" * 64},
        {"quality": {"loss": float("inf")}},
    ],
)
def test_training_report_rejects_invalid_evidence(changes):
    values = {
        "spec": training_spec(),
        "samples": 1,
        "corpus_sha256": "a" * 64,
        "model_sha256": "b" * 64,
        "quality": {"useful": True},
    }
    values.update(changes)
    with pytest.raises(TrainingError, match="report-invalid"):
        TrainingReport(**values)


def test_quality_metrics_accept_nested_lists_and_reject_unknown_types():
    assert _quality_is_finite([1.0, {"useful": True}, None]) is True
    assert _quality_is_finite([float("inf")]) is False
    assert _quality_is_finite((1.0,)) is False


def test_onnx_export_reports_missing_dependency(monkeypatch, tmp_path):
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "onnx":
            raise ImportError("missing")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    model = SimpleNamespace(input_width=1, weights=(1.0,), intercept=0.0)
    with pytest.raises(TrainingError, match="exporter-missing"):
        OnnxLinearExporter().export(model, tmp_path / "model.onnx")


def test_onnx_export_reports_validator_failure(monkeypatch, tmp_path):
    onnx = pytest.importorskip("onnx")
    monkeypatch.setattr(
        onnx.checker,
        "check_model",
        lambda _model: (_ for _ in ()).throw(ValueError("bad graph")),
    )
    model = SimpleNamespace(input_width=1, weights=(1.0,), intercept=0.0)
    with pytest.raises(TrainingError, match="ONNX validation failed"):
        OnnxLinearExporter().export(model, tmp_path / "model.onnx")


def test_onnx_export_writes_the_portable_tensor_contract(tmp_path):
    onnx = pytest.importorskip("onnx")
    destination = tmp_path / "nested" / "model.onnx"
    model = SimpleNamespace(input_width=2, weights=(0.25, -0.5), intercept=1.5)

    OnnxLinearExporter().export(model, destination)

    portable = onnx.load(destination)
    assert portable.ir_version <= 8
    assert [(entry.domain, entry.version) for entry in portable.opset_import] == [("", 13)]
    assert [node.op_type for node in portable.graph.node] == ["MatMul", "Add"]
    input_dimensions = portable.graph.input[0].type.tensor_type.shape.dim
    assert [dimension.dim_value for dimension in input_dimensions] == [
        1,
        2,
    ]
    assert [
        dimension.dim_value for dimension in portable.graph.output[0].type.tensor_type.shape.dim
    ] == [1, 1]


def installed_training(tmp_path) -> InstalledTraining:
    variant = InstalledVariant(
        "gpu",
        "local-forecast-gpu",
        "ncnn",
        tmp_path / "model.param",
        {},
    )
    return InstalledTraining("resource-scheduler", tmp_path / "manifest.json", (variant,))


def test_installed_training_document_and_successful_cli_are_actionable(
    monkeypatch, tmp_path, capsys
):
    installed = installed_training(tmp_path)
    observed = []

    def install(report, **kwargs):
        observed.append((report, kwargs))
        return installed

    monkeypatch.setattr("omnitensor.training.cli.install_training", install)

    code = install_main(
        [
            str(tmp_path / "training-report.json"),
            "--targets",
            "gpu",
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--bindings-root",
            str(tmp_path / "bindings"),
            "--build-root",
            str(tmp_path / "build"),
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert code == 0
    assert output["variants"] == [
        {
            "accelerator": "gpu",
            "artifactId": "local-forecast-gpu",
            "format": "ncnn",
            "installedAt": str(tmp_path / "model.param"),
        }
    ]
    assert output["next"] == "systemctl --user restart omnitensor.service"
    assert observed == [
        (
            tmp_path / "training-report.json",
            {
                "targets": ("gpu",),
                "artifact_root": tmp_path / "artifacts",
                "bindings_root": tmp_path / "bindings",
                "build_root": tmp_path / "build",
            },
        )
    ]


def test_install_cli_refuses_auto_when_no_compiler_exists(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("omnitensor.training.cli.available_targets", lambda: ())
    assert install_main([str(tmp_path / "training-report.json")]) == 1
    assert capsys.readouterr().err == (
        "installation failed: compiler-missing: no compatible native compiler found; "
        "install the producer tool for a target lane\n"
    )


def test_install_cli_help_is_an_operator_contract(capsys):
    with pytest.raises(SystemExit, match="0"):
        install_main(["--help"])
    output = capsys.readouterr().out
    assert "usage: omnitensor-install-trained-model" in output
    assert "Compile and install a trained model for local accelerator lanes" in output
    assert "auto or comma-separated tpu,npu,gpu" in output
    assert "compatible source and compiler" in output


def test_training_cli_rejects_empty_feature_and_target_lists():
    with pytest.raises(TrainingError) as feature_error:
        _features(" , ")
    assert feature_error.value.code == "features-invalid"
    assert feature_error.value.detail == "at least one feature is required"
    with pytest.raises(TrainingError) as target_error:
        _targets(" , ")
    assert target_error.value.code == "targets-invalid"
    assert target_error.value.detail == "at least one target is required"


def test_compiler_lookup_falls_back_to_path_and_reports_absence(monkeypatch):
    monkeypatch.setattr(
        "omnitensor.training.installation.shutil.which",
        lambda name: f"/tools/{name}" if name == "available-tool" else None,
    )
    assert _tool("available-tool") == Path("/tools/available-tool")
    assert _tool("absent-tool") is None


def test_compiler_lookup_prefers_only_executable_venv_siblings(monkeypatch, tmp_path):
    interpreter = tmp_path / "python"
    interpreter.write_text("", encoding="utf-8")
    compiler = tmp_path / "compiler"
    compiler.write_text("", encoding="utf-8")
    monkeypatch.setattr("omnitensor.training.installation.sys.executable", str(interpreter))
    monkeypatch.setattr("omnitensor.training.installation.shutil.which", lambda _name: None)

    compiler.chmod(0o755)
    assert _tool("compiler") == compiler
    compiler.chmod(0o644)
    assert _tool("compiler") is None
