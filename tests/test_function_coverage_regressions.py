from __future__ import annotations

import asyncio
import builtins
import hashlib
import json
from types import SimpleNamespace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor import acceptance
from omnitensor.control import _integral_weight
from omnitensor.job_codec import _request_id_from_text
from omnitensor.plugins.artifact_installation import (
    ArtifactInstallationError,
    _copy_digested,
)
from omnitensor.plugins.build_ingestion import BuildMetadataIngestor
from omnitensor.plugins.collection import (
    MAX_COLLECTED_ITEMS,
    BoundedCollector,
    CollectionError,
    ReplaySource,
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
from omnitensor.plugins.triggers import SourceStatus, Trigger, TriggerKind
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


def test_confinement_probe_reports_available_and_blocked_hosts(monkeypatch):
    from omnitensor.plugins import seccomp

    monkeypatch.setattr(seccomp, "confinement_error", lambda: "")
    available = acceptance.check_confinement()
    assert (available.name, available.ok) == ("confinement", True)
    assert available.detail
    monkeypatch.setattr(seccomp, "confinement_error", lambda: "unsupported architecture")
    blocked = acceptance.check_confinement()
    assert (blocked.name, blocked.ok) == ("confinement", False)
    assert "unsupported architecture" in blocked.detail


def test_ingestor_root_ports_expose_normalized_opt_in_paths(tmp_path):
    expected = (tmp_path.resolve(),)
    permissions = SimpleNamespace(allows=lambda _permission: True)
    assert BuildMetadataIngestor([tmp_path], permissions).roots == expected
    assert DocumentIngestor([tmp_path]).roots == expected


class SampleCollector(BoundedCollector):
    plugin_id = "sample-plugin"
    label = "sample"
    source_name = "smart-nvme-io"

    def identity_of(self, item):
        return item.identity

    def document_of(self, item):
        return {"id": item.identity}


def sample_trigger() -> Trigger:
    return Trigger("sample-plugin", "sample-1", TriggerKind.MANUAL, {}, 1)


def sample_collector(*snapshots) -> SampleCollector:
    source = ReplaySource(snapshots, label="sample")
    permissions = SimpleNamespace(allows=lambda _permission: True)
    return SampleCollector(source, permissions)


def test_base_collector_requires_each_subclass_to_declare_a_plugin_id():
    class ValidCollector(BoundedCollector):
        plugin_id = "valid-plugin"

    assert ValidCollector.plugin_id == "valid-plugin"
    for plugin_id in (None, ""):
        with pytest.raises(TypeError) as caught:

            class InvalidCollector(BoundedCollector):
                if plugin_id is not None:
                    locals()["plugin_id"] = plugin_id

        assert str(caught.value) == (
            "BoundedCollector subclasses must declare a non-empty plugin_id"
        )


def test_base_collector_default_churn_is_visible_in_public_output():
    before = SimpleNamespace(identity="item-1", value="before")
    after = SimpleNamespace(identity="item-1", value="after")
    subject = sample_collector(
        SourceSnapshot(SourceStatus.READY, 1, (before,)),
        SourceSnapshot(SourceStatus.READY, 2, (after,)),
    )

    first = asyncio.run(subject.collect(sample_trigger())).payload
    second = asyncio.run(subject.collect(sample_trigger())).payload

    assert first["churn"]["added"] == [{"id": "item-1"}]
    assert second["churn"]["changed"] == [{"id": "item-1", "fields": ["state"]}]


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
    subject = sample_collector(snapshot)
    with pytest.raises(CollectionError) as caught:
        asyncio.run(subject.collect(sample_trigger()))
    assert caught.value.code == "source-invalid"


def test_base_collector_accepts_the_exact_item_limit():
    items = tuple(SimpleNamespace(identity=f"item-{index}") for index in range(MAX_COLLECTED_ITEMS))
    subject = sample_collector(SourceSnapshot(SourceStatus.READY, 0, items))
    output = asyncio.run(subject.collect(sample_trigger())).payload
    assert len(output["items"]) == MAX_COLLECTED_ITEMS
    assert "truncatedItems" not in output


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
    monkeypatch.setattr(
        "omnitensor.training.cli.install_training",
        lambda *_args, **_kwargs: installed,
    )

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


def test_install_cli_refuses_auto_when_no_compiler_exists(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("omnitensor.training.cli.available_targets", lambda: ())
    assert install_main([str(tmp_path / "training-report.json")]) == 1
    assert "compiler-missing" in capsys.readouterr().err


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
