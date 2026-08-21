from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.plugins.network_collection import (
    MAX_NETWORK_COUNTER,
    NetworkConnectivity,
    NetworkCounters,
    NetworkLinkKind,
    NetworkLinkSample,
    NetworkLinkState,
    NetworkSnapshot,
)
from omnitensor.plugins.triggers import SourceStatus
from omnitensor.training.cli import network_train_main
from omnitensor.training.contracts import TrainingError
from omnitensor.training.network import (
    DEFAULT_MAX_FALSE_POSITIVE_RATE,
    DEFAULT_MIN_HOLDOUT_ROWS,
    DEFAULT_MIN_TRAINING_ROWS,
    FEATURE_GROUPS,
    KITSUNE_CITATION,
    KITSUNE_CODE_LICENSE,
    KITSUNE_CODE_URI,
    KITSUNE_PAPER_URI,
    MAX_REPLAY_LINE_BYTES,
    NETWORK_FEATURES,
    NETWORK_RECIPE,
    NORMAL_ONLY_CONFIRMATION,
    NetworkAnomalyTrainer,
    NetworkDataset,
    NetworkFeatureRow,
    NetworkReconstructionModel,
    OnnxNetworkExporter,
    _aggregate_features,
    _chronological_split,
    _counter_values,
    _fit_reconstruction,
    _fraction,
    _link_from_document,
    _percentile,
    _principal_direction,
    _snapshot_from_document,
    load_network_replay,
    network_feature_rows,
)


class FakeExporter:
    def __init__(self, content: bytes = b"portable network model") -> None:
        self.content = content
        self.model = None

    def export(self, model, destination) -> None:
        self.model = model
        Path(destination).write_bytes(self.content)


def link(
    observed_at_ms: int,
    value: int,
    *,
    stable_id: str = "wifi-main",
    state: NetworkLinkState = NetworkLinkState.UP,
    connectivity: NetworkConnectivity = NetworkConnectivity.FULL,
    carrier: bool = True,
    metered: bool = False,
    default_route: bool = True,
    signal_percent: int | None = 80,
) -> NetworkLinkSample:
    return NetworkLinkSample(
        stable_id,
        NetworkLinkKind.WIFI,
        state,
        connectivity,
        carrier,
        metered,
        default_route,
        signal_percent,
        NetworkCounters(value, value * 2, value // 10, value // 20, value // 25, value // 50),
        observed_at_ms,
    )


def snapshot(
    index: int,
    *,
    links: tuple[NetworkLinkSample, ...] | None = None,
    status: SourceStatus = SourceStatus.READY,
) -> NetworkSnapshot:
    observed = 1_000_000 + index * 1_000
    return NetworkSnapshot(
        status,
        observed,
        links if links is not None else (link(observed, 100 + index * 10),),
    )


def link_document(item: NetworkLinkSample) -> dict:
    return {
        "stableId": item.stable_id,
        "kind": str(item.kind),
        "state": str(item.state),
        "connectivity": str(item.connectivity),
        "carrier": item.carrier,
        "metered": item.metered,
        "defaultRoute": item.default_route,
        "signalPercent": item.signal_percent,
        "counters": {
            "receivedBytes": item.counters.received_bytes,
            "transmittedBytes": item.counters.transmitted_bytes,
            "receivedErrors": item.counters.received_errors,
            "transmittedErrors": item.counters.transmitted_errors,
            "receivedDrops": item.counters.received_drops,
            "transmittedDrops": item.counters.transmitted_drops,
        },
        "observedAtMs": item.observed_at_ms,
    }


def snapshot_document(item: NetworkSnapshot) -> dict:
    return {
        "schemaVersion": 2,
        "source": "network-manager-link-metadata",
        "sourceHealth": str(item.status),
        "observedAtMs": item.observed_at_ms,
        "links": [link_document(value) for value in item.links],
    }


def snapshot_document_v1(item: NetworkSnapshot, truncated_links: int = 0) -> dict:
    return {
        **snapshot_document(item),
        "schemaVersion": 1,
        "truncatedLinks": truncated_links,
    }


def write_replay(path: Path, snapshots: list[NetworkSnapshot]) -> bytes:
    raw = b"".join(
        json.dumps(snapshot_document(item), separators=(",", ":")).encode() + b"\n"
        for item in snapshots
    )
    path.write_bytes(raw)
    return raw


def normal_snapshots(count: int = 101) -> list[NetworkSnapshot]:
    return [snapshot(index) for index in range(count)]


def error_code(call) -> str:
    with pytest.raises(TrainingError) as raised:
        call()
    return raised.value.code


def training_error(call) -> TrainingError:
    with pytest.raises(TrainingError) as raised:
        call()
    return raised.value


def test_replay_loads_exact_bytes_and_discards_link_identity(tmp_path):
    path = tmp_path / "normal.jsonl"
    raw = write_replay(path, normal_snapshots())

    dataset = load_network_replay(path)

    assert dataset.replay_sha256 == hashlib.sha256(raw).hexdigest()
    assert dataset.snapshots == 101
    assert len(dataset.rows) == 100
    assert dataset.rows[0].observed_at_ms == 1_001_000
    assert dataset.rows[0].features == (
        10.0,
        20.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.8,
    )
    assert "wifi-main" not in repr(dataset.rows)


def test_train_emits_identity_free_portable_report_and_valid_onnx(tmp_path):
    replay = tmp_path / "normal.jsonl"
    write_replay(replay, normal_snapshots())
    dataset = load_network_replay(replay)

    report = NetworkAnomalyTrainer(
        confirm_normal=NORMAL_ONLY_CONFIRMATION,
        maximum_false_positive_rate=1.0,
    ).train(dataset, tmp_path / "fit")

    assert set(report) == {
        "version",
        "recipe",
        "inspiration",
        "dataset",
        "split",
        "features",
        "quality",
        "model",
        "tensorContract",
        "outputContract",
        "targets",
    }
    assert report["version"] == 1
    assert report["recipe"] == NETWORK_RECIPE
    assert report["inspiration"] == {
        "paper": KITSUNE_PAPER_URI,
        "code": KITSUNE_CODE_URI,
        "citation": KITSUNE_CITATION,
        "codeLicense": KITSUNE_CODE_LICENSE,
        "codeVendored": False,
    }
    assert report["dataset"] == {
        "replaySha256": dataset.replay_sha256,
        "snapshots": 101,
        "identityPersisted": False,
        "packetContentIngested": False,
        "operatorConfirmedNormal": True,
    }
    assert report["split"] == {
        "kind": "chronological",
        "training": 80,
        "holdout": 20,
        "holdoutFromMs": 1_081_000,
    }
    assert report["features"] == list(NETWORK_FEATURES)
    assert report["quality"]["anomalyRecall"] is None
    assert report["quality"]["holdoutFalsePositiveRate"] <= 1.0
    assert report["tensorContract"] == {
        "inputs": [{"shape": [1, 14], "dtype": "float32", "layout": "NC"}]
    }
    assert report["outputContract"] == {"kind": "raw"}
    assert report["targets"] == {"tpu": "uncompiled", "npu": "uncompiled", "gpu": "uncompiled"}
    report_path = tmp_path / "fit/network-training-report.json"
    rendered = report_path.read_text()
    assert json.loads(rendered) == report
    assert "wifi-main" not in rendered
    assert "packetPayload" not in rendered

    onnx = pytest.importorskip("onnx")
    assert onnx.__version__ == "1.22.0", "review serialized goldens with producer upgrades"
    model_path = tmp_path / "fit/model.onnx"
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    assert [node.op_type for node in model.graph.node] == [
        "Sub",
        "Div",
        "MatMul",
        "Sub",
        "Mul",
        "ReduceMean",
    ]
    input_shape = [item.dim_value for item in model.graph.input[0].type.tensor_type.shape.dim]
    output_shape = [item.dim_value for item in model.graph.output[0].type.tensor_type.shape.dim]
    assert input_shape == [1, 14]
    assert output_shape == [1, 1]
    assert model.opset_import[0].version == 13
    assert model.ir_version <= 8
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == (
        "51c003206857db8b16f7174b26565d748fe586523a166fb9d85a163824bb9c91"
    )
    assert hashlib.sha256(model.graph.SerializeToString()).hexdigest() == (
        "b9a0ab63326e01a595c55379b55abbe73a2f08139bb0428b1d8c5feb4afc6ac2"
    )
    assert hashlib.sha256(report_path.read_bytes()).hexdigest() == (
        "b1a809760a1541eaa4e51caca1c7ec9eb92b5c66a84cf348e445a8d98fe74e6a"
    )
    assert [
        (item.name, hashlib.sha256(item.SerializeToString()).hexdigest())
        for item in model.graph.initializer
    ] == [
        ("means", "be132fa73992a18fba0df54ca4ecc2517b2a37f646a289e0b299de9473918557"),
        ("scales", "53af452260213e4f7ff677fbc90aa6f392a0824d3910ac836cfd46d2a40a7cda"),
        (
            "projection",
            "d959db628ff667f20264552ad6397ec2925caaa07a2aad09ee0c541aed72d7cd",
        ),
    ]


def test_aggregate_features_cover_state_connectivity_and_hotplug():
    earlier = snapshot(
        0,
        links=(
            link(1_000_000, 100, stable_id="old"),
            link(1_000_000, 50, stable_id="gone"),
        ),
    )
    later = snapshot(
        2,
        links=(
            link(
                1_002_000,
                140,
                stable_id="old",
                state=NetworkLinkState.DEGRADED,
                connectivity=NetworkConnectivity.LIMITED,
                carrier=False,
                metered=True,
                default_route=False,
                signal_percent=20,
            ),
            link(
                1_002_000,
                999,
                stable_id="new",
                state=NetworkLinkState.DOWN,
                connectivity=NetworkConnectivity.NONE,
                carrier=True,
                signal_percent=None,
            ),
        ),
    )

    features = _aggregate_features(earlier, later)

    assert features[:6] == (20.0, 40.0, 2.0, 1.0, 0.5, 0.0)
    assert features[6:] == (2.0, 0.0, 0.5, 1.0, 0.5, 0.5, 0.5, 0.2)


def test_empty_link_snapshot_has_finite_zero_fractions(monkeypatch):
    result = _aggregate_features(snapshot(0, links=()), snapshot(1, links=()))
    assert result == (0.0,) * len(NETWORK_FEATURES)
    assert all(math.isfinite(value) for value in result)

    monkeypatch.setattr("omnitensor.training.network._fraction", lambda *_args: math.inf)
    assert error_code(lambda: _aggregate_features(snapshot(0), snapshot(1))) == ("features-invalid")


def test_counter_regression_refuses_replay():
    later = snapshot(1, links=(link(1_001_000, 99),))
    assert error_code(lambda: network_feature_rows((snapshot(0), later))) == "counter-regression"


@pytest.mark.parametrize(
    ("snapshots", "code"),
    [
        ("bad", "replay-invalid"),
        ((), "insufficient-history"),
        ((snapshot(0),), "insufficient-history"),
        ((snapshot(0), snapshot(0)), "observations-unordered"),
        ((snapshot(0, status=SourceStatus.DEGRADED), snapshot(1)), "snapshot-unhealthy"),
        ((object(), snapshot(1)), "snapshot-invalid"),
    ],
)
def test_feature_row_boundary_rejections(snapshots, code):
    assert error_code(lambda: network_feature_rows(snapshots)) == code


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schemaVersion", True),
        ("schemaVersion", 3),
        ("source", "packet-capture"),
        ("sourceHealth", "mystery"),
        ("observedAtMs", True),
        ("links", "bad"),
        ("truncatedLinks", 0),
    ],
)
def test_snapshot_document_contract_is_exact(field, value):
    document = snapshot_document(snapshot(0))
    document[field] = value
    assert error_code(lambda: _snapshot_from_document(document)) == "snapshot-invalid"


def test_a_persisted_version_1_snapshot_without_truncation_stays_readable():
    parsed = _snapshot_from_document(snapshot_document_v1(snapshot(0)))
    assert parsed == snapshot(0)


@pytest.mark.parametrize("truncated", [True, 1, -1, "0"])
def test_a_version_1_snapshot_missing_links_is_refused(truncated):
    document = snapshot_document_v1(snapshot(0), truncated_links=truncated)
    assert error_code(lambda: _snapshot_from_document(document)) == "snapshot-invalid"


def test_a_version_2_snapshot_may_not_carry_the_retired_truncation_counter():
    document = {**snapshot_document(snapshot(0)), "truncatedLinks": 0}
    assert error_code(lambda: _snapshot_from_document(document)) == "snapshot-invalid"


def test_snapshot_and_link_documents_reject_extra_or_missing_fields():
    document = snapshot_document(snapshot(0))
    document["packet"] = "secret"
    assert error_code(lambda: _snapshot_from_document(document)) == "snapshot-invalid"
    document = snapshot_document(snapshot(0))
    del document["source"]
    assert error_code(lambda: _snapshot_from_document(document)) == "snapshot-invalid"

    link_doc = link_document(link(1_000_000, 1))
    link_doc["ssid"] = "private"
    assert error_code(lambda: _link_from_document(link_doc)) == "snapshot-invalid"
    del link_doc["ssid"]
    del link_doc["carrier"]
    assert error_code(lambda: _link_from_document(link_doc)) == "snapshot-invalid"


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("link", "kind", "invalid"),
        ("link", "state", "invalid"),
        ("link", "connectivity", "invalid"),
        ("link", "carrier", 1),
        ("link", "signalPercent", 101),
        ("link", "observedAtMs", -1),
        ("counter", "receivedBytes", -1),
        ("counter", "transmittedBytes", MAX_NETWORK_COUNTER + 1),
    ],
)
def test_link_document_values_are_typed_and_bounded(section, field, value):
    document = snapshot_document(snapshot(0))
    link_doc = document["links"][0]
    target = link_doc if section == "link" else link_doc["counters"]
    target[field] = value
    assert error_code(lambda: _snapshot_from_document(document)) == "snapshot-invalid"


def test_counter_document_is_closed_and_duplicate_identity_is_rejected():
    document = link_document(link(1_000_000, 1))
    document["counters"]["packets"] = 1
    assert error_code(lambda: _link_from_document(document)) == "snapshot-invalid"

    duplicate = snapshot_document(snapshot(0, links=(link(1_000_000, 1), link(1_000_000, 2))))
    assert error_code(lambda: _snapshot_from_document(duplicate)) == "snapshot-invalid"


def test_replay_file_safety_and_bounds(tmp_path, monkeypatch):
    missing = tmp_path / "missing"
    missing_error = training_error(lambda: load_network_replay(missing))
    assert (missing_error.code, missing_error.detail) == (
        "replay-invalid",
        "network replay is not a safe regular file",
    )
    directory = tmp_path / "directory"
    directory.mkdir()
    assert error_code(lambda: load_network_replay(directory)) == "replay-invalid"
    real = tmp_path / "real"
    write_replay(real, normal_snapshots(2))
    alias = tmp_path / "alias"
    alias.symlink_to(real)
    assert error_code(lambda: load_network_replay(alias)) == "replay-invalid"

    monkeypatch.setattr(
        "omnitensor.training.network.MAX_REPLAY_BYTES",
        real.stat().st_size - 1,
    )
    size_error = training_error(lambda: load_network_replay(real))
    assert (size_error.code, size_error.detail) == (
        "replay-too-large",
        "network replay byte limit exceeded",
    )


@pytest.mark.parametrize("raw", [b"\n", b"not-json\n", b"\xff\n"])
def test_replay_rejects_blank_or_invalid_json_lines(tmp_path, raw):
    path = tmp_path / "bad.jsonl"
    path.write_bytes(raw)
    assert error_code(lambda: load_network_replay(path)) == "replay-invalid"


def test_replay_error_detail_and_exact_size_boundaries(tmp_path, monkeypatch):
    path = tmp_path / "replay.jsonl"
    path.write_bytes(b"not-json\n")
    invalid_json = training_error(lambda: load_network_replay(path))
    assert invalid_json.detail == "network replay line 1: invalid JSON"

    path.write_bytes(b"{}\n")
    invalid_document = training_error(lambda: load_network_replay(path))
    assert invalid_document.detail == (
        "network replay line 1: network snapshot identity is invalid"
    )

    raw = write_replay(path, normal_snapshots(2))
    monkeypatch.setattr("omnitensor.training.network.MAX_REPLAY_BYTES", len(raw))
    lines = raw.splitlines(keepends=True)
    assert len(lines[0]) == len(lines[1])
    monkeypatch.setattr("omnitensor.training.network.MAX_REPLAY_LINE_BYTES", len(lines[0]))
    monkeypatch.setattr("omnitensor.training.network.MAX_REPLAY_SNAPSHOTS", 2)
    assert load_network_replay(path).snapshots == 2


def test_replay_rejects_oversize_line_and_snapshot_count(tmp_path, monkeypatch):
    path = tmp_path / "large.jsonl"
    path.write_bytes(b"{" + b" " * MAX_REPLAY_LINE_BYTES + b"}\n")
    assert error_code(lambda: load_network_replay(path)) == "replay-invalid"

    write_replay(path, normal_snapshots(2))
    monkeypatch.setattr("omnitensor.training.network.MAX_REPLAY_SNAPSHOTS", 1)
    count_error = training_error(lambda: load_network_replay(path))
    assert (count_error.code, count_error.detail) == (
        "replay-too-large",
        "network replay snapshot limit exceeded",
    )


def test_one_millisecond_snapshot_interval_is_accepted():
    earlier = snapshot(0)
    observed = earlier.observed_at_ms + 1
    later = NetworkSnapshot(SourceStatus.READY, observed, (link(observed, 110),))
    rows = network_feature_rows((earlier, later))
    assert rows[0].observed_at_ms == observed
    assert rows[0].features[0] == 10_000


def test_training_snapshot_errors_preserve_stable_details():
    invalid = training_error(lambda: network_feature_rows((object(), snapshot(1))))
    assert (invalid.code, invalid.detail) == (
        "snapshot-invalid",
        "network snapshot is not typed",
    )
    unhealthy = training_error(
        lambda: network_feature_rows((snapshot(0, status=SourceStatus.UNAVAILABLE), snapshot(1)))
    )
    assert (unhealthy.code, unhealthy.detail) == (
        "snapshot-unhealthy",
        "normal training requires ready network snapshots",
    )


def test_confirmation_and_trainer_bounds_are_enforced():
    assert error_code(lambda: NetworkAnomalyTrainer(confirm_normal="no")) == (
        "normal-baseline-not-confirmed"
    )
    for value in (True, 0, 1.5):
        with pytest.raises(ValueError, match="row bounds"):
            NetworkAnomalyTrainer(
                confirm_normal=NORMAL_ONLY_CONFIRMATION,
                minimum_training_rows=value,
            )
    for value in (True, -0.1, 1.1, math.inf, "bad"):
        with pytest.raises(ValueError, match="false-positive"):
            NetworkAnomalyTrainer(
                confirm_normal=NORMAL_ONLY_CONFIRMATION,
                maximum_false_positive_rate=value,
            )


def test_trainer_refuses_short_or_unstable_replay_and_empty_export(tmp_path):
    short = NetworkDataset(
        tuple(NetworkFeatureRow(index, (0.0,) * 14) for index in range(4)),
        "a" * 64,
        5,
    )
    trainer = NetworkAnomalyTrainer(
        FakeExporter(),
        confirm_normal=NORMAL_ONLY_CONFIRMATION,
        minimum_training_rows=3,
        minimum_holdout_rows=2,
    )
    assert error_code(lambda: trainer.train(short, tmp_path)) == "insufficient-history"

    unstable_rows = tuple(
        NetworkFeatureRow(index, ((0.0,) * 14 if index < 8 else (100.0,) * 14))
        for index in range(10)
    )
    unstable = NetworkDataset(unstable_rows, "b" * 64, 11)
    strict = NetworkAnomalyTrainer(
        FakeExporter(),
        confirm_normal=NORMAL_ONLY_CONFIRMATION,
        minimum_training_rows=8,
        minimum_holdout_rows=2,
        maximum_false_positive_rate=0,
    )
    assert error_code(lambda: strict.train(unstable, tmp_path)) == "model-not-stable"

    empty = NetworkAnomalyTrainer(
        FakeExporter(b""),
        confirm_normal=NORMAL_ONLY_CONFIRMATION,
        minimum_training_rows=3,
        minimum_holdout_rows=2,
        maximum_false_positive_rate=1,
    )
    enough = NetworkDataset(
        tuple(NetworkFeatureRow(index, (float(index),) * 14) for index in range(5)),
        "c" * 64,
        6,
    )
    assert error_code(lambda: empty.train(enough, tmp_path / "empty")) == "export-failed"


def test_model_score_contract_and_reconstruction_fit():
    rows = tuple(
        NetworkFeatureRow(index, tuple(float(index * (column + 1)) for column in range(14)))
        for index in range(1, 30)
    )
    model = _fit_reconstruction(rows)
    assert len(model.means) == 14
    assert len(model.scales) == 14
    assert len(model.projection) == 14
    assert all(len(row) == 14 for row in model.projection)
    assert model.score(rows[10].features) < model.score((999.0,) * 14)
    assert error_code(lambda: model.score((1.0,) * 13)) == "features-invalid"
    assert error_code(lambda: model.score((True,) * 14)) == "features-invalid"
    assert error_code(lambda: model.score((math.inf,) * 14)) == "features-invalid"


def test_principal_direction_zero_covariance_and_projection_groups():
    assert _principal_direction(((0.0,) * 14,), FEATURE_GROUPS[0]) == (1.0, 0.0)
    direction = _principal_direction(((1.0, 2.0), (2.0, 4.0)), (0, 1))
    assert math.isclose(sum(value * value for value in direction), 1.0)


def test_split_percentile_fraction_and_counter_boundaries():
    rows = tuple(NetworkFeatureRow(index, (0.0,) * 14) for index in range(10))
    training, holdout = _chronological_split(rows, 3, 2)
    assert [item.observed_at_ms for item in training] == list(range(8))
    assert [item.observed_at_ms for item in holdout] == [8, 9]
    training, holdout = _chronological_split(rows[:5], 3, 2)
    assert len(training) == 3
    assert len(holdout) == 2
    assert error_code(lambda: _chronological_split(rows[:4], 3, 2)) == "insufficient-history"
    assert _percentile((3.0, 1.0, 2.0), 0.0) == 1.0
    assert _percentile((3.0, 1.0, 2.0), 0.5) == 2.0
    assert _percentile((3.0, 1.0, 2.0), 1.0) == 3.0
    assert error_code(lambda: _percentile((), 0.5)) == "insufficient-history"
    assert _fraction((), "up") == 0
    assert _fraction(("up", "down", "up"), "up") == pytest.approx(2 / 3)
    maximum = NetworkCounters(*(MAX_NETWORK_COUNTER for _index in range(6)))
    assert _counter_values(maximum) == (MAX_NETWORK_COUNTER,) * 6
    excessive = NetworkCounters(MAX_NETWORK_COUNTER + 1, 0, 0, 0, 0, 0)
    counter_error = training_error(lambda: _counter_values(excessive))
    assert (counter_error.code, counter_error.detail) == (
        "snapshot-invalid",
        "network counters exceed the supported range",
    )


def test_onnx_exporter_missing_dependency_validation_and_cleanup(tmp_path, monkeypatch):
    model = NetworkReconstructionModel(
        (0.0,) * 14,
        (1.0,) * 14,
        tuple(tuple(float(row == column) for column in range(14)) for row in range(14)),
    )
    original_import = __import__

    def without_onnx(name, *args, **kwargs):
        if name == "onnx":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", without_onnx)
    assert error_code(lambda: OnnxNetworkExporter().export(model, tmp_path / "missing.onnx")) == (
        "exporter-missing"
    )
    monkeypatch.setattr("builtins.__import__", original_import)

    import onnx

    monkeypatch.setattr(
        onnx.checker,
        "check_model",
        lambda _model: (_item for _item in ()).throw(ValueError("bad graph")),
    )
    assert error_code(lambda: OnnxNetworkExporter().export(model, tmp_path / "bad.onnx")) == (
        "export-failed"
    )
    monkeypatch.undo()
    monkeypatch.setattr(
        onnx,
        "save_model",
        lambda *_args: (_item for _item in ()).throw(OSError("disk")),
    )
    with pytest.raises(OSError, match="disk"):
        OnnxNetworkExporter().export(model, tmp_path / "failed.onnx")
    assert list(tmp_path.glob(".network-model-*")) == []


def test_network_cli_help_required_arguments_error_and_exact_envelope(
    tmp_path, capsys, monkeypatch
):
    with pytest.raises(SystemExit) as help_exit:
        network_train_main(["--help"])
    help_text = capsys.readouterr().out
    assert help_exit.value.code == 0
    assert "usage: omnitensor-train-network-model" in help_text
    assert "--confirm-normal-only CONFIRM_NORMAL_ONLY" in help_text
    assert "after reviewing the replay, pass exactly I-confirm-" in help_text
    assert "this-replay-is-normal" in help_text

    with pytest.raises(SystemExit) as missing:
        network_train_main([])
    assert missing.value.code == 2
    assert "--replay, --confirm-normal-only" in capsys.readouterr().err

    captured = {}

    def fake_load(path):
        captured["replay"] = path
        return "dataset"

    class CapturingTrainer:
        def __init__(self, **kwargs):
            captured["trainer"] = kwargs

        def train(self, dataset, output):
            captured["train"] = (dataset, output)
            return {"split": {"training": 64, "holdout": 16}, "quality": {"score": 0}}

    monkeypatch.setattr("omnitensor.training.cli.load_network_replay", fake_load)
    monkeypatch.setattr("omnitensor.training.cli.NetworkAnomalyTrainer", CapturingTrainer)
    code = network_train_main(
        [
            "--replay",
            str(tmp_path / "normal.jsonl"),
            "--output-dir",
            str(tmp_path / "fit"),
            "--confirm-normal-only",
            NORMAL_ONLY_CONFIRMATION,
        ]
    )
    rendered = capsys.readouterr().out
    assert code == 0
    assert captured == {
        "replay": tmp_path / "normal.jsonl",
        "trainer": {
            "confirm_normal": NORMAL_ONLY_CONFIRMATION,
            "minimum_training_rows": DEFAULT_MIN_TRAINING_ROWS,
            "minimum_holdout_rows": DEFAULT_MIN_HOLDOUT_ROWS,
            "maximum_false_positive_rate": DEFAULT_MAX_FALSE_POSITIVE_RATE,
        },
        "train": ("dataset", tmp_path / "fit"),
    }
    assert json.loads(rendered) == {
        "report": str(tmp_path / "fit/network-training-report.json"),
        "portableModel": str(tmp_path / "fit/model.onnx"),
        "samples": {"training": 64, "holdout": 16},
        "quality": {"score": 0},
        "next": "compile and qualify this source separately for each target lane",
    }


def test_network_cli_catches_training_and_path_errors(tmp_path, capsys):
    code = network_train_main(
        [
            "--replay",
            str(tmp_path / "missing"),
            "--confirm-normal-only",
            NORMAL_ONLY_CONFIRMATION,
        ]
    )
    assert code == 1
    assert "network training failed: replay-invalid" in capsys.readouterr().err


@given(
    received=st.integers(min_value=0, max_value=1_000_000),
    transmitted=st.integers(min_value=0, max_value=1_000_000),
    elapsed=st.integers(min_value=1, max_value=60_000),
    signal=st.one_of(st.none(), st.integers(min_value=0, max_value=100)),
)
def test_aggregate_feature_property_is_finite_ordered_and_identity_free(
    received, transmitted, elapsed, signal
):
    start = 1_000_000
    before = NetworkLinkSample(
        "private-id",
        NetworkLinkKind.ETHERNET,
        NetworkLinkState.UP,
        NetworkConnectivity.FULL,
        True,
        False,
        True,
        signal,
        NetworkCounters(0, 0, 0, 0, 0, 0),
        start,
    )
    after = NetworkLinkSample(
        "private-id",
        NetworkLinkKind.ETHERNET,
        NetworkLinkState.UP,
        NetworkConnectivity.FULL,
        True,
        False,
        True,
        signal,
        NetworkCounters(received, transmitted, 0, 0, 0, 0),
        start + elapsed,
    )
    features = _aggregate_features(
        NetworkSnapshot(SourceStatus.READY, start, (before,)),
        NetworkSnapshot(SourceStatus.READY, start + elapsed, (after,)),
    )
    assert len(features) == len(NETWORK_FEATURES)
    assert features[0] == pytest.approx(received / (elapsed / 1000))
    assert features[1] == pytest.approx(transmitted / (elapsed / 1000))
    assert all(math.isfinite(value) for value in features)
    assert "private-id" not in repr(features)


def test_training_snapshot_guard_survives_disabled_assertions(monkeypatch):
    from omnitensor.training import network as network_module

    monkeypatch.setattr(network_module, "network_snapshot_error", lambda _snapshot: "")
    with pytest.raises(TrainingError) as failure:
        network_module._require_training_snapshot(object())
    assert failure.value.code == "snapshot-invalid"
