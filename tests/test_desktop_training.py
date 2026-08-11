from __future__ import annotations

import hashlib
import json
import math
from contextlib import contextmanager
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.training.cli import desktop_revoke_main, desktop_train_main
from omnitensor.training.contracts import TrainingError
from omnitensor.training.desktop import (
    DEFAULT_MIN_ACCURACY,
    DEFAULT_MIN_CLASS_EXAMPLES,
    DEFAULT_MIN_MACRO_RECALL,
    DESKTOP_CONFIRMATION,
    DESKTOP_FEATURES,
    DESKTOP_RECIPE,
    DESKTOP_REVOCATION_CONFIRMATION,
    DesktopDataset,
    DesktopSuggestionTrainer,
    OnnxDesktopExporter,
    _bounded_integer,
    _build_example,
    _example_from_document,
    _history_line,
    _quality,
    _require_suggestions,
    _time_split,
    _unit_metric,
    load_desktop_history,
    revoke_desktop_history,
)


class FakeExporter:
    def __init__(self, content: bytes = b"portable desktop scores") -> None:
        self.content = content
        self.model = None

    def export(self, model, destination) -> None:
        self.model = model
        Path(destination).write_bytes(self.content)


def document(index: int, suggestion: str | None = None) -> dict:
    selected = suggestion or ("tile-left" if index % 2 == 0 else "maximize")
    narrow = selected == "tile-left"
    return {
        "schemaVersion": 1,
        "observedAtMs": 1_000_000 + index * 1_000,
        "workspaceCount": 4,
        "activeWorkspace": index % 4,
        "windowCount": 3,
        "visibleWindows": 2,
        "dialogWindows": 0,
        "utilityWindows": 0,
        "notificationWindows": 0,
        "focusedRole": "normal",
        "focusedWidthBucket": "narrow" if narrow else "wide",
        "focusedHeightBucket": "medium",
        "confirmedSuggestion": selected,
        "confirmation": DESKTOP_CONFIRMATION,
    }


def write_history(path: Path, count: int = 100) -> bytes:
    raw = b"".join(
        json.dumps(document(index), separators=(",", ":")).encode() + b"\n"
        for index in range(count)
    )
    path.write_bytes(raw)
    return raw


def training_error(call) -> TrainingError:
    with pytest.raises(TrainingError) as raised:
        call()
    return raised.value


def error_code(call) -> str:
    return training_error(call).code


def trainer(exporter=None, **changes) -> DesktopSuggestionTrainer:
    options = {
        "exporter": exporter,
        "minimum_class_examples": 2,
        "minimum_accuracy": 0.5,
        "minimum_macro_recall": 0.5,
    }
    options.update(changes)
    return DesktopSuggestionTrainer(**options)


def test_content_free_personal_history_trains_portable_scores(tmp_path):
    history = tmp_path / "desktop.jsonl"
    raw = write_history(history)
    dataset = load_desktop_history(history)
    report = trainer().train(dataset, tmp_path / "fit")

    assert dataset.history_sha256 == hashlib.sha256(raw).hexdigest()
    assert len(dataset.examples) == 100
    assert len(dataset.examples[0].features) == len(DESKTOP_FEATURES) == 20
    assert report["version"] == 1
    assert report["recipe"] == DESKTOP_RECIPE
    assert report["dataset"] == {
        "historySha256": dataset.history_sha256,
        "examples": 100,
        "contentCaptured": False,
        "applicationIdsPersisted": False,
        "windowIdsPersisted": False,
        "labels": "explicit-user-confirmation",
    }
    assert report["split"] == {
        "kind": "chronological",
        "training": 80,
        "holdout": 20,
        "holdoutFromMs": 1_080_000,
    }
    assert report["resultContract"] == {
        "version": 1,
        "kind": "desktop-layout-suggestion-scores",
        "suggestions": ["tile-left", "maximize"],
        "requiresUserConfirmation": True,
        "automaticWindowActions": False,
    }
    assert report["quality"] == {
        "accuracy": 1.0,
        "macroRecall": 1.0,
        "examples": 20,
    }
    assert report["tensorContract"] == {
        "inputs": [{"shape": [1, 20], "dtype": "float32", "layout": "NC"}]
    }
    assert report["featureContract"] == {
        "version": 1,
        "features": list(DESKTOP_FEATURES),
        "contentFree": True,
    }
    assert report["model"] == {
        "format": "onnx",
        "filename": "model.onnx",
        "sha256": hashlib.sha256((tmp_path / "fit/model.onnx").read_bytes()).hexdigest(),
    }
    assert report["outputContract"] == {"kind": "raw"}
    assert report["targets"] == {"tpu": "uncompiled", "npu": "uncompiled", "gpu": "uncompiled"}
    rendered = (tmp_path / "fit/desktop-training-report.json").read_text()
    assert json.loads(rendered) == report
    assert "private document" not in rendered
    assert "org.gnome" not in rendered
    assert "win-" not in rendered

    onnx = pytest.importorskip("onnx")
    model = onnx.load(tmp_path / "fit/model.onnx")
    onnx.checker.check_model(model)
    assert [node.op_type for node in model.graph.node] == ["Sub", "Div", "MatMul", "Add", "Sigmoid"]
    input_shape = [value.dim_value for value in model.graph.input[0].type.tensor_type.shape.dim]
    output_shape = [value.dim_value for value in model.graph.output[0].type.tensor_type.shape.dim]
    assert input_shape == [1, 20]
    assert output_shape == [1, 2]
    assert model.opset_import[0].version == 13
    assert model.ir_version <= 8


def test_record_contract_is_closed_versioned_and_confirmation_only():
    base = document(0)
    for field, value, code in (
        ("schemaVersion", 2, "record-invalid"),
        ("observedAtMs", True, "record-invalid"),
        ("workspaceCount", 0, "record-invalid"),
        ("activeWorkspace", 4, "record-invalid"),
        ("windowCount", 65, "record-invalid"),
        ("visibleWindows", 4, "record-invalid"),
        ("focusedRole", "editor", "record-invalid"),
        ("focusedWidthBucket", "huge", "record-invalid"),
        ("confirmedSuggestion", "close-all", "label-invalid"),
        ("confirmation", "implicit", "label-invalid"),
    ):
        changed = {**base, field: value}
        assert error_code(lambda changed=changed: _example_from_document(changed)) == code
    extra = {**base, "title": "private document"}
    assert error_code(lambda: _example_from_document(extra)) == "record-invalid"
    assert error_code(lambda: _example_from_document([])) == "record-invalid"


def test_empty_desktop_has_no_fake_focus_and_counts_are_consistent():
    empty = {
        **document(0),
        "windowCount": 0,
        "visibleWindows": 0,
        "focusedRole": "none",
        "focusedWidthBucket": "none",
        "focusedHeightBucket": "none",
    }
    example = _example_from_document(empty)
    assert all(math.isfinite(value) for value in example.features)
    for field in ("focusedRole", "focusedWidthBucket", "focusedHeightBucket"):
        bad = {**empty, field: "normal" if field == "focusedRole" else "medium"}
        assert error_code(lambda bad=bad: _example_from_document(bad)) == "record-invalid"
    over_roles = {**document(0), "dialogWindows": 2, "utilityWindows": 2}
    assert error_code(lambda: _example_from_document(over_roles)) == "record-invalid"


def test_feature_order_and_values_are_exact():
    example = _example_from_document(document(2, "tile-left"))
    assert example.observed_at_ms == 1_002_000
    assert example.suggestion == "tile-left"
    assert example.features == pytest.approx(
        (
            math.log1p(4),
            2 / 3,
            math.log1p(3),
            2 / 3,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
        )
    )


def test_history_file_bounds_safety_and_time_order(tmp_path, monkeypatch):
    missing = tmp_path / "missing.jsonl"
    error = training_error(lambda: load_desktop_history(missing))
    assert (error.code, error.detail) == (
        "history-invalid",
        "desktop history is not a safe regular file",
    )
    missing.mkdir()
    assert error_code(lambda: load_desktop_history(missing)) == "history-invalid"
    missing.rmdir()
    write_history(missing, 2)
    alias = tmp_path / "alias"
    alias.symlink_to(missing)
    assert error_code(lambda: load_desktop_history(alias)) == "history-invalid"
    monkeypatch.setattr("omnitensor.training.desktop.MAX_HISTORY_BYTES", missing.stat().st_size - 1)
    size_error = training_error(lambda: load_desktop_history(missing))
    assert (size_error.code, size_error.detail) == (
        "history-too-large",
        "desktop history byte limit exceeded",
    )
    monkeypatch.setattr("omnitensor.training.desktop.MAX_HISTORY_BYTES", missing.stat().st_size)
    assert len(load_desktop_history(missing).examples) == 2
    monkeypatch.undo()

    unordered = tmp_path / "unordered.jsonl"
    unordered.write_text(json.dumps(document(1)) + "\n" + json.dumps(document(0)) + "\n")
    assert error_code(lambda: load_desktop_history(unordered)) == "observations-unordered"


@pytest.mark.parametrize("raw", [b"\n", b"not-json\n", b"\xff\n"])
def test_history_lines_reject_empty_malformed_or_non_utf8(raw):
    error = training_error(lambda: _history_line(raw, 7))
    assert error.code == "history-invalid"
    assert error.detail.startswith("desktop history line 7")


def test_malformed_history_line_preserves_exact_cause():
    error = training_error(lambda: _history_line(b"not-json\n", 9))
    assert (error.code, error.detail) == (
        "history-invalid",
        "desktop history line 9: invalid JSON",
    )


def test_history_line_and_example_count_boundaries(tmp_path, monkeypatch):
    raw = json.dumps(document(0)).encode() + b"\n"
    monkeypatch.setattr("omnitensor.training.desktop.MAX_HISTORY_LINE_BYTES", len(raw))
    assert _history_line(raw, 1).observed_at_ms == 1_000_000
    path = tmp_path / "history.jsonl"
    write_history(path, 2)
    monkeypatch.setattr("omnitensor.training.desktop.MAX_HISTORY_EXAMPLES", 1)
    assert error_code(lambda: load_desktop_history(path)) == "history-too-large"
    monkeypatch.setattr("omnitensor.training.desktop.MAX_HISTORY_EXAMPLES", 2)
    assert len(load_desktop_history(path).examples) == 2
    path.write_bytes(b"")
    assert error_code(lambda: load_desktop_history(path)) == "insufficient-history"


def test_training_configuration_classes_and_splits_are_bounded(tmp_path):
    for value in (True, 0, 1.5):
        with pytest.raises(ValueError, match="minimum_class_examples"):
            DesktopSuggestionTrainer(minimum_class_examples=value)
    for name in ("minimum_accuracy", "minimum_macro_recall"):
        for value in (True, -1, 2, math.nan, "bad"):
            with pytest.raises(ValueError, match=name):
                DesktopSuggestionTrainer(**{name: value})
    assert _unit_metric(0, "metric") == 0.0
    assert _unit_metric(1, "metric") == 1.0
    assert _bounded_integer(0, 0, 1, "value") == 0

    one = (_example_from_document(document(0)),)
    assert error_code(lambda: _time_split(one)) == "insufficient-history"
    pair = one + (_example_from_document(document(1)),)
    assert tuple(map(len, _time_split(pair))) == (1, 1)
    assert error_code(
        lambda: trainer(FakeExporter()).train(
            DesktopDataset(pair, "a" * 64), tmp_path
        )
    ) == "class-imbalance"
    assert error_code(lambda: _require_suggestions(pair, ("tile-left",), 2, "holdout")) == (
        "class-imbalance"
    )
    two_left = pair + (_example_from_document(document(2)),)
    _require_suggestions(two_left, ("tile-left",), 2, "training")
    class_error = training_error(
        lambda: _require_suggestions(pair, ("tile-left",), 2, "holdout")
    )
    assert class_error.detail == "holdout needs at least 2 confirmations for tile-left"


@pytest.mark.parametrize(
    ("quality", "changes", "expected"),
    [
        (
            {"accuracy": 0.4, "macroRecall": 1.0, "examples": 20},
            {"minimum_accuracy": 0.5},
            ("model-not-useful", "held-out desktop accuracy is below the gate"),
        ),
        (
            {"accuracy": 1.0, "macroRecall": 0.4, "examples": 20},
            {"minimum_macro_recall": 0.5},
            ("model-not-useful", "held-out desktop macro recall is below the gate"),
        ),
    ],
)
def test_quality_gates_are_exact(tmp_path, monkeypatch, quality, changes, expected):
    history = tmp_path / "history.jsonl"
    write_history(history)
    dataset = load_desktop_history(history)
    monkeypatch.setattr("omnitensor.training.desktop._quality", lambda *_args: quality)
    error = training_error(
        lambda: trainer(FakeExporter(), **changes).train(dataset, tmp_path / expected[0])
    )
    assert (error.code, error.detail) == expected


def test_quality_boundaries_and_empty_export(tmp_path, monkeypatch):
    history = tmp_path / "history.jsonl"
    write_history(history)
    dataset = load_desktop_history(history)
    quality = {"accuracy": 0.5, "macroRecall": 0.5, "examples": 20}
    monkeypatch.setattr("omnitensor.training.desktop._quality", lambda *_args: quality)
    report = trainer(FakeExporter()).train(dataset, tmp_path / "boundary")
    assert report["quality"] == quality
    error = training_error(
        lambda: trainer(FakeExporter(b"")).train(dataset, tmp_path / "empty")
    )
    assert (error.code, error.detail) == (
        "export-failed",
        "exporter produced no portable model",
    )


def test_build_labels_and_quality_are_stable():
    suggestions = ("tile-left", "maximize")
    examples = tuple(_example_from_document(document(index)) for index in range(10))
    build = _build_example(examples[0], suggestions)
    assert (
        build.started_at_ms,
        build.features,
        build.build_failed,
        build.executed_optional,
        build.failed_optional,
    ) == (examples[0].observed_at_ms, examples[0].features, 0, ("1", "0"), ())

    class Model:
        def predict(self, features):
            return (1.0, 0.0) if features[13] == 1 else (0.0, 1.0)

    assert _quality(Model(), examples, suggestions) == {
        "accuracy": 1.0,
        "macroRecall": 1.0,
        "examples": 10,
    }


def test_onnx_exporter_missing_validation_and_cleanup(tmp_path, monkeypatch):
    from omnitensor.training.build import BuildAdvisorModel

    model = BuildAdvisorModel(
        (0.0,) * 20,
        (1.0,) * 20,
        tuple((1.0, -1.0) for _ in range(20)),
        (0.0, 0.0),
    )
    original_import = __import__

    def without_onnx(name, *args, **kwargs):
        if name == "onnx":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", without_onnx)
    assert error_code(lambda: OnnxDesktopExporter().export(model, tmp_path / "missing")) == (
        "exporter-missing"
    )
    monkeypatch.setattr("builtins.__import__", original_import)
    import onnx

    monkeypatch.setattr(
        onnx.checker,
        "check_model",
        lambda _model: (_item for _item in ()).throw(ValueError("bad")),
    )
    assert error_code(lambda: OnnxDesktopExporter().export(model, tmp_path / "bad")) == (
        "export-failed"
    )
    monkeypatch.undo()
    monkeypatch.setattr(
        onnx,
        "save_model",
        lambda *_args: (_item for _item in ()).throw(OSError("disk")),
    )
    with pytest.raises(OSError, match="disk"):
        OnnxDesktopExporter().export(model, tmp_path / "failed")
    assert list(tmp_path.glob(".desktop-model-*")) == []


def test_revocation_is_explicit_idempotent_locked_and_symlink_safe(tmp_path):
    history = tmp_path / "desktop.jsonl"
    write_history(history, 2)
    error = training_error(lambda: revoke_desktop_history(history, "wrong"))
    assert (error.code, error.detail) == (
        "revocation-not-confirmed",
        f"pass exactly {DESKTOP_REVOCATION_CONFIRMATION}",
    )
    assert history.is_file()
    assert revoke_desktop_history(history, DESKTOP_REVOCATION_CONFIRMATION) is True
    assert not history.exists()
    assert revoke_desktop_history(history, DESKTOP_REVOCATION_CONFIRMATION) is False
    assert (tmp_path / ".desktop-training.lock").is_file()

    target = tmp_path / "target.jsonl"
    target.write_text("private")
    alias = tmp_path / "alias.jsonl"
    alias.symlink_to(target)
    alias_error = training_error(
        lambda: revoke_desktop_history(alias, DESKTOP_REVOCATION_CONFIRMATION)
    )
    assert (alias_error.code, alias_error.detail) == (
        "history-invalid",
        "desktop history is not a safe regular file",
    )
    assert target.read_text() == "private"
    directory = tmp_path / "directory"
    directory.mkdir()
    assert error_code(
        lambda: revoke_desktop_history(directory, DESKTOP_REVOCATION_CONFIRMATION)
    ) == "history-invalid"


def test_revocation_rechecks_the_file_after_locking(tmp_path, monkeypatch):
    history = tmp_path / "desktop.jsonl"
    history.write_text("private")
    replacement = tmp_path / "replacement.jsonl"
    replacement.write_text("keep")

    @contextmanager
    def replace_during_lock(_root, _name):
        history.unlink()
        history.symlink_to(replacement)
        yield

    monkeypatch.setattr("omnitensor.training.desktop.store_lock", replace_during_lock)
    error = training_error(
        lambda: revoke_desktop_history(history, DESKTOP_REVOCATION_CONFIRMATION)
    )
    assert (error.code, error.detail) == (
        "history-invalid",
        "desktop history changed during revocation",
    )
    assert replacement.read_text() == "keep"


def test_desktop_clis_wire_exact_arguments_and_errors(tmp_path, capsys, monkeypatch):
    with pytest.raises(SystemExit) as train_help_exit:
        desktop_train_main(["--help"])
    train_help = capsys.readouterr().out
    assert train_help_exit.value.code == 0
    assert train_help.startswith(
        "usage: omnitensor-train-desktop-model [-h] --history HISTORY\n"
    )
    assert "Fit content-free personalized desktop suggestion scores" in train_help
    assert "--minimum-macro-recall MINIMUM_MACRO_RECALL" in train_help

    with pytest.raises(SystemExit) as revoke_help_exit:
        desktop_revoke_main(["--help"])
    revoke_help = capsys.readouterr().out
    assert revoke_help_exit.value.code == 0
    assert revoke_help.startswith(
        "usage: omnitensor-revoke-desktop-training-history [-h] --history HISTORY\n"
    )
    assert "Delete one explicit desktop training history file safely" in revoke_help
    assert f"pass exactly {DESKTOP_REVOCATION_CONFIRMATION}" in revoke_help

    captured = {}

    def fake_load(path):
        captured["load"] = path
        return "dataset"

    class CapturingTrainer:
        def __init__(self, **kwargs):
            captured["trainer"] = kwargs

        def train(self, dataset, output):
            captured["train"] = (dataset, output)
            return {"split": {"training": 80, "holdout": 20}, "quality": {"accuracy": 1}}

    monkeypatch.setattr("omnitensor.training.cli.load_desktop_history", fake_load)
    monkeypatch.setattr("omnitensor.training.cli.DesktopSuggestionTrainer", CapturingTrainer)
    output = tmp_path / "fit"
    assert desktop_train_main(
        [
            "--history",
            str(tmp_path / "history"),
            "--output-dir",
            str(output),
            "--minimum-class-examples",
            str(DEFAULT_MIN_CLASS_EXAMPLES),
            "--minimum-accuracy",
            str(DEFAULT_MIN_ACCURACY),
            "--minimum-macro-recall",
            str(DEFAULT_MIN_MACRO_RECALL),
        ]
    ) == 0
    assert captured == {
        "load": tmp_path / "history",
        "trainer": {
            "minimum_class_examples": DEFAULT_MIN_CLASS_EXAMPLES,
            "minimum_accuracy": DEFAULT_MIN_ACCURACY,
            "minimum_macro_recall": DEFAULT_MIN_MACRO_RECALL,
        },
        "train": ("dataset", output),
    }
    result = json.loads(capsys.readouterr().out)
    assert result == {
        "report": str(output / "desktop-training-report.json"),
        "portableModel": str(output / "model.onnx"),
        "samples": {"training": 80, "holdout": 20},
        "quality": {"accuracy": 1},
        "next": "compile and qualify separately; never apply a suggestion automatically",
    }

    monkeypatch.setattr(
        "omnitensor.training.cli.load_desktop_history",
        lambda _path: (_item for _item in ()).throw(TrainingError("bad", "history")),
    )
    assert desktop_train_main(["--history", "bad"]) == 1
    assert capsys.readouterr().err == "desktop training failed: bad: history\n"

    monkeypatch.setattr("omnitensor.training.cli.revoke_desktop_history", lambda *_args: True)
    assert desktop_revoke_main(
        [
            "--history",
            "history.jsonl",
            "--confirm-delete",
            DESKTOP_REVOCATION_CONFIRMATION,
        ]
    ) == 0
    assert capsys.readouterr().out == '{\n  "removed": true\n}\n'
    monkeypatch.setattr(
        "omnitensor.training.cli.revoke_desktop_history",
        lambda *_args: (_item for _item in ()).throw(TrainingError("bad", "revoke")),
    )
    assert desktop_revoke_main(
        ["--history", "bad", "--confirm-delete", DESKTOP_REVOCATION_CONFIRMATION]
    ) == 1
    assert capsys.readouterr().err == "desktop history revocation failed: bad: revoke\n"


@given(
    workspaces=st.integers(min_value=1, max_value=64),
    windows=st.integers(min_value=1, max_value=64),
    active_seed=st.integers(min_value=0, max_value=10_000),
    suggestion=st.sampled_from(("keep", "tile-left", "maximize")),
)
def test_content_free_feature_property(workspaces, windows, active_seed, suggestion):
    value = {
        **document(0, suggestion),
        "workspaceCount": workspaces,
        "activeWorkspace": active_seed % workspaces,
        "windowCount": windows,
        "visibleWindows": windows,
    }
    example = _example_from_document(value)
    assert len(example.features) == len(DESKTOP_FEATURES)
    assert all(math.isfinite(feature) and feature >= 0 for feature in example.features)
