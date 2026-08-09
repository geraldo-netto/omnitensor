from __future__ import annotations

import json

import pytest

from omnitensor.atomicio import write_json_atomic


def test_write_creates_parents_and_round_trips(tmp_path):
    target = tmp_path / "nested/dir/document.json"
    write_json_atomic(target, {"a": 1}, prefix=".doc-")
    assert json.loads(target.read_text()) == {"a": 1}


def test_write_replaces_existing_content_atomically(tmp_path):
    target = tmp_path / "document.json"
    write_json_atomic(target, {"generation": 1}, prefix=".doc-")
    write_json_atomic(target, {"generation": 2}, prefix=".doc-")
    assert json.loads(target.read_text()) == {"generation": 2}
    assert [entry.name for entry in tmp_path.iterdir()] == ["document.json"]


def test_failed_serialization_cleans_up_and_preserves_previous(tmp_path):
    target = tmp_path / "document.json"
    write_json_atomic(target, {"ok": True}, prefix=".doc-")
    with pytest.raises(TypeError):
        write_json_atomic(target, {"bad": {1, 2}}, prefix=".doc-")
    assert json.loads(target.read_text()) == {"ok": True}
    assert [entry.name for entry in tmp_path.iterdir()] == ["document.json"]
