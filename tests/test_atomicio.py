from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.atomicio import JsonTooLargeError, read_json_bounded, write_json_atomic


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


def test_bounded_read_round_trips_a_document_at_the_limit(tmp_path):
    target = tmp_path / "document.json"
    write_json_atomic(target, {"a": 1}, prefix=".doc-")
    exact = target.stat().st_size
    assert read_json_bounded(target, exact) == {"a": 1}


def test_bounded_read_rejects_a_document_one_byte_over_the_limit(tmp_path):
    target = tmp_path / "document.json"
    write_json_atomic(target, {"a": 1}, prefix=".doc-")
    with pytest.raises(JsonTooLargeError, match="exceeds"):
        read_json_bounded(target, target.stat().st_size - 1)


def test_bounded_read_never_reads_past_the_limit(tmp_path, monkeypatch):
    """The bound must hold on the descriptor, not on a prior stat()."""
    target = tmp_path / "document.json"
    target.write_bytes(b"x" * 4096)
    requested: list[int] = []
    original = Path.open

    class _Recorder:
        def __init__(self, stream):
            self._stream = stream

        def read(self, size=-1):
            requested.append(size)
            return self._stream.read(size)

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return self._stream.__exit__(*exc_info)

    monkeypatch.setattr(Path, "open", lambda self, *a, **kw: _Recorder(original(self, *a, **kw)))
    with pytest.raises(JsonTooLargeError):
        read_json_bounded(target, 16)
    assert requested == [17]


def test_bounded_read_survives_a_file_that_grows_after_the_open(tmp_path, monkeypatch):
    """A stat-then-read reader would size this file at 2 bytes and read 4 KiB."""
    target = tmp_path / "document.json"
    target.write_bytes(b"{}")
    original = Path.open

    def grow_then_open(self, *args, **kwargs):
        stream = original(self, *args, **kwargs)
        if self == target:
            descriptor = os.open(target, os.O_WRONLY | os.O_TRUNC)
            try:
                os.write(descriptor, b"x" * 4096)
            finally:
                os.close(descriptor)
        return stream

    monkeypatch.setattr(Path, "open", grow_then_open)
    with pytest.raises(JsonTooLargeError):
        read_json_bounded(target, 64)


def test_bounded_read_propagates_missing_files_as_oserror(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_json_bounded(tmp_path / "absent.json", 1024)


def test_bounded_read_rejects_invalid_json(tmp_path):
    target = tmp_path / "document.json"
    target.write_bytes(b"{nope")
    with pytest.raises(ValueError):
        read_json_bounded(target, 1024)


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, "8"])
def test_bounded_read_rejects_a_nonsensical_limit(tmp_path, bad):
    target = tmp_path / "document.json"
    write_json_atomic(target, {"a": 1}, prefix=".doc-")
    with pytest.raises(ValueError, match="max_bytes"):
        read_json_bounded(target, bad)


@given(payload=st.dictionaries(st.text(max_size=8), st.integers(), max_size=8))
def test_bounded_read_accepts_exactly_what_fits(tmp_path_factory, payload):
    target = tmp_path_factory.mktemp("bounded") / "document.json"
    write_json_atomic(target, payload, prefix=".doc-")
    size = target.stat().st_size
    assert read_json_bounded(target, size) == payload
    with pytest.raises(JsonTooLargeError):
        read_json_bounded(target, size - 1)
