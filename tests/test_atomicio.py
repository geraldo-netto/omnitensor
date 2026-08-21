from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

import omnitensor.atomicio as atomicio
from omnitensor.atomicio import (
    JsonTooLargeError,
    fsync_directory,
    read_json_bounded,
    remove_durable,
    write_bytes_atomic,
    write_json_atomic,
)


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


def test_json_write_preserves_exact_compact_utf8_bytes(tmp_path):
    target = tmp_path / "document.json"

    write_json_atomic(target, {"b": "é", "a": 1}, prefix=".doc-")

    assert target.read_bytes() == b'{"b":"\\u00e9","a":1}'


def test_atomic_bytes_replace_content_and_apply_the_requested_mode(tmp_path):
    target = tmp_path / "payload.bin"
    target.write_bytes(b"old")

    write_bytes_atomic(target, b"new", 0o640)

    assert target.read_bytes() == b"new"
    assert target.stat().st_mode & 0o777 == 0o640
    assert [entry.name for entry in tmp_path.iterdir()] == ["payload.bin"]


def test_atomic_bytes_exclusive_publish_refuses_every_existing_entry(tmp_path):
    target = tmp_path / "payload.bin"
    target.write_bytes(b"old")

    with pytest.raises(FileExistsError):
        write_bytes_atomic(target, b"new", 0o600, replace=False, prefix=".exclusive-")

    assert target.read_bytes() == b"old"
    assert [entry.name for entry in tmp_path.iterdir()] == ["payload.bin"]


@pytest.mark.parametrize("replace", [True, False])
def test_atomic_bytes_stage_sync_and_publish_order(tmp_path, monkeypatch, replace):
    target = tmp_path / "payload.bin"
    calls = []
    real_mkstemp = atomicio.tempfile.mkstemp
    real_replace = atomicio.os.replace
    real_link = atomicio.os.link

    def tracked_mkstemp(*, dir, prefix):
        descriptor, name = real_mkstemp(dir=dir, prefix=prefix)
        calls.append(("stage", Path(name), Path(dir), prefix))
        return descriptor, name

    def tracked_replace(source, destination):
        calls.append(("replace", Path(source), Path(destination)))
        return real_replace(source, destination)

    def tracked_link(source, destination, *, follow_symlinks):
        calls.append(("link", Path(source), Path(destination), follow_symlinks))
        return real_link(source, destination, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(atomicio.tempfile, "mkstemp", tracked_mkstemp)
    monkeypatch.setattr(
        atomicio.os,
        "fsync",
        lambda descriptor: calls.append(("file-fsync", descriptor)),
    )
    monkeypatch.setattr(atomicio.os, "replace", tracked_replace)
    monkeypatch.setattr(atomicio.os, "link", tracked_link)
    monkeypatch.setattr(
        atomicio,
        "fsync_directory",
        lambda directory: calls.append(("directory-fsync", Path(directory))),
    )

    write_bytes_atomic(
        target,
        b"payload",
        0o600,
        replace=replace,
        prefix=".ordered-",
    )

    stage = calls[0][1]
    assert calls[0] == ("stage", stage, target.parent, ".ordered-")
    assert calls[1][0] == "file-fsync"
    if replace:
        assert calls[2] == ("replace", stage, target)
    else:
        assert calls[2] == ("link", stage, target, False)
    assert calls[3] == ("directory-fsync", target.parent)
    assert stage.parent == target.parent
    assert not stage.exists()
    assert target.read_bytes() == b"payload"


@given(payload=st.binary(max_size=4096), mode=st.sampled_from([0o600, 0o640, 0o644]))
def test_atomic_bytes_property_round_trips_exact_payload_and_mode(tmp_path_factory, payload, mode):
    root = tmp_path_factory.mktemp("atomic-bytes")
    target = root / "nested/payload.bin"

    write_bytes_atomic(target, payload, mode)

    assert target.read_bytes() == payload
    assert target.stat().st_mode & 0o777 == mode


def test_atomic_exclusive_publish_has_one_winner_under_a_race(tmp_path):
    target = tmp_path / "winner.bin"

    def publish(payload):
        try:
            write_bytes_atomic(
                target,
                payload,
                0o600,
                replace=False,
                prefix=".race-",
            )
            return "published"
        except FileExistsError:
            return "exists"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(publish, (b"first", b"second")))

    assert sorted(outcomes) == ["exists", "published"]
    assert target.read_bytes() in {b"first", b"second"}
    assert [entry.name for entry in tmp_path.iterdir()] == ["winner.bin"]


def test_atomic_bytes_cleans_stage_and_preserves_target_on_baseexception(tmp_path, monkeypatch):
    class Interrupted(BaseException):
        pass

    target = tmp_path / "payload.bin"
    target.write_bytes(b"old")

    def interrupt_sync(_descriptor):
        raise Interrupted

    monkeypatch.setattr(atomicio.os, "fsync", interrupt_sync)

    with pytest.raises(Interrupted):
        write_bytes_atomic(target, b"new", 0o600, prefix=".interrupted-")

    assert target.read_bytes() == b"old"
    assert [entry.name for entry in tmp_path.iterdir()] == ["payload.bin"]


def test_atomic_bytes_cleans_stage_when_exclusive_publication_fails(tmp_path, monkeypatch):
    target = tmp_path / "payload.bin"

    def refuse_link(*_args, **_kwargs):
        raise OSError("link unavailable")

    monkeypatch.setattr(atomicio.os, "link", refuse_link)

    with pytest.raises(OSError, match="link unavailable"):
        write_bytes_atomic(
            target,
            b"new",
            0o600,
            replace=False,
            prefix=".refused-",
        )

    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_directory_sync_uses_a_read_only_directory_descriptor_and_closes(monkeypatch):
    calls = []
    monkeypatch.setattr(
        atomicio.os,
        "open",
        lambda directory, flags: calls.append(("open", directory, flags)) or 17,
    )
    monkeypatch.setattr(atomicio.os, "fsync", lambda fd: calls.append(("fsync", fd)))
    monkeypatch.setattr(atomicio.os, "close", lambda fd: calls.append(("close", fd)))

    fsync_directory(Path("/configured/output"))

    assert calls == [
        (
            "open",
            Path("/configured/output"),
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        ),
        ("fsync", 17),
        ("close", 17),
    ]


def test_directory_sync_is_best_effort(monkeypatch):
    monkeypatch.setattr(
        atomicio.os,
        "open",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("unsupported")),
    )

    assert fsync_directory(Path("/configured/output")) is None


def test_durable_remove_unlinks_then_syncs_the_parent(tmp_path, monkeypatch):
    target = tmp_path / "obsolete"
    target.write_bytes(b"old")
    synced = []
    monkeypatch.setattr(
        atomicio,
        "fsync_directory",
        lambda directory: synced.append(Path(directory)),
    )

    remove_durable(target)

    assert not target.exists()
    assert synced == [tmp_path]


def test_consumers_keep_patchable_aliases_to_the_canonical_primitives():
    from omnitensor import event_cli, lowlight, publisher_identity
    from omnitensor.plugins import artifact_cache, artifact_installation

    assert event_cli._write_bytes_atomic is write_bytes_atomic
    assert lowlight._write_bytes_atomic is write_bytes_atomic
    assert publisher_identity._write_bytes_atomic is write_bytes_atomic
    assert lowlight._fsync_directory is fsync_directory
    assert artifact_cache._fsync_directory is fsync_directory
    assert artifact_installation._fsync_directory is fsync_directory


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


def test_a_failed_publish_does_not_close_the_descriptor_twice(tmp_path, monkeypatch):
    """After fdopen takes ownership, the raw fd is not ours any more (OMNI-0602).

    The failure handler used to os.close(handle) unconditionally; once the
    stream had already closed that descriptor, the number could belong to
    anything a worker thread opened in the meantime — a live control-socket
    connection, a lease — and the handler closed it.
    """
    from omnitensor import atomicio

    target = tmp_path / "exists.bin"
    target.write_bytes(b"already here")
    explicitly_closed = []
    real_close = atomicio.os.close
    monkeypatch.setattr(
        atomicio.os, "close", lambda fd: (explicitly_closed.append(fd), real_close(fd))
    )
    staged = []
    real_mkstemp = atomicio.tempfile.mkstemp

    def recording_mkstemp(**kwargs):
        handle, name = real_mkstemp(**kwargs)
        staged.append(handle)
        return handle, name

    monkeypatch.setattr(atomicio.tempfile, "mkstemp", recording_mkstemp)

    # Exclusive publication against an existing target: os.link raises
    # after the stream has closed the staging descriptor.
    with pytest.raises(FileExistsError):
        atomicio.write_bytes_atomic(target, b"new", 0o600, replace=False)

    assert staged and staged[0] not in explicitly_closed
    assert target.read_bytes() == b"already here"
    # The staging file itself is still cleaned up.
    assert list(tmp_path.glob(".exists.bin.*")) == []
