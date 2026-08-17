from __future__ import annotations

import fcntl
import multiprocessing
import os
import time
from pathlib import Path

import pytest

from omnitensor.plugins.artifact_installation import (
    ARTIFACT_STORE_LOCK_FILE,
    artifact_store_lock,
)
from omnitensor.storelock import store_lock


def test_lock_creates_the_store_and_its_lock_file(tmp_path):
    root = tmp_path / "store"
    with store_lock(root, ".store.lock"):
        assert (root / ".store.lock").is_file()
    assert (root / ".store.lock").stat().st_mode & 0o777 == 0o600


def test_lock_is_reusable_after_release(tmp_path):
    for _ in range(3):
        with store_lock(tmp_path, ".store.lock"):
            pass


def test_lock_timeout_bounds_a_contended_wait(tmp_path):
    lock = tmp_path / ".store.lock"
    descriptor = os.open(lock, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    started = time.monotonic()
    try:
        with (
            pytest.raises(TimeoutError, match="timed out acquiring store lock"),
            store_lock(tmp_path, lock.name, timeout_seconds=0.02),
        ):
            pass
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    assert time.monotonic() - started < 0.2


def test_lock_releases_when_the_block_raises(tmp_path):
    with pytest.raises(RuntimeError), store_lock(tmp_path, ".store.lock"):
        raise RuntimeError("boom")
    with store_lock(tmp_path, ".store.lock"):
        pass


def test_lock_refuses_a_symlinked_lock_file(tmp_path):
    (tmp_path / "elsewhere").write_bytes(b"")
    (tmp_path / ".store.lock").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(OSError), store_lock(tmp_path, ".store.lock"):
        pass


def test_lock_refuses_a_lock_path_that_is_not_a_regular_file(tmp_path):
    (tmp_path / ".store.lock").mkdir()
    with pytest.raises(OSError), store_lock(tmp_path, ".store.lock"):
        pass


def test_lock_refuses_a_fifo_with_the_stable_nonregular_detail(tmp_path):
    lock = tmp_path / ".store.lock"
    os.mkfifo(lock)

    with pytest.raises(OSError) as caught, store_lock(tmp_path, lock.name):
        pass

    assert str(caught.value) == "store lock is not a regular file: .store.lock"


def _hold_then_append(root: str, name: str, log: str, started, hold: float) -> None:
    with store_lock(Path(root), name):
        started.set()
        time.sleep(hold)
        with open(log, "a", encoding="utf-8") as stream:
            stream.write("child\n")


def test_lock_serializes_two_processes(tmp_path):
    """A second holder must not enter the section until the first leaves."""
    log = tmp_path / "order.log"
    context = multiprocessing.get_context("spawn")
    started = context.Event()
    child = context.Process(
        target=_hold_then_append,
        args=(str(tmp_path), ".store.lock", str(log), started, 0.5),
    )
    child.start()
    try:
        assert started.wait(timeout=30)
        with store_lock(tmp_path, ".store.lock"):
            log.write_text(log.read_text(encoding="utf-8") + "parent\n", encoding="utf-8")
    finally:
        child.join(timeout=30)
    assert log.read_text(encoding="utf-8").split() == ["child", "parent"]


def test_artifact_store_lock_keeps_its_established_lock_file(tmp_path):
    with artifact_store_lock(tmp_path):
        assert (tmp_path / ARTIFACT_STORE_LOCK_FILE).is_file()
