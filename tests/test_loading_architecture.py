from __future__ import annotations

import asyncio
import os
import pickle
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from importrules import forbidden_imports, imported_modules

import omnitensor.job_ports as job_ports
from omnitensor.plugins import loading, worker_specs
from omnitensor.plugins import loading_accelerator as accelerator
from omnitensor.plugins import loading_staging as staging
from omnitensor.plugins.sandbox import SELECTED_FILES_PERMISSION
from omnitensor.plugins.supervisor_session import PluginWorkerError

ROOT = Path(__file__).parents[1]
LEGACY_MODULE = "omnitensor.plugins.loading"


def test_loading_facade_keeps_exact_contract_and_new_owner_identities():
    assert loading.JobDispatchError is job_ports.JobDispatchError
    assert loading.JobDispatchError.__module__ == "omnitensor.job_ports"
    assert loading.ArtifactProvider is worker_specs.ArtifactProvider
    assert loading.ArtifactProvider.__module__ == LEGACY_MODULE
    assert pickle.loads(pickle.dumps(loading.ArtifactProvider)) is loading.ArtifactProvider

    assert loading._external_worker_spec is worker_specs.external_worker_spec
    assert loading._worker_argv is worker_specs.worker_argv
    assert loading._artifact_bootstrap is worker_specs.artifact_bootstrap
    assert loading._artifact_paths is worker_specs.artifact_paths
    assert loading._prepare_worker_state_root is worker_specs.prepare_worker_state_root
    assert loading._plugin_state_path is worker_specs.plugin_state_path
    assert loading._resolved_artifacts is worker_specs.resolved_artifacts
    assert loading._accelerator_lease_path is accelerator.accelerator_lease_path
    assert loading._accelerator_paths is accelerator.accelerator_paths
    assert loading._vulkan_sysfs_resources is accelerator.vulkan_sysfs_resources
    assert loading._prepare_selected_files_root is staging.prepare_selected_files_root
    assert loading._staged_source_name is staging.staged_source_name
    assert loading._canonical_selected_source is staging.canonical_selected_source
    assert loading._open_selected_source is staging.open_selected_source

    for contract in (
        loading.PermissionGrantSource,
        loading.ReloadablePermissionGrantSource,
        loading.WorkerRevoker,
        loading.InstalledPluginSnapshot,
        loading.InstalledPluginRuntime,
    ):
        assert contract.__module__ == LEGACY_MODULE
        assert pickle.loads(pickle.dumps(contract)) is contract


def test_loading_leaves_import_before_facade_without_cycles():
    leaf_names = ("loading_staging", "loading_accelerator", "worker_specs")
    assert (
        forbidden_imports(
            [ROOT / f"src/omnitensor/plugins/{name}.py" for name in leaf_names],
            ["omnitensor.plugins.loading"],
            source_root=ROOT / "src",
        )
        == []
    )

    statement = ";".join(f"import omnitensor.plugins.{name}" for name in (*leaf_names, "loading"))
    completed = subprocess.run(
        [sys.executable, "-c", statement],
        check=False,
        capture_output=True,
        text=True,
    )
    assert (completed.returncode, completed.stderr) == (0, "")


def test_consent_uses_canonical_discovery_owners():
    consent = ROOT / "src/omnitensor/consent.py"
    assert (
        forbidden_imports([consent], ["omnitensor.plugins.loading"], source_root=ROOT / "src") == []
    )
    assert {
        "omnitensor.plugins.discovery",
        "omnitensor.plugins.identity",
        "omnitensor.plugins.manifest_compatibility",
    } <= imported_modules(consent, ROOT / "src")


def test_staging_cleans_on_base_exception(tmp_path):
    class Abort(BaseException):
        pass

    root = tmp_path / "broker"
    root.mkdir()

    def abort(*_args):
        raise Abort("cancelled")

    with pytest.raises(Abort, match="cancelled"):
        staging.stage_selected_sources(
            root,
            "external-example",
            "job-1",
            {"sources": ["/selected/source"]},
            copier=abort,
        )

    assert list((root / "external-example").iterdir()) == []


def test_cancelled_runtime_waits_for_staging_thread_and_removes_result(tmp_path, monkeypatch):
    broker = tmp_path / "broker"
    created = tmp_path / "completed-stage"
    started = threading.Event()
    release = threading.Event()

    def delayed_stage(_root, _plugin_id, _job_id, payload):
        created.mkdir()
        started.set()
        release.wait(timeout=5)
        return dict(payload), created

    monkeypatch.setattr(loading, "_stage_selected_sources", delayed_stage)
    runtime = loading.InstalledPluginRuntime(tmp_path, selected_files_root=broker)
    runtime._granted = {"external-example": frozenset({SELECTED_FILES_PERMISSION})}

    async def scenario():
        task = asyncio.create_task(
            runtime._stage_payload("job-1", "external-example", {"sources": ["source"]})
        )
        while not started.is_set():
            await asyncio.sleep(0)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    assert not created.exists()


def test_repeated_cancellation_cannot_abandon_staging_result(tmp_path, monkeypatch):
    broker = tmp_path / "broker"
    created = tmp_path / "completed-stage"
    started = threading.Event()
    release = threading.Event()

    def delayed_stage(_root, _plugin_id, _job_id, payload):
        created.mkdir()
        started.set()
        release.wait(timeout=5)
        return dict(payload), created

    monkeypatch.setattr(loading, "_stage_selected_sources", delayed_stage)
    runtime = loading.InstalledPluginRuntime(tmp_path, selected_files_root=broker)
    runtime._granted = {"external-example": frozenset({SELECTED_FILES_PERMISSION})}

    async def scenario():
        task = asyncio.create_task(
            runtime._stage_payload("job-1", "external-example", {"sources": ["source"]})
        )
        while not started.is_set():
            await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())

    assert not created.exists()


def test_selected_copy_refuses_ancestor_swap_before_open(tmp_path, monkeypatch):
    selected = tmp_path / "selected"
    selected.mkdir()
    source = selected / "source.txt"
    source.write_bytes(b"first")
    staged = tmp_path / "staged"
    staged.mkdir()
    retired = tmp_path / "retired"
    open_selected = staging.open_selected_source

    def swap_ancestor(candidate):
        selected.rename(retired)
        selected.mkdir()
        source.write_bytes(b"other")
        return open_selected(candidate)

    monkeypatch.setattr(staging, "open_selected_source", swap_ancestor)

    with pytest.raises(PluginWorkerError) as changed:
        staging.copy_selected_source(str(source), staged, 0, set())

    assert (changed.value.code, changed.value.detail) == (
        "selected-file-changed",
        "selected source changed while it was copied",
    )


def test_selected_copy_refuses_same_size_content_change_with_restored_mtime(tmp_path, monkeypatch):
    source = tmp_path / "source.txt"
    source.write_bytes(b"first")
    before = source.stat()
    staged = tmp_path / "staged"
    staged.mkdir()

    def mutate_after_copy(reader, writer, *, length):
        assert length == 1024 * 1024
        writer.write(reader.read())
        source.write_bytes(b"other")
        os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))

    monkeypatch.setattr(staging.shutil, "copyfileobj", mutate_after_copy)

    with pytest.raises(PluginWorkerError) as changed:
        staging.copy_selected_source(str(source), staged, 0, set())

    assert (changed.value.code, changed.value.detail) == (
        "selected-file-changed",
        "selected source changed while it was copied",
    )


def test_selected_copy_bounds_concurrent_growth(tmp_path, monkeypatch):
    source = tmp_path / "source.txt"
    source.write_bytes(b"first")
    staged = tmp_path / "staged"
    staged.mkdir()
    real_copy = staging.shutil.copyfileobj

    def grow_before_copy(reader, writer, *, length):
        source.write_bytes(b"first-and-unbounded-growth")
        real_copy(reader, writer, length=length)

    monkeypatch.setattr(staging.shutil, "copyfileobj", grow_before_copy)

    with pytest.raises(PluginWorkerError) as changed:
        staging.copy_selected_source(str(source), staged, 0, set())

    assert (changed.value.code, changed.value.detail) == (
        "selected-file-changed",
        "selected source changed while it was copied",
    )
    assert (staged / "00" / "source.txt").stat().st_size == 6
