from __future__ import annotations

import asyncio
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from conftest import sample_plugin_manifest

from omnitensor.discovery import Device
from omnitensor.plugins import (
    DEFAULT_MAX_FRAME_BYTES,
    MAX_WORKER_IMPORT_PATHS,
    HandshakeOffer,
    InstalledPluginRuntime,
    InstalledPluginSnapshot,
    IPCFrame,
    IPCProtocolError,
    PluginCatalog,
    PluginProgress,
    PluginRequest,
    PluginResult,
    PluginResultStatus,
    PluginSource,
    PluginWorkerError,
    ResolvedPlugin,
    WorkerMessageType,
    WorkerState,
    WorkerStatus,
    decode_frame,
    encode_frame,
    entry_points_from_distributions,
    execute_frame,
    external_worker_specs,
    handshake_frame,
)
from omnitensor.plugins import loading as loading_module
from omnitensor.plugins import worker as worker_module
from omnitensor.plugins.protocol import (
    PluginContext,
    PluginHealth,
    PluginHealthStatus,
    WorkloadPlugin,
)
from omnitensor.plugins.worker import (
    ExternalPluginLoadError,
    _read_exact,
    _read_frame,
    load_external_plugin,
    serve_worker,
    serve_worker_requests,
)
from omnitensor.sdk import CancellationController, cancelled_result, succeeded_result
from omnitensor.service import OmniTensorService


class TrackingPlugin:
    plugin_id = "external-example"

    def __init__(self):
        self.events = []

    async def start(self, context):
        self.events.append(("start", context))

    async def health(self):
        return PluginHealth(PluginHealthStatus.READY, "ready", 0)

    async def stop(self):
        self.events.append(("stop", None))

    async def execute(self, request, cancellation, progress):
        raise AssertionError("worker bootstrap must not execute jobs")


class FakeDistribution:
    def __init__(self, name="external-dist"):
        self.name = name


class FakeEntryPoint:
    def __init__(
        self,
        *,
        name="external-example",
        value="external_package:Plugin",
        distribution="external-dist",
        factory=TrackingPlugin,
    ):
        self.name = name
        self.value = value
        self.dist = FakeDistribution(distribution) if distribution is not None else None
        self._factory = factory
        self.load_calls = 0

    def load(self):
        self.load_calls += 1
        return self._factory


def _resolved(source=PluginSource.EXTERNAL, tmp_path=Path("/tmp")):
    manifest = sample_plugin_manifest("external-example")
    manifest["plugin"]["protocol"] = {
        "minimum": 2,
        "maximum": 4,
        "capabilities": ["cancel", "execute", "health", "progress"],
    }
    return ResolvedPlugin(
        "external-example",
        "1.0.0",
        source,
        "external-dist",
        "1.0.0",
        "external-example",
        "external_package:Plugin" if source is PluginSource.EXTERNAL else None,
        tmp_path / "omnitensor-plugin.json",
        manifest,
    )


def _frames(data: bytes):
    frames = []
    offset = 0
    while offset < len(data):
        size = int.from_bytes(data[offset : offset + 4], "big")
        end = offset + 4 + size
        frames.append(decode_frame(data[offset:end]))
        offset = end
    return frames


def test_external_worker_specs_are_deterministic_and_do_not_import_plugins(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    external = _resolved(tmp_path=tmp_path)
    bundled = _resolved(PluginSource.BUNDLED, tmp_path)

    specs = external_worker_specs(
        (bundled, external),
        python_executable="/usr/bin/python3",
        worker_import_paths=(site,),
    )

    assert len(specs) == 1
    spec = specs[0]
    assert spec.plugin_id == "external-example"
    assert spec.minimum_protocol == 2
    assert spec.maximum_protocol == 4
    assert spec.capabilities == frozenset({"cancel", "execute", "health", "progress"})
    assert spec.sandbox is not None
    assert spec.sandbox.python_path == str(
        Path(__file__).resolve().parents[1] / "src"
    )
    assert spec.argv == (
        "/usr/bin/python3",
        "-m",
        "omnitensor.plugins.worker",
        "--plugin-id",
        "external-example",
        "--entry-point",
        "external-example",
        "--target",
        "external_package:Plugin",
        "--distribution",
        "external-dist",
        "--import-path",
        str(site),
    )


def test_external_worker_specs_pass_only_granted_declared_permissions(tmp_path):
    site = tmp_path / "site"
    source = tmp_path / "source"
    site.mkdir()
    source.mkdir()
    plugin = _resolved(tmp_path=tmp_path)
    permission = f"read:{source}"
    plugin.manifest["plugin"]["permissions"] = [permission]

    [spec] = external_worker_specs(
        (plugin,),
        worker_import_paths=(site,),
        granted_permissions={plugin.plugin_id: {permission}},
    )

    assert spec.argv[-2:] == ("--permission", permission)
    assert spec.sandbox is not None
    assert spec.sandbox.read_paths == (str(source),)


def test_external_worker_specs_fail_closed_on_undeclared_grant(tmp_path):
    plugin = _resolved(tmp_path=tmp_path)
    with pytest.raises(ValueError, match="undeclared"):
        external_worker_specs(
            (plugin,),
            granted_permissions={plugin.plugin_id: {f"read:{tmp_path}"}},
        )


def test_installed_runtime_admits_dispatches_and_forwards_worker_progress(tmp_path):
    class Supervisor:
        def __init__(self):
            self.requests = []

        def statuses(self):
            return (WorkerStatus("external-example", WorkerState.READY, 10, 1, "ready"),)

        async def execute(self, request, progress):
            self.requests.append(request)
            await progress.report(PluginProgress(request.job_id, "extract", 0.5, "", 11))
            return PluginResult(
                request.job_id,
                PluginResultStatus.SUCCEEDED,
                {"events": []},
                "done",
                12,
            )

    supervisor = Supervisor()
    observed = []
    runtime = InstalledPluginRuntime(
        tmp_path,
        supervisor=supervisor,
        progress_sink=observed.append,
        clock_ms=lambda: 10,
    )
    plugin = _resolved(tmp_path=tmp_path)
    runtime._snapshot = InstalledPluginSnapshot(PluginCatalog((plugin,), ()), supervisor.statuses())
    runtime._granted = {"external-example": frozenset()}

    runtime.admit("external-example", {})
    output = asyncio.run(runtime.dispatch("job-1", "external-example", {}))

    assert runtime.plugin_ids() == frozenset({"external-example"})
    assert output == {"events": []}
    assert supervisor.requests == [
        PluginRequest("job-1", "external-example", "manual", {}, 10, None)
    ]
    assert observed == [PluginProgress("job-1", "extract", 0.5, "", 11)]


def test_selected_sources_are_brokered_into_worker_sandbox_and_removed(tmp_path):
    source = tmp_path / "private.txt"
    source.write_text("private event", encoding="utf-8")
    broker = tmp_path / "broker"

    class Supervisor:
        requests = []

        def statuses(self):
            return (
                WorkerStatus("external-example", WorkerState.READY, 10, 1, "ready"),
            )

        async def execute(self, request, _progress):
            self.requests.append(request)
            [staged] = request.payload["sources"]
            assert Path(staged).is_relative_to(broker)
            assert Path(staged) != source
            assert Path(staged).read_text(encoding="utf-8") == "private event"
            return PluginResult(
                request.job_id,
                PluginResultStatus.SUCCEEDED,
                {"events": []},
                "done",
                12,
            )

    supervisor = Supervisor()
    runtime = InstalledPluginRuntime(
        tmp_path,
        supervisor=supervisor,
        selected_files_root=broker,
        clock_ms=lambda: 10,
    )
    plugin = _resolved(tmp_path=tmp_path)
    plugin.manifest["plugin"]["permissions"] = ["files:read-selected"]
    plugin.manifest["plugin"]["schemas"]["input"] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["sources"],
        "properties": {"sources": {"type": "array", "minItems": 1}},
    }
    runtime._snapshot = InstalledPluginSnapshot(
        PluginCatalog((plugin,), ()), supervisor.statuses()
    )
    runtime._granted = {
        "external-example": frozenset({"files:read-selected"})
    }

    output = asyncio.run(
        runtime.dispatch(
            "job-1", "external-example", {"sources": [str(source)]}
        )
    )

    assert output == {"events": []}
    assert source.read_text(encoding="utf-8") == "private event"
    assert list((broker / "external-example").iterdir()) == []


def test_selected_source_broker_refuses_aliases_duplicates_and_missing_root(tmp_path):
    source = tmp_path / "private.txt"
    source.write_text("private event", encoding="utf-8")
    alias = tmp_path / "alias.txt"
    alias.symlink_to(source)

    class Supervisor:
        def statuses(self):
            return (
                WorkerStatus("external-example", WorkerState.READY, 10, 1, "ready"),
            )

        async def execute(self, _request, _progress):
            raise AssertionError("invalid selected files must not reach the worker")

    plugin = _resolved(tmp_path=tmp_path)
    plugin.manifest["plugin"]["permissions"] = ["files:read-selected"]
    plugin.manifest["plugin"]["schemas"]["input"] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["sources"],
        "properties": {"sources": {"type": "array", "minItems": 1}},
    }

    def runtime(root):
        subject = InstalledPluginRuntime(
            tmp_path,
            supervisor=Supervisor(),
            selected_files_root=root,
        )
        subject._snapshot = InstalledPluginSnapshot(
            PluginCatalog((plugin,), ()), subject._supervisor.statuses()
        )
        subject._granted = {
            "external-example": frozenset({"files:read-selected"})
        }
        return subject

    for sources, code in (
        ([str(alias)], "selected-file-invalid"),
        ([str(source), str(source)], "selected-file-invalid"),
    ):
        with pytest.raises(PluginWorkerError) as error:
            asyncio.run(
                runtime(tmp_path / "broker").dispatch(
                    "job-1", "external-example", {"sources": sources}
                )
            )
        assert error.value.code == code

    with pytest.raises(PluginWorkerError) as unavailable:
        asyncio.run(
            runtime(None).dispatch(
                "job-1", "external-example", {"sources": [str(source)]}
            )
        )
    assert unavailable.value.code == "selected-files-unavailable"


def test_selected_source_broker_refuses_missing_empty_and_unopenable_files(
    tmp_path, monkeypatch
):
    missing = tmp_path / "missing.txt"
    with pytest.raises(PluginWorkerError) as absent:
        loading_module._canonical_selected_source(missing)
    assert absent.value.code == "selected-file-unavailable"

    empty = tmp_path / "empty.txt"
    empty.touch()
    descriptor = loading_module._open_selected_source(empty)
    try:
        with pytest.raises(PluginWorkerError) as invalid:
            loading_module._validate_selected_source_stat(os.fstat(descriptor))
        assert invalid.value.code == "selected-file-invalid"
    finally:
        os.close(descriptor)

    monkeypatch.setattr(loading_module.os, "open", lambda *_args: (_ for _ in ()).throw(OSError()))
    with pytest.raises(PluginWorkerError) as unavailable:
        loading_module._open_selected_source(empty)
    assert unavailable.value.code == "selected-file-unavailable"


def test_worker_request_dispatch_refuses_mismatches_and_emits_bounded_errors():
    async def scenario():
        plugin = TrackingPlugin()
        writer = io.BytesIO()
        active = {}
        assert (
            await worker_module._handle_request_frame(
                plugin,
                IPCFrame(1, WorkerMessageType.CANCEL, None, {"reason": "shutdown"}),
                writer,
                1,
                active,
            )
            is False
        )
        wrong = execute_frame(
            PluginRequest("wrong-job", "other-plugin", "manual", {}, 1, None)
        )
        assert await worker_module._handle_request_frame(
            plugin, wrong, writer, 1, active
        )
        token = CancellationController()
        active["duplicate"] = (asyncio.current_task(), token)
        duplicate = execute_frame(
            PluginRequest("duplicate", "external-example", "manual", {}, 1, None)
        )
        assert await worker_module._handle_request_frame(
            plugin, duplicate, writer, 1, active
        )
        assert await worker_module._handle_request_frame(
            plugin,
            IPCFrame(1, WorkerMessageType.CANCEL, "duplicate", {"reason": "cancel"}),
            writer,
            1,
            active,
        )
        assert token.cancelled is True
        assert token.reason == "cancelled by service"
        assert await worker_module._handle_request_frame(
            plugin,
            IPCFrame(1, WorkerMessageType.HEALTH, "health", {}),
            writer,
            1,
            active,
        )
        return writer

    output = _frames(asyncio.run(scenario()).getvalue())
    assert [frame.payload["code"] for frame in output] == [
        "plugin-identity-mismatch",
        "duplicate-request",
        "unsupported-message",
    ]
    assert [frame.request_id for frame in output] == [
        "wrong-job",
        "duplicate",
        "health",
    ]


def test_worker_progress_and_terminal_failures_stay_correlated():
    request = PluginRequest("job-1", "external-example", "manual", {}, 1, None)

    async def scenario():
        with pytest.raises(IPCProtocolError) as mismatch:
            await worker_module._WorkerProgress("job-1", io.BytesIO(), 1).report(
                PluginProgress("other-job", "extract", 0.5, "", 2)
            )
        assert mismatch.value.code == "invalid-progress"

        class WrongResult:
            async def execute(self, *_arguments):
                return PluginResult(
                    "other-job", PluginResultStatus.SUCCEEDED, {}, "", 2
                )

        class Cancelled:
            async def execute(self, *_arguments):
                raise asyncio.CancelledError

        class Broken:
            async def execute(self, *_arguments):
                raise ValueError("private")

        outputs = []
        for plugin in (WrongResult(), Cancelled(), Broken()):
            writer = io.BytesIO()
            await worker_module._execute_request(
                plugin, request, CancellationController(), writer, 1
            )
            outputs.append(_frames(writer.getvalue()))
        return outputs

    wrong, cancelled, broken = asyncio.run(scenario())
    assert wrong[0].payload["code"] == "plugin-execution-failed"
    assert cancelled[0].type is WorkerMessageType.RESULT
    assert cancelled[0].payload["status"] == "cancelled"
    assert cancelled[0].payload["detail"] == "plugin request cancelled"
    assert broken[0].payload == {
        "code": "plugin-execution-failed",
        "detail": "worker refused the message",
    }


def test_installed_runtime_admission_is_fail_closed(tmp_path):
    class Supervisor:
        def __init__(self, state):
            self.state = state

        def statuses(self):
            return (WorkerStatus("external-example", self.state, None, 1, "state"),)

    from omnitensor.jobs import JobDispatchError

    supervisor = Supervisor(WorkerState.FAILED)
    runtime = InstalledPluginRuntime(tmp_path, supervisor=supervisor)
    plugin = _resolved(tmp_path=tmp_path)
    runtime._snapshot = InstalledPluginSnapshot(PluginCatalog((plugin,), ()), supervisor.statuses())
    runtime._granted = {"external-example": frozenset()}

    with pytest.raises(JobDispatchError) as unavailable:
        runtime.admit("external-example", {})
    assert unavailable.value.code == "worker-unavailable"

    supervisor.state = WorkerState.READY
    plugin.manifest["plugin"]["permissions"] = ["files:read-selected"]
    with pytest.raises(JobDispatchError) as consent:
        runtime.admit("external-example", {})
    assert consent.value.code == "consent-missing"
    runtime._granted = {"external-example": frozenset({"files:read-selected"})}
    plugin.manifest["plugin"]["schemas"]["input"] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["sources"],
        "properties": {"sources": {"type": "array", "minItems": 1}},
    }
    with pytest.raises(JobDispatchError) as payload:
        runtime.admit("external-example", {})
    assert payload.value.code == "payload-invalid"
    plugin.manifest["plugin"]["schemas"]["input"] = {"type": "not-a-json-type"}
    with pytest.raises(JobDispatchError) as contract:
        runtime.admit("external-example", {"sources": ["/private.txt"]})
    assert contract.value.code == "plugin-contract-invalid"


def test_installed_runtime_admission_errors_are_stable_contracts(tmp_path):
    from omnitensor.jobs import JobDispatchError

    class Supervisor:
        statuses_value = ()

        def statuses(self):
            return self.statuses_value

    supervisor = Supervisor()
    runtime = InstalledPluginRuntime(tmp_path, supervisor=supervisor)

    def refused(plugin_id, payload, code, message):
        with pytest.raises(JobDispatchError) as error:
            runtime.admit(plugin_id, payload)
        assert (error.value.code, error.value.message) == (code, message)

    refused("missing", {}, "workload-unknown", "No installed plugin: missing")

    plugin = _resolved(tmp_path=tmp_path)
    runtime._snapshot = InstalledPluginSnapshot(PluginCatalog((plugin,), ()), ())
    refused(
        "external-example", {}, "worker-unavailable", "Plugin worker is not ready"
    )

    supervisor.statuses_value = (
        WorkerStatus("external-example", WorkerState.READY, 10, 1, "ready"),
    )
    plugin.manifest["plugin"]["protocol"]["capabilities"] = []
    refused(
        "external-example",
        {},
        "worker-incompatible",
        "Plugin does not declare execution",
    )

    plugin.manifest["plugin"]["protocol"]["capabilities"] = ["execute"]
    plugin.manifest["plugin"]["permissions"] = ["files:read-selected"]
    refused(
        "external-example", {}, "consent-missing", "Plugin permissions are not granted"
    )

    runtime._granted = {
        "external-example": frozenset({"files:read-selected"})
    }
    refused(
        "external-example",
        [],
        "payload-invalid",
        "Plugin payload must be an object",
    )

    plugin.manifest["plugin"]["schemas"]["input"] = {"type": "invalid"}
    refused(
        "external-example",
        {},
        "plugin-contract-invalid",
        "Plugin input schema is invalid",
    )

    plugin.manifest["plugin"]["schemas"]["input"] = {
        "type": "object",
        "required": ["sources"],
    }
    refused(
        "external-example",
        {},
        "payload-invalid",
        "Payload violates the plugin input contract",
    )


def test_installed_runtime_ignores_bundled_identity_when_dispatching(tmp_path):
    bundled = _resolved(PluginSource.BUNDLED, tmp_path)
    runtime = InstalledPluginRuntime(tmp_path)
    runtime._snapshot = InstalledPluginSnapshot(PluginCatalog((bundled,), ()), ())

    assert runtime._plugin("external-example") is None


def test_installed_runtime_start_wires_exact_catalog_grants_and_worker_spec(
    tmp_path, monkeypatch
):
    bundled = _resolved(PluginSource.BUNDLED, tmp_path)
    external = _resolved(tmp_path=tmp_path)
    external.manifest["plugin"]["permissions"] = ["files:read-selected"]
    candidates = (object(),)
    identified = object()
    catalog = PluginCatalog((bundled, external), ())
    workers = (
        WorkerStatus("external-example", WorkerState.READY, 10, 1, "ready"),
    )
    observed = []

    class Grants:
        def active_permissions(self, plugin_id, declared):
            observed.append(("grant", plugin_id, declared))
            return frozenset(declared)

    class Supervisor:
        def statuses(self):
            return workers

        async def start(self, specs):
            observed.append(("start", specs, runtime._snapshot))
            return workers

    def discover(**options):
        observed.append(("discover", options))
        return candidates

    def resolve_identities(value):
        observed.append(("identity", value))
        return identified

    def resolve_compatibility(value):
        observed.append(("compatibility", value))
        return catalog

    expected_specs = (object(),)

    def specs(plugins, **options):
        observed.append(("specs", plugins, options))
        return expected_specs

    monkeypatch.setattr(loading_module, "discover_plugin_metadata", discover)
    monkeypatch.setattr(loading_module, "resolve_plugin_identities", resolve_identities)
    monkeypatch.setattr(
        loading_module, "resolve_plugin_compatibility", resolve_compatibility
    )
    monkeypatch.setattr(loading_module, "external_worker_specs", specs)
    broker = tmp_path / "nested" / "broker"
    runtime = InstalledPluginRuntime(
        tmp_path / "bundled",
        supervisor=Supervisor(),
        entry_points_provider=lambda: (),
        python_executable="/usr/bin/python3",
        worker_import_paths=(tmp_path,),
        grant_source=Grants(),
        selected_files_root=broker,
    )

    snapshot = asyncio.run(runtime.start())

    assert snapshot == InstalledPluginSnapshot(catalog, workers)
    assert observed == [
        (
            "discover",
            {
                "bundled_root": tmp_path / "bundled",
                "entry_points_provider": runtime._entry_points_provider,
            },
        ),
        ("identity", candidates),
        ("compatibility", identified),
        ("grant", "external-example", {"files:read-selected"}),
        (
            "specs",
            catalog.plugins,
            {
                "python_executable": "/usr/bin/python3",
                "worker_import_paths": (str(tmp_path),),
                "granted_permissions": {
                    "external-example": frozenset({"files:read-selected"})
                },
                "selected_files_root": broker.resolve(),
            },
        ),
        (
            "start",
            expected_specs,
            InstalledPluginSnapshot(catalog, ()),
        ),
    ]
    assert runtime._granted == {
        "external-example": frozenset({"files:read-selected"})
    }


def test_installed_runtime_maps_terminal_worker_results_and_cleans_staging(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.txt"
    source.write_text("event", encoding="utf-8")
    broker = tmp_path / "broker"
    removed = []
    real_rmtree = loading_module.shutil.rmtree

    def remove(path, ignore_errors):
        removed.append((path, ignore_errors))
        real_rmtree(path, ignore_errors=ignore_errors)

    monkeypatch.setattr(loading_module.shutil, "rmtree", remove)

    class Supervisor:
        result = PluginResult(
            "job-1", PluginResultStatus.FAILED, {}, "provider refused", 12
        )

        def statuses(self):
            return (
                WorkerStatus("external-example", WorkerState.READY, 10, 1, "ready"),
            )

        async def execute(self, _request, _progress):
            return self.result

    supervisor = Supervisor()
    runtime = InstalledPluginRuntime(
        tmp_path,
        supervisor=supervisor,
        selected_files_root=broker,
    )
    plugin = _resolved(tmp_path=tmp_path)
    plugin.manifest["plugin"]["permissions"] = ["files:read-selected"]
    plugin.manifest["plugin"]["schemas"]["input"] = {"type": "object"}
    runtime._snapshot = InstalledPluginSnapshot(
        PluginCatalog((plugin,), ()), supervisor.statuses()
    )
    runtime._granted = {
        "external-example": frozenset({"files:read-selected"})
    }

    with pytest.raises(PluginWorkerError) as failed:
        asyncio.run(
            runtime.dispatch(
                "job-1", "external-example", {"sources": [str(source)]}
            )
        )
    assert (failed.value.code, failed.value.detail) == (
        "plugin-failed",
        "provider refused",
    )
    assert len(removed) == 1
    assert removed[0][0].parent == broker / "external-example"
    assert removed[0][1] is True

    supervisor.result = PluginResult(
        "job-1", PluginResultStatus.FAILED, {}, "", 12
    )
    with pytest.raises(PluginWorkerError) as defaulted:
        asyncio.run(
            runtime.dispatch(
                "job-1", "external-example", {"sources": [str(source)]}
            )
        )
    assert (defaulted.value.code, defaulted.value.detail) == (
        "plugin-failed",
        "plugin request failed",
    )

    supervisor.result = PluginResult(
        "job-1", PluginResultStatus.CANCELLED, {}, "user", 12
    )
    with pytest.raises(asyncio.CancelledError, match="user"):
        asyncio.run(
            runtime.dispatch(
                "job-1", "external-example", {"sources": [str(source)]}
            )
        )


def test_selected_file_broker_contract_boundaries_and_cleanup(tmp_path, monkeypatch):
    broker = tmp_path / "nested" / "broker"
    prepared = loading_module._prepare_selected_files_root(broker)
    assert prepared == broker.resolve()
    assert prepared.stat().st_mode & 0o777 == 0o700
    with pytest.raises(ValueError) as relative:
        loading_module._prepare_selected_files_root(Path("relative"))
    assert str(relative.value) == "selected_files_root must be absolute"

    for sources in (None, (), [], [""], [1], ["x"] * 33):
        with pytest.raises(PluginWorkerError) as invalid:
            loading_module._stage_selected_sources(
                prepared, "external-example", "job-1", {"sources": sources}
            )
        assert (invalid.value.code, invalid.value.detail) == (
            "selected-files-invalid",
            "sources must name 1-32 selected files",
        )

    source = tmp_path / "event.TXT"
    source.write_text("event", encoding="utf-8")
    for plugin_id, job_id in (("../plugin", "job-1"), ("plugin", "job/1")):
        with pytest.raises(PluginWorkerError) as invalid:
            loading_module._stage_selected_sources(
                prepared, plugin_id, job_id, {"sources": [str(source)]}
            )
        assert (invalid.value.code, invalid.value.detail) == (
            "selected-files-invalid",
            "request identity is invalid",
        )

    rewritten, staged = loading_module._stage_selected_sources(
        prepared,
        "external-example",
        "job-1",
        {"sources": [str(source)], "locale": "en"},
    )
    assert staged.name.startswith("job-1-")
    assert staged.parent == prepared / "external-example"
    assert staged.parent.stat().st_mode & 0o777 == 0o700
    assert rewritten == {
        "sources": [str(staged / "00.txt")],
        "locale": "en",
    }
    assert (staged / "00.txt").read_text(encoding="utf-8") == "event"
    loading_module.shutil.rmtree(staged)

    copied = []

    def fake_copy(path, destination, index, observed):
        copied.append((path, destination, index, observed))
        return destination / f"{index:02d}.txt"

    monkeypatch.setattr(loading_module, "_copy_selected_source", fake_copy)
    rewritten, staged = loading_module._stage_selected_sources(
        prepared,
        "external-example",
        "job-32",
        {"sources": [f"/source-{index}" for index in range(32)]},
    )
    assert len(rewritten["sources"]) == 32
    assert [call[2] for call in copied] == list(range(32))
    assert all(call[1] == staged for call in copied)
    assert all(call[3] is copied[0][3] for call in copied)
    loading_module.shutil.rmtree(staged)

    def fail_copy(*_arguments):
        raise RuntimeError("copy failed")

    monkeypatch.setattr(loading_module, "_copy_selected_source", fail_copy)
    with pytest.raises(RuntimeError, match="copy failed"):
        loading_module._stage_selected_sources(
            prepared, "external-example", "job-fail", {"sources": [str(source)]}
        )
    assert list((prepared / "external-example").iterdir()) == []


def test_selected_file_copy_contracts_are_exact(tmp_path, monkeypatch):
    staged = tmp_path / "staged"
    staged.mkdir()
    first = tmp_path / "FIRST.TXT"
    second = tmp_path / "second.bad-suffix-long"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    observed = set()
    copy_calls = []
    real_copy = loading_module.shutil.copyfileobj

    def copy(reader, writer, length):
        copy_calls.append(length)
        real_copy(reader, writer, length=length)

    monkeypatch.setattr(loading_module.shutil, "copyfileobj", copy)
    first_copy = loading_module._copy_selected_source(first, staged, 0, observed)
    second_copy = loading_module._copy_selected_source(second, staged, 1, observed)
    assert first_copy == staged / "00.txt"
    assert second_copy == staged / "01.bin"
    assert first_copy.read_text(encoding="utf-8") == "first"
    assert second_copy.read_text(encoding="utf-8") == "second"
    assert copy_calls == [1024 * 1024, 1024 * 1024]
    assert len(observed) == 2

    with pytest.raises(PluginWorkerError) as duplicate:
        loading_module._copy_selected_source(first, staged, 2, observed)
    assert (duplicate.value.code, duplicate.value.detail) == (
        "selected-file-invalid",
        "the same selected file appears more than once",
    )


def test_selected_file_helpers_reject_exact_race_and_file_boundaries(
    tmp_path, monkeypatch
):
    source = tmp_path / "source.txt"
    source.write_text("event", encoding="utf-8")
    alias = tmp_path / "alias.txt"
    alias.symlink_to(source)

    for candidate, code, detail in (
        (
            Path("relative"),
            "selected-file-unavailable",
            "selected source cannot be opened",
        ),
        (
            alias,
            "selected-file-invalid",
            "selected source must be a canonical absolute path",
        ),
    ):
        with pytest.raises(PluginWorkerError) as error:
            loading_module._canonical_selected_source(candidate)
        assert (error.value.code, error.value.detail) == (code, detail)

    opened = []

    def open_file(path, flags):
        opened.append((path, flags))
        raise OSError("denied")

    monkeypatch.setattr(loading_module.os, "open", open_file)
    with pytest.raises(PluginWorkerError) as unavailable:
        loading_module._open_selected_source(source)
    assert (unavailable.value.code, unavailable.value.detail) == (
        "selected-file-unavailable",
        "selected source cannot be opened",
    )
    assert opened == [
        (
            source,
            loading_module.os.O_RDONLY
            | loading_module.os.O_CLOEXEC
            | loading_module.os.O_NOFOLLOW,
        )
    ]

    regular = source.stat()
    loading_module._validate_selected_source_stat(
        os.stat_result(
            (
                regular.st_mode,
                regular.st_ino,
                regular.st_dev,
                regular.st_nlink,
                regular.st_uid,
                regular.st_gid,
                loading_module.MAX_SELECTED_SOURCE_BYTES,
                regular.st_atime,
                regular.st_mtime,
                regular.st_ctime,
            )
        )
    )
    for status in (
        os.stat_result((regular.st_mode, 1, 1, 1, 1, 1, 0, 1, 1, 1)),
        os.stat_result(
            (
                regular.st_mode,
                1,
                1,
                1,
                1,
                1,
                loading_module.MAX_SELECTED_SOURCE_BYTES + 1,
                1,
                1,
                1,
            )
        ),
        tmp_path.stat(),
    ):
        with pytest.raises(PluginWorkerError) as invalid:
            loading_module._validate_selected_source_stat(status)
        assert (invalid.value.code, invalid.value.detail) == (
            "selected-file-invalid",
            "selected source must be a bounded regular file",
        )

    destination = tmp_path / "destination"
    destination.write_text("event", encoding="utf-8")
    assert not loading_module._selected_source_changed(regular, regular, destination)
    changed = list(regular)
    changed[1] += 1
    assert loading_module._selected_source_changed(
        regular, os.stat_result(changed), destination
    )
    changed = list(regular)
    changed[6] += 1
    assert loading_module._selected_source_changed(
        regular, os.stat_result(changed), destination
    )
    changed = list(regular)
    changed[8] += 1
    assert loading_module._selected_source_changed(
        regular, os.stat_result(changed), destination
    )
    destination.write_text("different", encoding="utf-8")
    assert loading_module._selected_source_changed(regular, regular, destination)


def test_selected_file_copy_contains_race_and_copy_errors(tmp_path, monkeypatch):
    staged = tmp_path / "staged"
    staged.mkdir()
    source = tmp_path / "source.txt"
    source.write_text("event", encoding="utf-8")

    monkeypatch.setattr(loading_module, "_selected_source_changed", lambda *_args: True)
    with pytest.raises(PluginWorkerError) as changed:
        loading_module._copy_selected_source(source, staged, 0, set())
    assert (changed.value.code, changed.value.detail) == (
        "selected-file-changed",
        "selected source changed while it was copied",
    )

    monkeypatch.setattr(loading_module, "_selected_source_changed", lambda *_args: False)
    monkeypatch.setattr(
        loading_module.shutil,
        "copyfileobj",
        lambda *_args, **_options: (_ for _ in ()).throw(OSError("disk")),
    )
    with pytest.raises(PluginWorkerError) as unavailable:
        loading_module._copy_selected_source(source, staged, 1, set())
    assert (unavailable.value.code, unavailable.value.detail) == (
        "selected-file-unavailable",
        "selected source cannot be copied",
    )


def test_worker_spec_inputs_are_bounded(tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    plugin = _resolved(tmp_path=tmp_path)

    with pytest.raises(ValueError, match="python_executable"):
        external_worker_specs((plugin,), python_executable="")
    with pytest.raises(ValueError, match="python_executable"):
        external_worker_specs((plugin,), python_executable="bad\0path")
    with pytest.raises(ValueError, match="absolute directories"):
        external_worker_specs((plugin,), worker_import_paths=(Path("relative"),))
    with pytest.raises(ValueError, match="absolute directories"):
        external_worker_specs((plugin,), worker_import_paths=(tmp_path / "missing",))
    with pytest.raises(ValueError, match="unique"):
        external_worker_specs((plugin,), worker_import_paths=(site, site))
    with pytest.raises(ValueError, match=f"at most {MAX_WORKER_IMPORT_PATHS}"):
        external_worker_specs(
            (plugin,),
            worker_import_paths=(site,) * (MAX_WORKER_IMPORT_PATHS + 1),
        )


def test_worker_loads_only_the_exact_validated_entry_point():
    entry_point = FakeEntryPoint()
    plugin = load_external_plugin(
        "external-example",
        "external-example",
        "external_package:Plugin",
        "external-dist",
        entry_points_provider=lambda **selection: [entry_point]
        if selection == {"group": "omnitensor.workloads"}
        else [],
    )

    assert isinstance(plugin, TrackingPlugin)
    assert entry_point.load_calls == 1


@pytest.mark.parametrize(
    "entry_point",
    [
        FakeEntryPoint(name="substitute"),
        FakeEntryPoint(value="other:Plugin"),
        FakeEntryPoint(distribution="other-dist"),
        FakeEntryPoint(distribution=None),
    ],
)
def test_worker_rejects_changed_installed_identity(entry_point):
    with pytest.raises(ExternalPluginLoadError) as error:
        load_external_plugin(
            "external-example",
            "external-example",
            "external_package:Plugin",
            "external-dist",
            entry_points_provider=lambda **_selection: [entry_point],
        )
    assert str(error.value) == "installed entry-point identity changed"
    assert entry_point.load_calls == 0


def test_worker_contains_enumeration_load_and_contract_failures():
    def fail_enumeration(**_selection):
        raise OSError("private")

    def fail_factory():
        raise RuntimeError("private")

    with pytest.raises(ExternalPluginLoadError, match="enumeration failed: OSError"):
        load_external_plugin("p", "p", "p:P", "p", entry_points_provider=fail_enumeration)
    with pytest.raises(ExternalPluginLoadError, match="load failed: RuntimeError"):
        load_external_plugin(
            "external-example",
            "external-example",
            "external_package:Plugin",
            "external-dist",
            entry_points_provider=lambda **_selection: [
                FakeEntryPoint(factory=fail_factory)
            ],
        )
    with pytest.raises(ExternalPluginLoadError) as error:
        load_external_plugin(
            "external-example",
            "external-example",
            "external_package:Plugin",
            "external-dist",
            entry_points_provider=lambda **_selection: [FakeEntryPoint(factory=dict)],
        )
    assert str(error.value) == "entry point does not implement the declared plugin"
    duplicate = FakeEntryPoint()
    with pytest.raises(ExternalPluginLoadError, match="identity changed"):
        load_external_plugin(
            "external-example",
            "external-example",
            "external_package:Plugin",
            "external-dist",
            entry_points_provider=lambda **_selection: [duplicate, duplicate],
        )


def test_worker_acknowledges_the_handshake_before_starting_the_plugin():
    service_offer = HandshakeOffer(
        "external-example", 1, 3, frozenset({"cancel", "progress"})
    )
    health = encode_frame(
        handshake_frame(service_offer).__class__(
            1, WorkerMessageType.HEALTH, "request-1", {}
        )
    )
    cancel = encode_frame(
        handshake_frame(service_offer).__class__(
            1, WorkerMessageType.CANCEL, None, {"reason": "shutdown"}
        )
    )
    reader = io.BytesIO(encode_frame(handshake_frame(service_offer)) + health + cancel)
    writer = io.BytesIO()
    plugin = TrackingPlugin()

    agreement = serve_worker(
        plugin,
        reader,
        writer,
        minimum_protocol=2,
        maximum_protocol=4,
    )

    assert agreement.protocol_version == 3
    assert agreement.capabilities == frozenset({"cancel", "progress"})
    assert [event[0] for event in plugin.events] == ["start", "stop"]
    assert plugin.events[0][1].protocol_version == 3
    assert plugin.events[0][1].plugin_id == "external-example"
    assert plugin.events[0][1].configuration == {}
    assert plugin.events[0][1].permissions == frozenset()
    output = _frames(writer.getvalue())
    assert [frame.type for frame in output] == [
        WorkerMessageType.HELLO,
        WorkerMessageType.READY,
        WorkerMessageType.ERROR,
    ]
    assert output[1].payload == {"pluginId": "external-example"}
    assert output[2].request_id == "request-1"
    assert output[2].payload == {
        "code": "unsupported-message",
        "detail": "message is not implemented",
    }


def test_worker_receives_only_the_service_supplied_active_permissions():
    service_offer = HandshakeOffer("external-example", 1, 1, frozenset())
    reader = io.BytesIO(encode_frame(handshake_frame(service_offer)))
    plugin = TrackingPlugin()

    serve_worker(
        plugin,
        reader,
        io.BytesIO(),
        permissions=frozenset({"read:/allowed"}),
    )

    assert plugin.events[0][1].permissions == frozenset({"read:/allowed"})


def test_worker_default_protocol_and_eof_shutdown():
    service_offer = HandshakeOffer(
        "external-example", 1, 1, frozenset({"cancel", "health"})
    )
    plugin = TrackingPlugin()
    writer = io.BytesIO()

    agreement = serve_worker(
        plugin,
        io.BytesIO(encode_frame(handshake_frame(service_offer))),
        writer,
    )

    assert agreement.protocol_version == 1
    assert [event[0] for event in plugin.events] == ["start", "stop"]
    assert [frame.type for frame in _frames(writer.getvalue())] == [
        WorkerMessageType.HELLO,
        WorkerMessageType.READY,
    ]


def test_executable_worker_streams_progress_and_one_terminal_result():
    class ExecutablePlugin(TrackingPlugin):
        async def execute(self, request, cancellation, progress):
            cancellation.raise_if_cancelled()
            await progress.report(PluginProgress(request.job_id, "extract", 0.5, "", 12))
            return succeeded_result(request, {"events": []}, completed_at_ms=15)

    service_offer = HandshakeOffer(
        "external-example",
        1,
        1,
        frozenset({"cancel", "execute", "health", "progress"}),
    )
    request = PluginRequest(
        "job-1",
        "external-example",
        "manual",
        {"sources": ["/private/event.txt"]},
        10,
        None,
    )
    reader = io.BytesIO(
        encode_frame(handshake_frame(service_offer)) + encode_frame(execute_frame(request))
    )
    writer = io.BytesIO()
    plugin = ExecutablePlugin()

    agreement = asyncio.run(serve_worker_requests(plugin, reader, writer))

    assert agreement.capabilities == service_offer.capabilities
    frames = _frames(writer.getvalue())
    assert [frame.type for frame in frames] == [
        WorkerMessageType.HELLO,
        WorkerMessageType.READY,
        WorkerMessageType.PROGRESS,
        WorkerMessageType.RESULT,
    ]
    assert frames[2].payload == {
        "detail": "",
        "fraction": 0.5,
        "observedAt": 12,
        "stage": "extract",
    }
    assert frames[3].payload["status"] == "succeeded"
    assert frames[3].payload["output"] == {"events": []}
    assert [event[0] for event in plugin.events] == ["start", "stop"]
    context = plugin.events[0][1]
    assert context == PluginContext(
        "external-example",
        1,
        {},
        frozenset(),
    )
    assert frames[1].payload == {"pluginId": "external-example"}


def test_executable_worker_receives_cancel_while_request_is_running():
    class WaitingPlugin(TrackingPlugin):
        async def execute(self, request, cancellation, progress):
            await cancellation.wait()
            return cancelled_result(request, "cancelled", completed_at_ms=20)

    service_offer = HandshakeOffer(
        "external-example", 1, 1, frozenset({"cancel", "execute"})
    )
    request = PluginRequest("job-1", "external-example", "manual", {}, 10, None)
    cancel = encode_frame(
        handshake_frame(service_offer).__class__(
            1, WorkerMessageType.CANCEL, "job-1", {"reason": "user"}
        )
    )
    reader = io.BytesIO(
        encode_frame(handshake_frame(service_offer))
        + encode_frame(execute_frame(request))
        + cancel
    )
    writer = io.BytesIO()

    asyncio.run(serve_worker_requests(WaitingPlugin(), reader, writer))

    terminal = _frames(writer.getvalue())[-1]
    assert terminal.type is WorkerMessageType.RESULT
    assert terminal.request_id == "job-1"
    assert terminal.payload["status"] == "cancelled"


class _ChunkedReader:
    def __init__(self, chunks):
        self._chunks = iter(chunks)
        self.requests = []

    def read(self, size):
        self.requests.append(size)
        return next(self._chunks, b"")


def test_worker_blocking_frame_reader_bounds_and_reports_truncation():
    reader = _ChunkedReader((b"a", b"bc", b"d"))
    assert _read_exact(reader, 4) == b"abcd"
    assert reader.requests == [4, 3, 1]
    with pytest.raises(EOFError):
        _read_exact(io.BytesIO(), 1)

    truncated = _ChunkedReader((b"a", b""))
    with pytest.raises(IPCProtocolError) as error:
        _read_exact(truncated, 2)
    assert error.value.code == "truncated-frame"
    assert error.value.detail == "expected 2 bytes; received 1"

    oversized = (DEFAULT_MAX_FRAME_BYTES + 1).to_bytes(4, "big")
    with pytest.raises(IPCProtocolError) as error:
        _read_frame(io.BytesIO(oversized))
    assert error.value.code == "frame-too-large"
    assert error.value.detail == (
        f"declared {DEFAULT_MAX_FRAME_BYTES + 1} bytes; "
        f"limit is {DEFAULT_MAX_FRAME_BYTES}"
    )


def test_worker_main_parses_identity_and_import_roots(monkeypatch, tmp_path):
    site = tmp_path / "site"
    site.mkdir()
    plugin = TrackingPlugin()
    loaded = []
    served = []

    def load(*identity):
        loaded.append(identity)
        return plugin

    async def serve(candidate, reader, writer, **options):
        served.append((candidate, reader, writer, options))

    stdin = type("Input", (), {"buffer": io.BytesIO()})()
    channel = io.BytesIO()
    claims = []
    monkeypatch.setattr(worker_module, "load_external_plugin", load)
    monkeypatch.setattr(worker_module, "serve_worker_requests", serve)
    monkeypatch.setattr(
        worker_module,
        "claim_frame_channel",
        lambda: claims.append(len(loaded)) or channel,
    )
    monkeypatch.setattr(worker_module.sys, "stdin", stdin)
    monkeypatch.setattr(worker_module.sys, "path", list(sys.path))

    # --no-seccomp: installing the real filter here would confine the test
    # process itself, and a seccomp filter cannot be removed once applied.
    worker_module.main(
        [
            "--no-seccomp",
            "--plugin-id",
            "external-example",
            "--entry-point",
            "external-example",
            "--target",
            "external_package:Plugin",
            "--distribution",
            "external-dist",
            "--import-path",
            str(site),
        ]
    )

    assert loaded == [
        (
            "external-example",
            "external-example",
            "external_package:Plugin",
            "external-dist",
        )
    ]
    assert served == [(plugin, stdin.buffer, channel, {"permissions": frozenset()})]
    assert claims == [0], "the frame channel must be claimed before the plugin is loaded"
    assert worker_module.sys.path[0] == str(site)


def _write_external_wheel(path: Path) -> None:
    manifest = sample_plugin_manifest("third-party-plugin")
    manifest["version"] = "1.0.0"
    manifest["plugin"]["protocol"]["capabilities"] = [
        "cancel",
        "execute",
        "health",
        "progress",
    ]
    manifest["plugin"]["schemas"]["output"] = {
        "type": "object",
        "additionalProperties": False,
        "required": ["ok"],
        "properties": {"ok": {"const": True}},
    }
    package = """\
from omnitensor.sdk import ManagedPlugin, succeeded_result

class ThirdPartyPlugin(ManagedPlugin):
    plugin_id = "third-party-plugin"

    async def execute(self, request, cancellation, progress):
        return succeeded_result(
            request, {"ok": True}, completed_at_ms=1, detail="fixture"
        )
"""
    dist_info = "third_party_omnitensor-1.0.0.dist-info"
    files = {
        "third_party_plugin/__init__.py": package,
        f"{dist_info}/METADATA": (
            "Metadata-Version: 2.1\nName: third-party-omnitensor\nVersion: 1.0.0\n"
        ),
        f"{dist_info}/WHEEL": (
            "Wheel-Version: 1.0\nGenerator: omnitensor-test\n"
            "Root-Is-Purelib: true\nTag: py3-none-any\n"
        ),
        f"{dist_info}/entry_points.txt": (
            "[omnitensor.workloads]\n"
            "third-party-plugin = third_party_plugin:ThirdPartyPlugin\n"
        ),
        "omnitensor-plugin.json": json.dumps(manifest),
    }
    record = "".join(f"{name},,\n" for name in (*files, f"{dist_info}/RECORD"))
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
        for name, contents in files.items():
            wheel.writestr(name, contents)
        wheel.writestr(f"{dist_info}/RECORD", record)


class _Discovery:
    def detect(self):
        return [Device("tpu-pcie-0", "tpu", "Test TPU", "pcie")]

    def utilization(self, _device):
        return None


class _Publisher:
    def publish(self, _snapshot):
        return None

    def retract(self):
        return None


class _Transport:
    async def start(self, _handler):
        return None

    async def stop(self):
        return None


def test_installed_wheel_is_discovered_and_loaded_after_service_restart(tmp_path):
    wheel = tmp_path / "third_party_omnitensor-1.0.0-py3-none-any.whl"
    site = tmp_path / "site"
    workloads = tmp_path / "workloads"
    site.mkdir()
    workloads.mkdir()
    _write_external_wheel(wheel)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-compile",
            "--no-deps",
            "--no-index",
            "--target",
            str(site),
            str(wheel),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    provider = entry_points_from_distributions((site,))

    async def run_once(index):
        runtime = InstalledPluginRuntime(
            workloads,
            entry_points_provider=provider,
            worker_import_paths=(site,),
            python_executable=sys.executable,
        )
        assert runtime.snapshot.catalog.plugins == ()
        assert runtime.snapshot.workers == ()
        service = OmniTensorService(
            tmp_path / f"snapshot-{index}.json",
            tmp_path / f"policy-{index}.json",
            workloads,
            discovery=_Discovery(),
            publisher=_Publisher(),
            transport=_Transport(),
            plugin_runtime=runtime,
            publish_interval_s=0.01,
            discovery_interval_s=60,
        )
        runner = asyncio.create_task(service.run())
        try:
            for _ in range(200):
                if any(
                    worker.state is WorkerState.READY
                    for worker in runtime.snapshot.workers
                ):
                    break
                await asyncio.sleep(0.005)
            started = runtime.snapshot
            assert started.workers
            accepted = json.loads(
                await service.jobs.submit_job_text(
                    json.dumps(
                        {
                            "version": 1,
                            "requestId": f"execute-{index}",
                            "workloadId": "third-party-plugin",
                            "payload": {},
                        }
                    )
                )
            )
            assert accepted["status"] == "accepted"
            for attempt in range(200):
                result = json.loads(
                    await service.jobs.job_result_text(
                        json.dumps(
                            {
                                "version": 1,
                                "requestId": f"result-{index}-{attempt}",
                                "jobId": accepted["jobId"],
                            }
                        )
                    )
                )
                if result["state"] != "running":
                    break
                await asyncio.sleep(0.005)
            assert result["state"] == "succeeded", json.dumps(result, sort_keys=True)
            assert result["output"] == {"ok": True}
        finally:
            service._stopping.set()
            await asyncio.wait_for(runner, timeout=2)
        return started, runtime.snapshot

    async def scenario():
        return await run_once(1), await run_once(2)

    first, restarted = asyncio.run(scenario())
    for started, stopped in (first, restarted):
        external = [
            plugin
            for plugin in started.catalog.plugins
            if plugin.source is PluginSource.EXTERNAL
        ]
        assert [plugin.plugin_id for plugin in external] == [
            "third-party-plugin"
        ], started.catalog.rejections
        assert started.catalog.rejections == ()
        assert len(started.workers) == 1
        assert started.workers[0].state is WorkerState.READY
        assert stopped.workers[0].state is WorkerState.STOPPED


def test_worker_rejects_a_plugin_that_omits_its_identity():
    class Anonymous:
        async def start(self, context): ...

        async def health(self): ...

        async def stop(self): ...

        async def execute(self, request, cancellation, progress): ...

    assert isinstance(Anonymous(), WorkloadPlugin)
    with pytest.raises(ExternalPluginLoadError) as error:
        load_external_plugin(
            "external-example",
            "external-example",
            "external_package:Plugin",
            "external-dist",
            entry_points_provider=lambda **_selection: [FakeEntryPoint(factory=Anonymous)],
        )
    assert str(error.value) == "entry point does not implement the declared plugin"


_CHATTY_PLUGIN = '''\
import sys

print("noise from import", flush=True)
sys.stdout.write("more noise\\n")

from omnitensor.plugins.protocol import PluginHealth, PluginHealthStatus


class Plugin:
    plugin_id = "chatty-plugin"

    async def start(self, context):
        print("noise from start", flush=True)

    async def health(self):
        return PluginHealth(PluginHealthStatus.READY, "ready", 0)

    async def stop(self):
        print("noise from stop", flush=True)

    async def execute(self, request, cancellation, progress):
        raise AssertionError("worker bootstrap must not execute jobs")
'''

_CHATTY_BOOTSTRAP = '''\
from omnitensor.plugins import worker


def _load(plugin_id, entry_point, target, distribution):
    import chatty_plugin

    return chatty_plugin.Plugin()


worker.load_external_plugin = _load
worker.main(
    [
        "--plugin-id",
        "chatty-plugin",
        "--entry-point",
        "chatty-plugin",
        "--target",
        "chatty_plugin:Plugin",
        "--distribution",
        "chatty-dist",
    ]
)
'''


def test_plugin_stdout_never_reaches_the_frame_channel(tmp_path):
    """A single stray print used to desynchronise the service's frame reader."""
    (tmp_path / "chatty_plugin.py").write_text(_CHATTY_PLUGIN, encoding="utf-8")
    (tmp_path / "bootstrap.py").write_text(_CHATTY_BOOTSTRAP, encoding="utf-8")
    offer = HandshakeOffer("chatty-plugin", 1, 1, frozenset({"cancel"}))

    worker = subprocess.Popen(
        [sys.executable, str(tmp_path / "bootstrap.py")],
        cwd=tmp_path,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
    )
    try:
        worker.stdin.write(encode_frame(handshake_frame(offer)))
        worker.stdin.flush()
        agreement = _read_frame(worker.stdout)
        ready = _read_frame(worker.stdout)
        worker.stdin.close()
        stdout_tail = worker.stdout.read()
        stderr = worker.stderr.read().decode()
    finally:
        worker.kill()
        worker.wait(timeout=30)

    assert agreement.payload["pluginId"] == "chatty-plugin"
    assert ready.type is WorkerMessageType.READY
    assert stdout_tail == b"", "plugin output leaked into the frame channel"
    assert "noise from import" in stderr
    assert "noise from start" in stderr


def test_worker_answers_the_handshake_before_its_plugin_starts():
    """Startup time used to be charged to the service's handshake deadline."""
    service_offer = HandshakeOffer("external-example", 1, 1, frozenset({"cancel"}))
    writer = io.BytesIO()

    class ObservantPlugin(TrackingPlugin):
        frames_at_start = None

        async def start(self, context):
            type(self).frames_at_start = _frames(writer.getvalue())
            await super().start(context)

    serve_worker(
        ObservantPlugin(),
        io.BytesIO(encode_frame(handshake_frame(service_offer))),
        writer,
    )

    assert [frame.type for frame in ObservantPlugin.frames_at_start] == [
        WorkerMessageType.HELLO
    ]
    assert [frame.type for frame in _frames(writer.getvalue())] == [
        WorkerMessageType.HELLO,
        WorkerMessageType.READY,
    ]


def test_the_worker_confines_itself_before_it_imports_the_plugin(monkeypatch, tmp_path):
    """Importing a plugin already runs its code; a later filter is too late."""
    order = []

    def install():
        order.append("install")
        return 15

    def load(*identity):
        order.append("load")
        return TrackingPlugin()

    monkeypatch.setattr(worker_module, "install_filter", install)
    monkeypatch.setattr(worker_module, "load_external_plugin", load)
    async def serve(*_args, **_options):
        order.append("serve")

    monkeypatch.setattr(worker_module, "serve_worker_requests", serve)
    monkeypatch.setattr(
        worker_module, "claim_frame_channel", lambda: io.BytesIO()
    )
    monkeypatch.setattr(
        worker_module.sys, "stdin", type("Input", (), {"buffer": io.BytesIO()})()
    )

    worker_module.main(
        [
            "--plugin-id",
            "external-example",
            "--entry-point",
            "external-example",
            "--target",
            "external_package:Plugin",
            "--distribution",
            "external-dist",
        ]
    )

    assert order == ["install", "load", "serve"]


def test_a_worker_told_not_to_confine_itself_does_not(monkeypatch):
    """Only for kernels that cannot install one; it is never the default."""
    installed = []
    monkeypatch.setattr(worker_module, "install_filter", lambda: installed.append(1))
    monkeypatch.setattr(worker_module, "load_external_plugin", lambda *a: TrackingPlugin())
    async def serve(*_args, **_options):
        return None

    monkeypatch.setattr(worker_module, "serve_worker_requests", serve)
    monkeypatch.setattr(worker_module, "claim_frame_channel", lambda: io.BytesIO())
    monkeypatch.setattr(
        worker_module.sys, "stdin", type("Input", (), {"buffer": io.BytesIO()})()
    )

    worker_module.main(
        [
            "--no-seccomp",
            "--plugin-id",
            "external-example",
            "--entry-point",
            "external-example",
            "--target",
            "external_package:Plugin",
            "--distribution",
            "external-dist",
        ]
    )

    assert installed == []
