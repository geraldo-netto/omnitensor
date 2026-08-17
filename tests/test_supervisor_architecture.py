"""Stable facade and acyclic owner boundaries for worker supervision."""

from __future__ import annotations

import ast
import asyncio
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from omnitensor import plugins
from omnitensor.plugins import loading, loading_staging, supervisor, worker_specs
from omnitensor.plugins import supervisor_diagnostics as diagnostics
from omnitensor.plugins import supervisor_process as process
from omnitensor.plugins import supervisor_recovery as recovery
from omnitensor.plugins import supervisor_session as session

PUBLIC_FACADE_NAMES = (
    "AsyncioSubprocessLauncher",
    "DEFAULT_CANCEL_TIMEOUT_SECONDS",
    "DEFAULT_HANDSHAKE_TIMEOUT_SECONDS",
    "DEFAULT_STARTUP_TIMEOUT_SECONDS",
    "DEFAULT_STOP_TIMEOUT_SECONDS",
    "HandshakeAgreement",
    "IPCProtocolError",
    "MAX_SUPERVISED_WORKERS",
    "MAX_WORKER_ARGUMENTS",
    "MAX_WORKER_ARGUMENT_CHARS",
    "MAX_WORKER_DIAGNOSTICS",
    "PluginRequest",
    "PluginResult",
    "PluginWorkerError",
    "PluginWorkerSupervisor",
    "ProgressReporter",
    "Sequence",
    "WorkerDiagnostic",
    "WorkerDiagnosticCode",
    "WorkerFailureObserver",
    "WorkerLauncher",
    "WorkerProcess",
    "WorkerRecoveryPolicy",
    "WorkerSpec",
    "WorkerState",
    "WorkerStatus",
    "annotations",
    "asyncio",
    "await_worker_ready",
    "execute_frame",
    "handshake_frame",
    "parse_progress",
    "parse_result",
    "perform_service_handshake",
    "read_frame",
    "write_frame",
)


def test_supervisor_exposes_only_what_it_uses_and_keeps_owner_identity():
    # The coordinator imports what its own body needs and nothing else. Anything
    # a caller wants comes from the owning module or from the package barrel,
    # which routes each name at its owner rather than through here.
    assert tuple(name for name in dir(supervisor) if not name.startswith("_")) == (
        PUBLIC_FACADE_NAMES
    )
    owner_pairs = {
        "AsyncioSubprocessLauncher": process.AsyncioSubprocessLauncher,
        "PluginWorkerError": session.PluginWorkerError,
        "WorkerDiagnostic": diagnostics.WorkerDiagnostic,
        "WorkerDiagnosticCode": diagnostics.WorkerDiagnosticCode,
        "WorkerFailureObserver": recovery.WorkerFailureObserver,
        "WorkerLauncher": process.WorkerLauncher,
        "WorkerProcess": process.WorkerProcess,
        "WorkerRecoveryPolicy": recovery.WorkerRecoveryPolicy,
        "WorkerSpec": process.WorkerSpec,
        "WorkerState": diagnostics.WorkerState,
        "WorkerStatus": diagnostics.WorkerStatus,
    }
    for name, owner in owner_pairs.items():
        assert getattr(supervisor, name) is owner
        assert getattr(plugins, name) is owner
        assert owner.__module__ == "omnitensor.plugins.supervisor"

    assert loading.PluginWorkerError is session.PluginWorkerError
    assert loading.WorkerState is diagnostics.WorkerState
    assert loading.WorkerStatus is diagnostics.WorkerStatus
    assert loading.WorkerSpec is process.WorkerSpec
    assert loading_staging.PluginWorkerError is session.PluginWorkerError
    assert worker_specs.WorkerSpec is process.WorkerSpec


def test_supervisor_values_preserve_pickle_and_error_failure_contracts():
    values = (
        supervisor.WorkerState.READY,
        supervisor.WorkerDiagnosticCode.EXITED,
        supervisor.WorkerStatus("worker", supervisor.WorkerState.READY, 7, 1, "ready"),
        supervisor.WorkerDiagnostic(
            "worker",
            supervisor.WorkerDiagnosticCode.EXITED,
            "exited",
            7,
            0,
            0,
        ),
        supervisor.WorkerRecoveryPolicy(),
        supervisor.WorkerSpec("worker", ("python", "worker.py")),
    )
    for value in values:
        restored = pickle.loads(pickle.dumps(value))
        assert restored == value
        assert type(restored) is type(value)

    with pytest.raises(TypeError, match="missing 1 required positional argument: 'detail'"):
        pickle.loads(pickle.dumps(supervisor.PluginWorkerError("code", "detail")))


def test_supervisor_leaves_have_no_facade_or_loading_backedge():
    for owner in (diagnostics, process, recovery, session):
        tree = ast.parse(Path(owner.__file__).read_text(encoding="utf-8"))
        assert not any(
            isinstance(node, ast.ImportFrom)
            and node.module in {
                "supervisor",
                "loading",
                "omnitensor.plugins.supervisor",
                "omnitensor.plugins.loading",
            }
            for node in ast.walk(tree)
        )

    script = """
import importlib
import sys
for name in (
    'supervisor_diagnostics',
    'supervisor_process',
    'supervisor_session',
    'supervisor_recovery',
):
    importlib.import_module(f'omnitensor.plugins.{name}')
assert 'omnitensor.plugins.supervisor' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_supervisor_coordinator_resolves_facade_defaults_and_callbacks(monkeypatch):
    launcher = object()
    policy = object()
    monkeypatch.setattr(supervisor, "AsyncioSubprocessLauncher", lambda: launcher)
    monkeypatch.setattr(supervisor, "WorkerRecoveryPolicy", lambda: policy)
    subject = supervisor.PluginWorkerSupervisor()
    assert subject._launcher is launcher
    assert subject._recovery_policy is policy

    validation = []
    offer_validator = object()
    monkeypatch.setattr(supervisor, "MAX_SUPERVISED_WORKERS", 7)
    monkeypatch.setattr(supervisor, "MAX_WORKER_ARGUMENTS", 8)
    monkeypatch.setattr(supervisor, "MAX_WORKER_ARGUMENT_CHARS", 9)
    monkeypatch.setattr(supervisor, "handshake_frame", offer_validator)
    monkeypatch.setattr(
        supervisor,
        "_validate_specs_owner",
        lambda specs, **options: validation.append((specs, options)) or tuple(specs),
    )
    spec = supervisor.WorkerSpec("worker", ("python",))
    assert supervisor._validate_specs([spec]) == (spec,)
    assert validation == [
        (
            [spec],
            {
                "max_workers": 7,
                "max_arguments": 8,
                "max_argument_chars": 9,
                "validate_offer": offer_validator,
            },
        )
    ]


def test_process_and_session_leaves_build_frames_through_their_owners(monkeypatch):
    offer = object()
    offer_calls = []
    monkeypatch.setattr(
        process,
        "HandshakeOffer",
        lambda *values: offer_calls.append(values) or offer,
    )
    spec = process.WorkerSpec(
        "worker",
        ("python",),
        minimum_protocol=1,
        maximum_protocol=2,
        capabilities=frozenset({"execute"}),
    )
    assert spec.offer() is offer
    assert offer_calls == [("worker", 1, 2, frozenset({"execute"}))]

    frame_calls = []
    monkeypatch.setattr(
        session,
        "IPCFrame",
        lambda *values: frame_calls.append(values) or ("frame", values),
    )
    source = SimpleNamespace(type="result", request_id="job", payload={"value": 1})
    assert session.with_protocol(source, 7) == (
        "frame",
        (7, "result", "job", {"value": 1}),
    )
    assert frame_calls == [(7, "result", "job", {"value": 1})]


def test_authenticated_launch_receives_live_facade_session_hooks(monkeypatch):
    events = []
    agreement = object()
    process_value = object()

    async def launch_authenticated(*args, **options):
        events.append((args, options))
        return process_value, agreement

    hooks = {
        "perform_service_handshake": object(),
        "await_worker_ready": object(),
        "_force_stop": object(),
        "_handshake_failure": object(),
        "_startup_failure": object(),
    }
    monkeypatch.setattr(session, "launch_authenticated", launch_authenticated)
    for name, value in hooks.items():
        monkeypatch.setattr(supervisor, name, value)
    launcher = object()
    subject = supervisor.PluginWorkerSupervisor(launcher)
    spec = supervisor.WorkerSpec("worker", ("python",))

    assert asyncio.run(subject._launch_authenticated(spec)) == (process_value, agreement)
    [(args, options)] = events
    assert args == (launcher, spec)
    assert options == {
        "handshake_timeout": supervisor.DEFAULT_HANDSHAKE_TIMEOUT_SECONDS,
        "startup_timeout": supervisor.DEFAULT_STARTUP_TIMEOUT_SECONDS,
        "stop_timeout": supervisor.DEFAULT_STOP_TIMEOUT_SECONDS,
        "handshake": hooks["perform_service_handshake"],
        "await_ready": hooks["await_worker_ready"],
        "force_stop": hooks["_force_stop"],
        "handshake_failure_of": hooks["_handshake_failure"],
        "startup_failure_of": hooks["_startup_failure"],
    }


def test_recovery_catches_the_live_facade_start_error(monkeypatch):
    class LiveStartError(Exception):
        def __init__(self, detail, pid, *, charges_restart=True):
            self.detail = detail
            self.pid = pid
            self.charges_restart = charges_restart
            super().__init__(detail)

    monkeypatch.setattr(supervisor, "_WorkerStartError", LiveStartError)

    async def scenario():
        observed = []

        class Host:
            def __init__(self):
                self._lock = asyncio.Lock()
                self._running = True
                self._recovery_policy = recovery.WorkerRecoveryPolicy(
                    max_restarts=1,
                    initial_backoff_seconds=0.0001,
                    max_backoff_seconds=0.0001,
                )
                self._recoveries = {"worker": asyncio.current_task()}
                self._restart_attempts = {"worker": 0}
                self._statuses = {
                    "worker": diagnostics.WorkerStatus(
                        "worker",
                        diagnostics.WorkerState.FAILED,
                        5,
                        1,
                        "failed",
                    )
                }
                self._stop_timeout = 0.1

            async def _cancel_orphaned_jobs(self, plugin_id, detail):
                observed.append(("orphans", plugin_id, detail))

            async def _launch_authenticated(self, spec):
                raise LiveStartError("restart failed", 6)

            def _record_diagnostic(self, diagnostic):
                observed.append(diagnostic)

        host = Host()
        await recovery.recover_worker(
            host,
            process.WorkerSpec("worker", ("python",)),
            "exited",
            force_stop=lambda *_args: None,
        )
        return host, observed

    host, observed = asyncio.run(scenario())
    assert host._statuses["worker"].state is diagnostics.WorkerState.EXHAUSTED
    assert [item.code for item in observed[1:]] == [
        diagnostics.WorkerDiagnosticCode.RESTART_FAILED,
        diagnostics.WorkerDiagnosticCode.EXHAUSTED,
    ]
