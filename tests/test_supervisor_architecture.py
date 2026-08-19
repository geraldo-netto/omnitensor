"""Stable facade and acyclic owner boundaries for worker supervision."""

from __future__ import annotations

import ast
import asyncio
import functools
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
    "IDLE_WORKER_DETAIL",
    "IPCProtocolError",
    "MAX_SUPERVISED_WORKERS",
    "MAX_WORKER_ARGUMENTS",
    "MAX_WORKER_ARGUMENT_CHARS",
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
        # Each says where it is defined, not where it is re-exported: the
        # supervisor is a coordinator, and a traceback that names it for a
        # class defined in a leaf sends the reader to the wrong file.
        assert owner.__module__.startswith("omnitensor.plugins.supervisor_")

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
            and node.module
            in {
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


def test_recovery_catches_the_real_start_error_whatever_the_facade_is_rebound_to():
    """OMNI-0452: the caught type is the imported class, not a registry lookup.

    `recover_worker` used to resolve the exception type out of
    `sys.modules['omnitensor.plugins.supervisor']`, so anything rebinding
    `supervisor._WorkerStartError` made it stop catching launch failures and
    the recovery task died silently.
    """

    class DecoyError(Exception):
        pass

    original = supervisor._WorkerStartError
    supervisor._WorkerStartError = DecoyError
    try:
        host, observed = asyncio.run(_exhaust_one_restart())
    finally:
        supervisor._WorkerStartError = original

    assert host._statuses["worker"].state is diagnostics.WorkerState.EXHAUSTED
    assert [item.code for item in observed[1:]] == [
        diagnostics.WorkerDiagnosticCode.RESTART_FAILED,
        diagnostics.WorkerDiagnosticCode.EXHAUSTED,
    ]


async def _exhaust_one_restart():
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
            raise session.WorkerStartError("restart failed", 6)

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


def test_a_recovery_task_that_dies_leaves_a_diagnostic_and_no_stale_entry():
    """OMNI-0407: nothing awaits the task `monitor_worker` creates."""
    observed = []

    class Host:
        def __init__(self):
            self._restart_attempts = {"worker": 2}
            self._record_diagnostic = observed.append

    async def scenario():
        host = Host()

        async def explode():
            raise RuntimeError("boom")

        task = asyncio.ensure_future(explode())
        host._recoveries = {"worker": task}
        task.add_done_callback(functools.partial(recovery._recovery_finished, host, "worker"))
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)
        return host

    host = asyncio.run(scenario())

    assert host._recoveries == {}
    [diagnostic] = observed
    assert diagnostic.code is diagnostics.WorkerDiagnosticCode.RESTART_FAILED
    assert diagnostic.detail == "recovery task failed: RuntimeError"
    assert diagnostic.restart_attempt == 2


def test_recovery_survives_a_revoke_that_forgot_the_plugin_mid_flight():
    """OMNI-0407: `revoke` pops both maps a recovery indexed with bare `[]`."""

    class Host:
        _lock = asyncio.Lock()
        _running = False
        _recovery_policy = recovery.WorkerRecoveryPolicy(max_restarts=1)
        _recoveries: dict = {}
        _restart_attempts: dict = {}
        _statuses: dict = {}

        async def _cancel_orphaned_jobs(self, plugin_id, detail):
            return None

        def _record_diagnostic(self, diagnostic):
            return None

    asyncio.run(
        recovery.recover_worker(
            Host(),
            process.WorkerSpec("worker", ("python",)),
            "exited",
            force_stop=lambda *_args: None,
        )
    )


def test_a_cancelled_launch_still_kills_the_child_it_started():
    """OMNI-0406: an unshielded cleanup dies at its own first await.

    force_stop closes the writer and waits for the process. Both are awaits,
    and in an already-cancelled task every await raises CancelledError again,
    so the child was never terminated and outlived the service holding its
    GPU. The cleanup must be shielded to run to completion.
    """

    async def scenario():
        stopped = []
        started = asyncio.Event()

        class Process:
            pid = 41
            reader = object()
            writer = object()
            returncode = None

        process = Process()

        class Launcher:
            async def launch(self, _spec):
                return process

        async def never_handshakes(*_args, **_options):
            started.set()
            await asyncio.sleep(3600)

        async def force_stop(target, _timeout):
            # Two suspension points, exactly like the real cleanup chain.
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            stopped.append(target)
            return True

        task = asyncio.ensure_future(
            session.launch_authenticated(
                Launcher(),
                supervisor.WorkerSpec("worker", ("python",)),
                handshake_timeout=5.0,
                startup_timeout=5.0,
                stop_timeout=0.01,
                handshake=never_handshakes,
                await_ready=never_handshakes,
                force_stop=force_stop,
            )
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert stopped == [process], "the cancelled launch left its child running"

    asyncio.run(scenario())


def test_a_cancelled_job_cancellation_still_kills_an_unresponsive_worker():
    """OMNI-0406: the same shield is needed on the cancel_request path."""

    async def scenario():
        stopped = []
        entered = asyncio.Event()

        class Process:
            pid = 42
            writer = object()
            returncode = None

        slot = SimpleNamespace(
            process=Process(),
            agreement=SimpleNamespace(protocol_version=1),
        )

        async def write(*_args, **_options):
            return None

        async def read_result_of(*_args, **_options):
            entered.set()
            await asyncio.sleep(3600)

        async def force_stop(target, _timeout):
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            stopped.append(target)
            return True

        task = asyncio.ensure_future(
            session.cancel_request(
                slot,
                "job-1",
                cancel_timeout=5.0,
                stop_timeout=0.01,
                read_result_of=read_result_of,
                write=write,
                force_stop=force_stop,
            )
        )
        await entered.wait()
        task.cancel()
        # cancel_request absorbs the cancellation by design: its caller in
        # PluginWorkerSupervisor shields it and re-raises.
        await asyncio.gather(task, return_exceptions=True)

        assert stopped == [slot.process], "the unresponsive worker was never stopped"

    asyncio.run(scenario())


def test_a_dead_worker_carries_its_stderr_into_the_failure_detail():
    """OMNI-0381: the traceback that names the cause reached only a closed pipe.

    A worker that dies on import writes a complete traceback to stderr and then
    fails the handshake. Reporting only `worker handshake rejected:
    truncated-frame` hides the ImportError behind the transport symptom.
    """

    async def scenario():
        traceback_text = (
            "Traceback (most recent call last):\nImportError: cannot import name 'MAX_VISUALS'\n"
        )
        stderr = asyncio.StreamReader()
        stderr.feed_data(traceback_text.encode())
        stderr.feed_eof()

        class Process:
            pid = 41
            reader = object()
            writer = object()
            diagnostics = stderr
            returncode = None

        class Launcher:
            async def launch(self, _spec):
                return Process()

        async def rejects(*_args, **_options):
            raise supervisor.IPCProtocolError("truncated-frame", "short read")

        async def force_stop(_target, _timeout):
            return True

        with pytest.raises(session.WorkerStartError) as error:
            await session.launch_authenticated(
                Launcher(),
                supervisor.WorkerSpec("worker", ("python",)),
                handshake_timeout=5.0,
                startup_timeout=5.0,
                stop_timeout=0.01,
                handshake=rejects,
                await_ready=rejects,
                force_stop=force_stop,
            )

        assert "worker handshake rejected: truncated-frame" in error.value.detail
        assert "ImportError: cannot import name 'MAX_VISUALS'" in error.value.detail

    asyncio.run(scenario())


def test_draining_worker_stderr_is_bounded_in_bytes_and_in_time():
    """A chatty or silent worker must not blow up or hold up the report."""

    async def scenario():
        loud = asyncio.StreamReader()
        loud.feed_data(b"a" * 40 + b"b" * 40)
        loud.feed_eof()
        tail = await process.drain_worker_diagnostics(SimpleNamespace(diagnostics=loud), limit=16)
        assert tail == "b" * 16

        never_closes = asyncio.StreamReader()
        never_closes.feed_data(b"partial")
        assert (
            await process.drain_worker_diagnostics(
                SimpleNamespace(diagnostics=never_closes), timeout=0.01
            )
            == "partial"
        )

        assert await process.drain_worker_diagnostics(SimpleNamespace()) == ""

    asyncio.run(scenario())


def test_the_shutdown_cancel_carries_the_negotiated_frame_version():
    """Every other service-to-worker write stamps the version the handshake
    agreed; this one stamped the module constant, so a worker speaking a
    negotiated version 2 was told to shut down in a dialect it may refuse."""

    async def scenario():
        written = []

        class Process:
            pid = 42
            writer = object()
            returncode = None

        slot = SimpleNamespace(
            process=Process(),
            agreement=SimpleNamespace(protocol_version=2),
        )

        async def write(_writer, frame):
            written.append(frame)

        async def close_writer(_writer):
            return None

        async def terminate(_process, _timeout):
            return True

        await session.stop_process(
            slot,
            timeout=0.01,
            write=write,
            close_writer=close_writer,
            terminate=terminate,
        )
        return written

    written = asyncio.run(scenario())

    assert [frame.version for frame in written] == [2]


def test_a_worker_frame_in_another_version_is_refused_not_parsed():
    """`parse_execute`/`parse_progress`/`parse_result` never looked at
    `frame.version`, so a frame in a version the handshake did not agree was
    parsed as though its field meanings were settled."""
    from omnitensor.plugins import IPCFrame, IPCProtocolError, WorkerMessageType

    async def scenario():
        class Process:
            pid = 42
            reader = object()
            writer = object()
            returncode = None

        slot = SimpleNamespace(
            process=Process(),
            agreement=SimpleNamespace(protocol_version=1),
        )

        async def read(_reader):
            return IPCFrame(
                2,
                WorkerMessageType.RESULT,
                "job-1",
                {"status": "succeeded", "output": {}, "detail": "", "completedAt": 1},
            )

        with pytest.raises(session.PluginWorkerError) as failure:
            await session.read_result(slot, "job-1", None, read=read)
        return failure.value

    failure = asyncio.run(scenario())

    assert failure.code == "worker-protocol-failed"
    assert "version" in failure.detail

    # And a direct caller of the parser gets the same refusal.
    frame = IPCFrame(
        2,
        WorkerMessageType.RESULT,
        "job-1",
        {"status": "succeeded", "output": {}, "detail": "", "completedAt": 1},
    )
    with pytest.raises(IPCProtocolError) as refused:
        plugins.parse_result(frame, protocol_version=1)
    assert refused.value.code == "frame-version-incompatible"
