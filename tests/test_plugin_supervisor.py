from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from omnitensor.plugins import (
    MAX_SUPERVISED_WORKERS,
    MAX_WORKER_ARGUMENT_CHARS,
    MAX_WORKER_ARGUMENTS,
    MAX_WORKER_DIAGNOSTICS,
    AsyncioSubprocessLauncher,
    HandshakeOffer,
    IPCFrame,
    PluginProgress,
    PluginRequest,
    PluginResult,
    PluginResultStatus,
    PluginWorkerError,
    PluginWorkerSupervisor,
    WorkerDiagnostic,
    WorkerDiagnosticCode,
    WorkerMessageType,
    WorkerRecoveryPolicy,
    WorkerSpec,
    WorkerState,
    WorkerStatus,
    decode_frame,
    encode_frame,
    handshake_frame,
    progress_frame,
    ready_frame,
    result_frame,
)


def run_scenario(coroutine):
    async def bounded():
        return await asyncio.wait_for(coroutine, timeout=0.25)

    return asyncio.run(bounded())


class FakeWriter:
    def __init__(self, process, response_offer, events, *, drain_error=None, close_error=None):
        self.process = process
        self.response_offer = response_offer
        self.events = events
        self.drain_error = drain_error
        self.close_error = close_error
        self.writes = []
        self.closed = False
        self._responded = False

    def write(self, data):
        self.writes.append(data)
        if self.response_offer is not None and not self._responded:
            self._responded = True
            self.process.reader.feed_data(encode_frame(handshake_frame(self.response_offer)))
            if self.process.reports_ready:
                ready = encode_frame(ready_frame(self.response_offer.plugin_id))
                if self.process.ready_delay:
                    asyncio.get_running_loop().call_later(
                        self.process.ready_delay,
                        self.process.reader.feed_data,
                        ready,
                    )
                else:
                    self.process.reader.feed_data(ready)

    async def drain(self):
        if self.drain_error is not None:
            raise self.drain_error

    def close(self):
        self.closed = True
        self.events.append(f"close:{self.process.plugin_id}")
        if self.process.exit_on_close:
            self.process.exit(0)

    async def wait_closed(self):
        if self.close_error is not None:
            raise self.close_error


class FakeProcess:
    def __init__(
        self,
        plugin_id,
        response_offer,
        events,
        *,
        pid=100,
        exit_on_close=True,
        exit_on_terminate=True,
        drain_error=None,
        close_error=None,
        reports_ready=True,
        ready_delay=0.0,
    ):
        self.plugin_id = plugin_id
        self.pid = pid
        self.reports_ready = reports_ready
        self.ready_delay = ready_delay
        self.reader = asyncio.StreamReader()
        self.returncode = None
        self.exit_on_close = exit_on_close
        self.exit_on_terminate = exit_on_terminate
        self.events = events
        self.writer = FakeWriter(
            self,
            response_offer,
            events,
            drain_error=drain_error,
            close_error=close_error,
        )
        self._exited = asyncio.Event()
        self.terminate_calls = 0
        self.kill_calls = 0

    async def wait(self):
        await self._exited.wait()
        return self.returncode

    def exit(self, returncode):
        if self.returncode is None:
            self.returncode = returncode
            self._exited.set()

    def terminate(self):
        self.events.append(f"terminate:{self.plugin_id}")
        self.terminate_calls += 1
        if self.exit_on_terminate:
            self.exit(-15)

    def kill(self):
        self.events.append(f"kill:{self.plugin_id}")
        self.kill_calls += 1
        self.exit(-9)


class FakeLauncher:
    def __init__(self, processes=None, failures=None):
        self.processes = processes or {}
        self.failures = failures or {}
        self.calls = []

    async def launch(self, spec):
        self.calls.append(spec.plugin_id)
        if spec.plugin_id in self.failures:
            raise self.failures[spec.plugin_id]
        return self.processes[spec.plugin_id]


class SequencedLauncher:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    async def launch(self, spec):
        self.calls.append(spec.plugin_id)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FailureObserver:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    async def cancel_orphaned_jobs(self, plugin_id, detail):
        self.calls.append((plugin_id, detail))
        if self.error is not None:
            raise self.error


def worker_spec(plugin_id):
    return WorkerSpec(
        plugin_id,
        ("python", "worker.py", plugin_id),
        minimum_protocol=1,
        maximum_protocol=2,
        capabilities=frozenset({"cancel", "health"}),
    )


def worker_offer(plugin_id, *, minimum=1, maximum=2):
    return HandshakeOffer(
        plugin_id,
        minimum,
        maximum,
        frozenset({"cancel", "progress"}),
    )


def test_starts_in_identity_order_and_stops_in_reverse_without_leaks():
    async def scenario():
        events = []
        processes = {
            plugin_id: FakeProcess(
                plugin_id,
                worker_offer(plugin_id),
                events,
                pid=index,
            )
            for index, plugin_id in enumerate(("alpha", "beta", "zeta"), start=10)
        }
        launcher = FakeLauncher(processes)
        supervisor = PluginWorkerSupervisor(launcher)

        started = await supervisor.start(
            [worker_spec("zeta"), worker_spec("alpha"), worker_spec("beta")]
        )

        assert supervisor.running
        assert launcher.calls == ["alpha", "beta", "zeta"]
        assert [status.plugin_id for status in started] == ["alpha", "beta", "zeta"]
        assert all(status.state is WorkerState.READY for status in started)
        assert all(status.protocol_version == 2 for status in started)
        assert [status.pid for status in started] == [10, 11, 12]

        stopped = await supervisor.stop()

        assert not supervisor.running
        assert events == ["close:zeta", "close:beta", "close:alpha"]
        assert all(status.state is WorkerState.STOPPED for status in stopped)
        assert all(process.returncode == 0 for process in processes.values())
        for process in processes.values():
            assert process.terminate_calls == process.kill_calls == 0
            assert decode_frame(process.writer.writes[-1]).payload == {"reason": "shutdown"}
        assert await supervisor.stop() == stopped

    run_scenario(scenario())


def test_live_revocation_stops_one_worker_without_recovery_and_cancels_orphans():
    async def scenario():
        events = []
        process = FakeProcess("alpha", worker_offer("alpha"), events, pid=91)
        observer = FailureObserver()
        supervisor = PluginWorkerSupervisor(
            FakeLauncher({"alpha": process}), failure_observer=observer
        )
        await supervisor.start([worker_spec("alpha")])
        monitor = supervisor._slots["alpha"].monitor
        recovery = asyncio.create_task(asyncio.sleep(60))
        supervisor._recoveries["alpha"] = recovery
        supervisor._statuses["alpha"] = WorkerStatus(
            "alpha", WorkerState.READY, 91, 2, "ready", 2
        )

        status = await supervisor.revoke("alpha", "permission changed")

        assert status == WorkerStatus("alpha", WorkerState.STOPPED, 91, 2, "permission changed", 2)
        assert observer.calls == [("alpha", "permission changed")]
        assert process.returncode == 0
        assert recovery.cancelled()
        assert monitor is not None and monitor.done()
        assert "alpha" not in supervisor._slots
        assert "alpha" not in supervisor._recoveries
        assert decode_frame(process.writer.writes[-1]).payload == {"reason": "shutdown"}
        assert await supervisor.revoke("alpha", "again") == status
        assert await supervisor.stop() == (status,)

    run_scenario(scenario())


def test_supervisor_executes_and_forwards_correlated_progress():
    async def scenario():
        events = []
        offer = HandshakeOffer("events", 1, 1, frozenset({"cancel", "execute", "progress"}))
        process = FakeProcess("events", offer, events)
        supervisor = PluginWorkerSupervisor(FakeLauncher({"events": process}))
        spec = WorkerSpec(
            "events",
            ("python", "worker.py", "events"),
            capabilities=frozenset({"cancel", "execute", "progress"}),
        )
        await supervisor.start((spec,))

        class Progress:
            def __init__(self):
                self.items = []

            async def report(self, item):
                self.items.append(item)

        progress = Progress()
        request = PluginRequest("job-1", "events", "manual", {"sources": []}, 10, None)
        pending = asyncio.create_task(supervisor.execute(request, progress))
        await asyncio.sleep(0)
        sent = decode_frame(process.writer.writes[-1])
        assert sent.type is WorkerMessageType.EXECUTE
        assert sent.request_id == "job-1"
        process.reader.feed_data(
            encode_frame(progress_frame(PluginProgress("job-1", "extract", 0.5, "", 11)))
        )
        expected = PluginResult("job-1", PluginResultStatus.SUCCEEDED, {"events": []}, "done", 12)
        process.reader.feed_data(encode_frame(result_frame(expected)))

        assert await pending == expected
        assert progress.items == [PluginProgress("job-1", "extract", 0.5, "", 11)]
        await supervisor.stop()

    run_scenario(scenario())


def test_supervisor_cancels_and_drains_worker_before_releasing_channel():
    async def scenario():
        events = []
        offer = HandshakeOffer("events", 1, 1, frozenset({"cancel", "execute"}))
        process = FakeProcess("events", offer, events)
        supervisor = PluginWorkerSupervisor(FakeLauncher({"events": process}), stop_timeout=0.05)
        spec = WorkerSpec(
            "events",
            ("python", "worker.py", "events"),
            capabilities=frozenset({"cancel", "execute"}),
        )
        await supervisor.start((spec,))
        request = PluginRequest("job-1", "events", "manual", {}, 10, None)
        pending = asyncio.create_task(supervisor.execute(request))
        await asyncio.sleep(0)
        pending.cancel()
        for _ in range(10):
            await asyncio.sleep(0)
            if decode_frame(process.writer.writes[-1]).type is WorkerMessageType.CANCEL:
                break
        assert decode_frame(process.writer.writes[-1]) == IPCFrame(
            1,
            WorkerMessageType.CANCEL,
            "job-1",
            {"reason": "job cancelled"},
        )
        cancelled = PluginResult("job-1", PluginResultStatus.CANCELLED, {}, "cancelled", 12)
        process.reader.feed_data(encode_frame(result_frame(cancelled)))
        with pytest.raises(asyncio.CancelledError):
            await pending
        assert process.returncode is None
        await supervisor.stop()

    run_scenario(scenario())


def test_request_cancel_drain_does_not_share_the_short_process_stop_deadline():
    async def scenario():
        events = []
        offer = HandshakeOffer("events", 1, 1, frozenset({"cancel", "execute"}))
        process = FakeProcess("events", offer, events)
        supervisor = PluginWorkerSupervisor(
            FakeLauncher({"events": process}),
            stop_timeout=0.001,
            cancel_timeout=0.05,
        )
        await supervisor.start(
            (
                WorkerSpec(
                    "events",
                    ("python", "worker.py"),
                    capabilities=frozenset({"cancel", "execute"}),
                ),
            )
        )
        request = PluginRequest("job-1", "events", "manual", {}, 10, None)
        pending = asyncio.create_task(supervisor.execute(request))
        await asyncio.sleep(0)
        pending.cancel()
        for _ in range(10):
            await asyncio.sleep(0)
            if decode_frame(process.writer.writes[-1]).type is WorkerMessageType.CANCEL:
                break
        await asyncio.sleep(0.01)
        cancelled = PluginResult("job-1", PluginResultStatus.CANCELLED, {}, "cancelled", 12)
        process.reader.feed_data(encode_frame(result_frame(cancelled)))

        with pytest.raises(asyncio.CancelledError):
            await pending
        assert process.terminate_calls == process.kill_calls == 0
        await supervisor.stop()

    run_scenario(scenario())


def test_supervisor_refuses_missing_or_incompatible_executable_worker():
    async def scenario():
        supervisor = PluginWorkerSupervisor()
        request = PluginRequest("job-1", "missing", "manual", {}, 1, None)
        with pytest.raises(PluginWorkerError) as missing:
            await supervisor.execute(request)
        assert missing.value.code == "worker-unavailable"
        assert str(missing.value) == "worker-unavailable: plugin worker is not ready"

        events = []
        process = FakeProcess("plain", worker_offer("plain"), events)
        plain = PluginWorkerSupervisor(FakeLauncher({"plain": process}))
        await plain.start((worker_spec("plain"),))
        with pytest.raises(PluginWorkerError) as incompatible:
            await plain.execute(PluginRequest("job", "plain", "manual", {}, 1, None))
        assert incompatible.value.code == "worker-incompatible"
        await plain.stop()

    run_scenario(scenario())


def test_supervisor_rejects_worker_error_and_response_protocol_drift():
    async def scenario():
        events = []
        offer = HandshakeOffer("events", 1, 1, frozenset({"execute", "progress"}))
        process = FakeProcess("events", offer, events)
        supervisor = PluginWorkerSupervisor(FakeLauncher({"events": process}))
        await supervisor.start(
            (
                WorkerSpec(
                    "events",
                    ("python", "worker.py"),
                    capabilities=frozenset({"execute", "progress"}),
                ),
            )
        )

        async def refused(frame):
            pending = asyncio.create_task(
                supervisor.execute(PluginRequest("job-1", "events", "manual", {}, 1, None))
            )
            await asyncio.sleep(0)
            process.reader.feed_data(encode_frame(frame))
            with pytest.raises(PluginWorkerError) as error:
                await pending
            return error.value

        worker_error = await refused(
            IPCFrame(
                1,
                WorkerMessageType.ERROR,
                "job-1",
                {"code": "provider-unavailable", "detail": "bounded"},
            )
        )
        assert (worker_error.code, worker_error.detail) == (
            "provider-unavailable",
            "bounded",
        )
        malformed = await refused(
            IPCFrame(1, WorkerMessageType.ERROR, "job-1", {"code": "", "detail": 1})
        )
        assert (malformed.code, malformed.detail) == (
            "worker-protocol-failed",
            "worker error response is invalid",
        )
        unexpected = await refused(IPCFrame(1, WorkerMessageType.HEALTH, "job-1", {}))
        assert (unexpected.code, unexpected.detail) == (
            "worker-protocol-failed",
            "unexpected worker message: health",
        )
        mismatched = await refused(
            result_frame(PluginResult("other-job", PluginResultStatus.SUCCEEDED, {}, "done", 2))
        )
        assert (mismatched.code, mismatched.detail) == (
            "worker-protocol-failed",
            "worker response names another request",
        )
        await supervisor.stop()

    run_scenario(scenario())


def test_supervisor_maps_broken_channel_and_forces_unresponsive_cancel(tmp_path):
    async def scenario():
        events = []
        offer = HandshakeOffer("events", 1, 1, frozenset({"cancel", "execute"}))
        broken = FakeProcess("events", offer, events)
        supervisor = PluginWorkerSupervisor(FakeLauncher({"events": broken}))
        await supervisor.start(
            (
                WorkerSpec(
                    "events",
                    ("python", "worker.py"),
                    capabilities=frozenset({"cancel", "execute"}),
                ),
            )
        )
        pending = asyncio.create_task(
            supervisor.execute(PluginRequest("job-1", "events", "manual", {}, 1, None))
        )
        await asyncio.sleep(0)
        broken.reader.feed_eof()
        with pytest.raises(PluginWorkerError) as channel:
            await pending
        assert channel.value.code == "worker-protocol-failed"
        await supervisor.stop()

        stuck = FakeProcess(
            "stuck",
            HandshakeOffer("stuck", 1, 1, frozenset({"cancel", "execute"})),
            events,
            exit_on_close=False,
        )
        cancelling = PluginWorkerSupervisor(
            FakeLauncher({"stuck": stuck}),
            stop_timeout=0.01,
            cancel_timeout=0.01,
        )
        await cancelling.start(
            (
                WorkerSpec(
                    "stuck",
                    ("python", "worker.py"),
                    capabilities=frozenset({"cancel", "execute"}),
                ),
            )
        )
        task = asyncio.create_task(
            cancelling.execute(PluginRequest("job-2", "stuck", "manual", {}, 1, None))
        )
        for _ in range(10):
            await asyncio.sleep(0)
            if decode_frame(stuck.writer.writes[-1]).type is WorkerMessageType.EXECUTE:
                break
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stuck.terminate_calls == 1
        await cancelling.stop()

    run_scenario(scenario())


def test_launch_and_handshake_failures_are_isolated_and_children_are_reaped():
    async def scenario():
        events = []
        wrong = FakeProcess("wrong", worker_offer("substitute"), events, pid=21)
        good = FakeProcess("good", worker_offer("good"), events, pid=22)
        launcher = FakeLauncher(
            {"wrong": wrong, "good": good},
            failures={"broken": OSError("private")},
        )
        supervisor = PluginWorkerSupervisor(launcher, stop_timeout=0.01)

        statuses = await supervisor.start(
            [worker_spec("wrong"), worker_spec("good"), worker_spec("broken")]
        )

        by_id = {status.plugin_id: status for status in statuses}
        assert by_id["broken"].detail == "worker launch failed: OSError"
        assert by_id["broken"].pid is None
        assert by_id["wrong"].detail == ("worker handshake rejected: plugin-identity-mismatch")
        assert by_id["wrong"].pid == 21
        assert by_id["wrong"].protocol_version is None
        assert wrong.returncode == 0
        assert by_id["good"].state is WorkerState.READY
        await supervisor.stop()
        assert good.returncode == 0

    run_scenario(scenario())


def test_handshake_timeout_and_transport_failure_are_stable():
    async def scenario():
        events = []
        silent = FakeProcess("silent", None, events)
        broken = FakeProcess(
            "transport",
            worker_offer("transport"),
            events,
            drain_error=ValueError("private"),
        )
        supervisor = PluginWorkerSupervisor(
            FakeLauncher({"silent": silent, "transport": broken}),
            handshake_timeout=0.001,
            stop_timeout=0.001,
        )

        statuses = await supervisor.start([worker_spec("transport"), worker_spec("silent")])

        by_id = {status.plugin_id: status for status in statuses}
        assert by_id["silent"].detail == "worker handshake timed out"
        assert by_id["transport"].detail == "worker handshake failed: ValueError"
        assert silent.returncode == broken.returncode == 0
        await supervisor.stop()

    run_scenario(scenario())


@pytest.mark.parametrize("returncode", [0, 7])
def test_monitor_records_unexpected_worker_exit(returncode):
    async def scenario():
        events = []
        process = FakeProcess("watched", worker_offer("watched"), events)
        supervisor = PluginWorkerSupervisor(
            FakeLauncher({"watched": process}),
            recovery_policy=WorkerRecoveryPolicy(max_restarts=0),
        )
        await supervisor.start([worker_spec("watched")])

        process.exit(returncode)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        status = supervisor.statuses()[0]
        expected_state = WorkerState.EXITED if returncode == 0 else WorkerState.FAILED
        expected_detail = (
            "worker exited" if returncode == 0 else f"worker exited with status {returncode}"
        )
        assert status.state is expected_state
        assert status.detail == expected_detail
        stopped = await supervisor.stop()
        assert stopped == (status,)

    run_scenario(scenario())


def test_stubborn_worker_is_terminated_then_killed():
    async def scenario():
        events = []
        process = FakeProcess(
            "stubborn",
            worker_offer("stubborn"),
            events,
            exit_on_close=False,
            exit_on_terminate=False,
            close_error=ConnectionError("closed"),
        )
        supervisor = PluginWorkerSupervisor(
            FakeLauncher({"stubborn": process}),
            stop_timeout=0.001,
        )
        await supervisor.start([worker_spec("stubborn")])

        stopped = await supervisor.stop()

        assert stopped[0].state is WorkerState.STOPPED
        assert process.returncode == -9
        assert process.terminate_calls == process.kill_calls == 1
        assert events == ["close:stubborn", "terminate:stubborn", "kill:stubborn"]

    run_scenario(scenario())


def test_worker_that_ignores_eof_exits_after_terminate():
    async def scenario():
        events = []
        process = FakeProcess(
            "terminated",
            worker_offer("terminated"),
            events,
            exit_on_close=False,
        )
        supervisor = PluginWorkerSupervisor(
            FakeLauncher({"terminated": process}),
            stop_timeout=0.001,
        )
        await supervisor.start([worker_spec("terminated")])

        await supervisor.stop()

        assert process.returncode == -15
        assert process.terminate_calls == 1
        assert process.kill_calls == 0

    run_scenario(scenario())


def test_already_exited_worker_needs_no_shutdown_signal():
    async def scenario():
        events = []
        process = FakeProcess("exited", worker_offer("exited"), events)
        supervisor = PluginWorkerSupervisor(FakeLauncher({"exited": process}))
        await supervisor.start([worker_spec("exited")])
        process.exit(0)

        stopped = await supervisor.stop()

        assert stopped[0].state is WorkerState.STOPPED
        assert len(process.writer.writes) == 1
        assert events == []

    run_scenario(scenario())


def test_crashed_worker_cancels_orphans_and_recovers_with_diagnostics():
    async def scenario():
        events = []
        crashed = FakeProcess("recovering", worker_offer("recovering"), events, pid=31)
        recovered = FakeProcess("recovering", worker_offer("recovering"), events, pid=32)
        launcher = SequencedLauncher([crashed, recovered])
        observer = FailureObserver()
        supervisor = PluginWorkerSupervisor(
            launcher,
            recovery_policy=WorkerRecoveryPolicy(
                max_restarts=2,
                initial_backoff_seconds=0.001,
                max_backoff_seconds=0.001,
                backoff_multiplier=1,
            ),
            failure_observer=observer,
        )
        await supervisor.start([worker_spec("recovering")])

        crashed.exit(17)
        await asyncio.sleep(0.01)

        status = supervisor.statuses()[0]
        assert status.state is WorkerState.READY
        assert status.pid == 32
        assert status.restart_attempts == 1
        assert status.detail == "worker recovered after 1 restart attempts"
        assert observer.calls == [("recovering", "worker exited with status 17")]
        assert [item.code for item in supervisor.diagnostics("recovering")] == [
            WorkerDiagnosticCode.EXITED,
            WorkerDiagnosticCode.RECOVERED,
        ]
        assert launcher.calls == ["recovering", "recovering"]
        await supervisor.stop()

    run_scenario(scenario())


def test_restart_failures_are_redacted_and_budget_exhaustion_is_terminal():
    async def scenario():
        events = []
        crashed = FakeProcess("exhausted", worker_offer("exhausted"), events, pid=41)
        launcher = SequencedLauncher([crashed, OSError("secret one"), RuntimeError("secret two")])
        supervisor = PluginWorkerSupervisor(
            launcher,
            recovery_policy=WorkerRecoveryPolicy(
                max_restarts=2,
                initial_backoff_seconds=0.001,
                max_backoff_seconds=0.001,
                backoff_multiplier=1,
            ),
        )
        await supervisor.start([worker_spec("exhausted")])

        crashed.exit(9)
        await asyncio.sleep(0.02)

        status = supervisor.statuses()[0]
        assert status.state is WorkerState.EXHAUSTED
        assert status.restart_attempts == 2
        assert status.detail == (
            "worker restart budget exhausted after 2 attempts: worker launch failed: RuntimeError"
        )
        assert "secret" not in status.detail
        diagnostics = supervisor.diagnostics("exhausted")
        assert [item.code for item in diagnostics] == [
            WorkerDiagnosticCode.EXITED,
            WorkerDiagnosticCode.RESTART_FAILED,
            WorkerDiagnosticCode.RESTART_FAILED,
            WorkerDiagnosticCode.EXHAUSTED,
        ]
        assert [item.restart_attempt for item in diagnostics] == [0, 1, 2, 2]
        await asyncio.sleep(0.01)
        assert launcher.calls == ["exhausted"] * 3
        await supervisor.stop()

    run_scenario(scenario())


def test_recovery_reaps_rejected_handshake_then_uses_next_attempt():
    async def scenario():
        events = []
        crashed = FakeProcess("retrying", worker_offer("retrying"), events, pid=71)
        rejected = FakeProcess("retrying", worker_offer("substitute"), events, pid=72)
        recovered = FakeProcess("retrying", worker_offer("retrying"), events, pid=73)
        supervisor = PluginWorkerSupervisor(
            SequencedLauncher([crashed, rejected, recovered]),
            recovery_policy=WorkerRecoveryPolicy(
                max_restarts=2,
                initial_backoff_seconds=0.001,
                max_backoff_seconds=0.001,
                backoff_multiplier=1,
            ),
        )
        await supervisor.start([worker_spec("retrying")])
        crashed.exit(6)
        await asyncio.sleep(0.02)

        status = supervisor.statuses()[0]
        assert status.pid == 73
        assert status.restart_attempts == 2
        failed = supervisor.diagnostics("retrying")[1]
        assert failed == WorkerDiagnostic(
            "retrying",
            WorkerDiagnosticCode.RESTART_FAILED,
            "worker handshake rejected: plugin-identity-mismatch",
            72,
            None,
            1,
        )
        assert rejected.returncode == 0
        assert rejected.writer.closed
        await supervisor.stop()

    run_scenario(scenario())


def test_recovery_diagnostics_drop_oldest_entries_at_hard_bound():
    async def scenario():
        events = []
        crashed = FakeProcess("bounded", worker_offer("bounded"), events)
        failures = [OSError(str(index)) for index in range(16)]
        supervisor = PluginWorkerSupervisor(
            SequencedLauncher([crashed, *failures]),
            recovery_policy=WorkerRecoveryPolicy(
                max_restarts=16,
                initial_backoff_seconds=0.0001,
                max_backoff_seconds=0.0001,
                backoff_multiplier=1,
            ),
        )
        await supervisor.start([worker_spec("bounded")])
        crashed.exit(8)
        await asyncio.sleep(0.05)

        diagnostics = supervisor.diagnostics("bounded")
        assert len(diagnostics) == MAX_WORKER_DIAGNOSTICS
        assert diagnostics[0].restart_attempt == 2
        assert diagnostics[-1].code is WorkerDiagnosticCode.EXHAUSTED
        await supervisor.stop()

    run_scenario(scenario())


def test_observer_failure_is_contained_and_stop_cancels_pending_recovery():
    async def scenario():
        events = []
        crashed = FakeProcess("stopping", worker_offer("stopping"), events, pid=51)
        replacement = FakeProcess("stopping", worker_offer("stopping"), events, pid=52)
        launcher = SequencedLauncher([crashed, replacement])
        supervisor = PluginWorkerSupervisor(
            launcher,
            recovery_policy=WorkerRecoveryPolicy(
                max_restarts=1,
                initial_backoff_seconds=0.1,
                max_backoff_seconds=0.1,
            ),
            failure_observer=FailureObserver(ValueError("private")),
        )
        await supervisor.start([worker_spec("stopping")])
        crashed.exit(2)
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        restarting = supervisor.statuses()[0]
        assert restarting == WorkerStatus(
            "stopping",
            WorkerState.RESTARTING,
            51,
            2,
            "worker restart attempt 1",
            1,
        )

        stopped = await supervisor.stop()

        assert stopped[0].state is WorkerState.STOPPED
        assert stopped[0].detail == "worker recovery stopped"
        assert launcher.calls == ["stopping"]
        assert [item.code for item in supervisor.diagnostics("stopping")] == [
            WorkerDiagnosticCode.EXITED,
            WorkerDiagnosticCode.ORPHAN_CANCEL_FAILED,
        ]
        assert supervisor.diagnostics("missing") == ()

    run_scenario(scenario())


def test_stop_reaps_replacement_cancelled_during_handshake():
    async def scenario():
        events = []
        crashed = FakeProcess("reaping", worker_offer("reaping"), events, pid=61)
        silent = FakeProcess("reaping", None, events, pid=62)
        supervisor = PluginWorkerSupervisor(
            SequencedLauncher([crashed, silent]),
            handshake_timeout=1,
            recovery_policy=WorkerRecoveryPolicy(
                max_restarts=1,
                initial_backoff_seconds=0.0001,
                max_backoff_seconds=0.0001,
            ),
        )
        await supervisor.start([worker_spec("reaping")])
        crashed.exit(4)
        await asyncio.sleep(0.01)

        stopped = await supervisor.stop()

        assert stopped[0].state is WorkerState.STOPPED
        assert silent.writer.closed
        assert silent.returncode == 0

    run_scenario(scenario())


@pytest.mark.parametrize(
    "changes, match",
    [
        ({"max_restarts": True}, "max_restarts"),
        ({"max_restarts": -1}, "max_restarts"),
        ({"max_restarts": 17}, "max_restarts"),
        ({"initial_backoff_seconds": 0}, "initial_backoff_seconds"),
        (
            {"initial_backoff_seconds": 2, "max_backoff_seconds": 1},
            "must not exceed",
        ),
        ({"backoff_multiplier": True}, "backoff_multiplier"),
        ({"backoff_multiplier": 0.5}, "backoff_multiplier"),
        ({"backoff_multiplier": float("inf")}, "backoff_multiplier"),
    ],
)
def test_recovery_policy_is_bounded(changes, match):
    with pytest.raises(ValueError, match=match):
        WorkerRecoveryPolicy(**changes)


def test_recovery_backoff_is_exponential_and_capped():
    policy = WorkerRecoveryPolicy(
        max_restarts=3,
        initial_backoff_seconds=0.5,
        max_backoff_seconds=1.5,
        backoff_multiplier=2,
    )

    assert [policy.delay(attempt) for attempt in (1, 2, 3)] == [0.5, 1.0, 1.5]
    for attempt in (True, 0, 17, 1.5):
        with pytest.raises(ValueError, match="attempt"):
            policy.delay(attempt)


@pytest.mark.parametrize("timeout", [0, -1, True, "1", float("inf"), float("nan")])
def test_timeouts_must_be_positive_numbers(timeout):
    with pytest.raises(ValueError, match="handshake_timeout must be positive"):
        PluginWorkerSupervisor(handshake_timeout=timeout)
    with pytest.raises(ValueError, match="stop_timeout must be positive"):
        PluginWorkerSupervisor(stop_timeout=timeout)
    with pytest.raises(ValueError, match="cancel_timeout must be positive"):
        PluginWorkerSupervisor(cancel_timeout=timeout)


def test_worker_specs_are_bounded_unique_and_protocol_validated():
    async def rejected(specs, match):
        supervisor = PluginWorkerSupervisor(FakeLauncher())
        with pytest.raises(ValueError, match=match):
            await supervisor.start(specs)
        assert not supervisor.running

    async def scenario():
        await rejected(
            [worker_spec(f"plugin-{index}") for index in range(MAX_SUPERVISED_WORKERS + 1)],
            "at most 128 workers",
        )
        await rejected([worker_spec("same"), worker_spec("same")], "must be unique")
        await rejected([WorkerSpec("empty", ())], "argv must contain")
        await rejected(
            [WorkerSpec("many", tuple("x" for _ in range(MAX_WORKER_ARGUMENTS + 1)))],
            "argv must contain",
        )
        for argument in (None, "", "bad\0arg", "x" * (MAX_WORKER_ARGUMENT_CHARS + 1)):
            await rejected([WorkerSpec("invalid", (argument,))], "invalid argument")
        await rejected([WorkerSpec("Bad_ID", ("worker",))], "invalid plugin id")

    run_scenario(scenario())


def test_start_is_single_use_until_stopped_and_empty_catalog_is_valid():
    async def scenario():
        supervisor = PluginWorkerSupervisor(FakeLauncher())
        assert await supervisor.start([]) == ()
        assert supervisor.running
        with pytest.raises(RuntimeError, match="already running"):
            await supervisor.start([])
        assert await supervisor.stop() == ()
        assert await supervisor.start([]) == ()
        await supervisor.stop()

    run_scenario(scenario())


@dataclass
class StubSubprocess:
    stdout: object = None
    stdin: object = None
    pid: int = 77
    returncode: int | None = None
    terminate_calls: int = 0
    kill_calls: int = 0

    async def wait(self):
        return 0

    def terminate(self):
        self.terminate_calls += 1

    def kill(self):
        self.kill_calls += 1


def test_asyncio_launcher_uses_isolated_bounded_process_options(monkeypatch):
    async def scenario():
        reader = asyncio.StreamReader()
        events = []
        process_owner = type("Owner", (), {"plugin_id": "real", "reader": reader})()
        writer = FakeWriter(process_owner, None, events)
        process = StubSubprocess(stdout=reader, stdin=writer)
        calls = []

        async def create(*argv, **options):
            calls.append((argv, options))
            return process

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
        launched = await AsyncioSubprocessLauncher().launch(worker_spec("real"))

        assert launched.pid == 77
        assert launched.returncode is None
        assert await launched.wait() == 0
        launched.terminate()
        launched.kill()
        assert process.terminate_calls == process.kill_calls == 1
        argv, options = calls[0]
        assert argv == ("python", "worker.py", "real")
        assert options["stdin"] is asyncio.subprocess.PIPE
        assert options["stdout"] is asyncio.subprocess.PIPE
        assert options["stderr"] is asyncio.subprocess.DEVNULL
        assert options["close_fds"] is True
        assert options["start_new_session"] is True

    run_scenario(scenario())


def test_asyncio_launcher_wraps_a_sandboxed_worker(monkeypatch, tmp_path):
    from omnitensor.plugins import FilesystemSandbox

    async def scenario():
        reader = asyncio.StreamReader()
        owner = type("Owner", (), {"plugin_id": "real", "reader": reader})()
        process = StubSubprocess(stdout=reader, stdin=FakeWriter(owner, None, []))
        calls = []

        async def create(*argv, **options):
            calls.append((argv, options))
            return process

        permission = f"read:{tmp_path}"
        sandbox = FilesystemSandbox.from_permissions({permission}, {permission})
        original = worker_spec("real")
        spec = WorkerSpec(
            original.plugin_id,
            original.argv,
            original.minimum_protocol,
            original.maximum_protocol,
            original.capabilities,
            sandbox,
        )
        monkeypatch.setattr(asyncio, "create_subprocess_exec", create)

        await AsyncioSubprocessLauncher().launch(spec)

        assert calls[0][0] == sandbox.wrap(spec.argv)

    run_scenario(scenario())


def test_asyncio_launcher_rejects_missing_process_pipes(monkeypatch):
    async def scenario():
        async def create(*_argv, **_options):
            return StubSubprocess()

        monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
        with pytest.raises(RuntimeError, match="worker pipes are unavailable"):
            await AsyncioSubprocessLauncher().launch(worker_spec("real"))

    run_scenario(scenario())


class GatedLock:
    """The supervisor lock with one acquire parked on demand.

    Reproduces the window in which a recovery has relaunched a worker but has
    not yet taken the lock that would install it into a slot: the worker is
    alive and nothing else owns it.
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self._gate = asyncio.Event()
        self.reached = asyncio.Event()
        self.armed = False

    async def __aenter__(self):
        if self.armed:
            self.armed = False
            self.reached.set()
            await self._gate.wait()
        await self._lock.acquire()
        return None

    async def __aexit__(self, *exc_info):
        self._lock.release()
        return False


def test_cancelling_a_recovery_after_relaunch_reaps_the_new_worker():
    async def scenario():
        events = []
        crashed = FakeProcess("alpha", worker_offer("alpha"), events, pid=61)
        replacement = FakeProcess(
            "alpha",
            worker_offer("alpha"),
            events,
            pid=62,
            exit_on_close=False,
        )
        launcher = SequencedLauncher([crashed, replacement])
        supervisor = PluginWorkerSupervisor(
            launcher,
            stop_timeout=0.01,
            recovery_policy=WorkerRecoveryPolicy(
                max_restarts=1,
                initial_backoff_seconds=0.01,
                max_backoff_seconds=0.01,
            ),
        )
        await supervisor.start([worker_spec("alpha")])

        gate = GatedLock()
        supervisor._lock = gate
        crashed.exit(2)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert supervisor.statuses()[0].state is WorkerState.RESTARTING

        # The next acquire is the post-relaunch one; park the recovery there.
        gate.armed = True
        await asyncio.wait_for(gate.reached.wait(), timeout=1)
        assert launcher.calls == ["alpha", "alpha"]
        assert replacement.returncode is None

        stopped = await supervisor.stop()

        assert replacement.returncode is not None, "relaunched worker was never reaped"
        assert replacement.writer.closed
        assert stopped[0].state is WorkerState.STOPPED
        assert not supervisor.running

    run_scenario(scenario())


def test_an_unkillable_worker_does_not_abort_the_rest_of_shutdown():
    """stop() used to raise out mid-loop, leaving later workers running."""

    async def scenario():
        events = []
        unkillable = FakeProcess(
            "beta",
            worker_offer("beta"),
            events,
            pid=71,
            exit_on_close=False,
            exit_on_terminate=False,
        )
        unkillable.kill = lambda: events.append("kill:beta")
        healthy = FakeProcess("alpha", worker_offer("alpha"), events, pid=70)
        launcher = FakeLauncher({"alpha": healthy, "beta": unkillable})
        supervisor = PluginWorkerSupervisor(launcher, stop_timeout=0.01)
        await supervisor.start([worker_spec("alpha"), worker_spec("beta")])

        stopped = await supervisor.stop()

        by_id = {status.plugin_id: status for status in stopped}
        assert by_id["beta"].detail == "worker survived kill"
        assert by_id["alpha"].detail == "worker stopped"
        assert healthy.returncode == 0, "the next worker was never stopped"
        assert [item.code for item in supervisor.diagnostics("beta")] == [
            WorkerDiagnosticCode.STOP_FAILED
        ]
        assert not supervisor.running
        assert supervisor._startup_order == []
        assert await supervisor.stop() == stopped

    run_scenario(scenario())


def test_a_failed_handshake_reports_its_own_error_when_the_child_will_not_die():
    """_force_stop must not replace the failure it was cleaning up after."""

    async def scenario():
        events = []
        stubborn = FakeProcess(
            "alpha",
            None,
            events,
            pid=72,
            exit_on_close=False,
            exit_on_terminate=False,
        )
        stubborn.kill = lambda: events.append("kill:alpha")
        supervisor = PluginWorkerSupervisor(
            FakeLauncher({"alpha": stubborn}),
            handshake_timeout=0.01,
            stop_timeout=0.01,
        )

        started = await supervisor.start([worker_spec("alpha")])

        assert started[0].state is WorkerState.FAILED
        assert started[0].detail == "worker handshake timed out"

    run_scenario(scenario())


def test_slow_plugin_startup_is_not_bounded_by_the_handshake_deadline():
    """One deadline over handshake plus startup made a slow plugin unstartable."""

    async def scenario():
        events = []
        slow = FakeProcess("alpha", worker_offer("alpha"), events, pid=80, ready_delay=0.05)
        supervisor = PluginWorkerSupervisor(
            FakeLauncher({"alpha": slow}),
            handshake_timeout=0.01,
            startup_timeout=1.0,
            stop_timeout=0.01,
        )

        started = await supervisor.start([worker_spec("alpha")])

        assert started[0].state is WorkerState.READY
        assert started[0].protocol_version == 2
        await supervisor.stop()

    run_scenario(scenario())


def test_a_worker_that_never_reports_ready_fails_with_its_own_reason():
    async def scenario():
        events = []
        stuck = FakeProcess(
            "alpha",
            worker_offer("alpha"),
            events,
            pid=81,
            reports_ready=False,
        )
        supervisor = PluginWorkerSupervisor(
            FakeLauncher({"alpha": stuck}),
            startup_timeout=0.01,
            stop_timeout=0.01,
        )

        started = await supervisor.start([worker_spec("alpha")])

        assert started[0].state is WorkerState.FAILED
        assert started[0].detail == "worker startup timed out"
        assert started[0].pid == 81
        assert stuck.returncode is not None, "the stuck worker was not reaped"

    run_scenario(scenario())


def test_a_startup_timeout_does_not_consume_the_restart_budget():
    async def scenario():
        events = []
        crashed = FakeProcess("alpha", worker_offer("alpha"), events, pid=82)
        replacement = FakeProcess(
            "alpha",
            worker_offer("alpha"),
            events,
            pid=83,
            reports_ready=False,
        )
        launcher = SequencedLauncher([crashed, replacement])
        supervisor = PluginWorkerSupervisor(
            launcher,
            startup_timeout=0.01,
            stop_timeout=0.01,
            recovery_policy=WorkerRecoveryPolicy(
                max_restarts=3,
                initial_backoff_seconds=0.001,
                max_backoff_seconds=0.001,
            ),
        )
        await supervisor.start([worker_spec("alpha")])
        crashed.exit(2)
        for _ in range(200):
            if supervisor.statuses()[0].state is WorkerState.FAILED:
                break
            await asyncio.sleep(0.001)

        status = supervisor.statuses()[0]
        assert status.state is WorkerState.FAILED
        assert status.detail == "worker startup timed out"
        assert launcher.calls == ["alpha", "alpha"], "the budget was spent on retries"
        codes = [item.code for item in supervisor.diagnostics("alpha")]
        assert codes == [
            WorkerDiagnosticCode.EXITED,
            WorkerDiagnosticCode.STARTUP_TIMEOUT,
        ]
        assert WorkerDiagnosticCode.EXHAUSTED not in codes
        await supervisor.stop()

    run_scenario(scenario())
