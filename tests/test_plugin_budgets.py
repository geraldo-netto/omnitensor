from __future__ import annotations

import asyncio
import errno
import math
from collections.abc import Awaitable
from pathlib import Path
from typing import TypeVar

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.plugins import (
    DEFAULT_CALL_TIMEOUT_SECONDS,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_DESCRIPTORS,
    DEFAULT_MAX_MEMORY_BYTES,
    DEFAULT_MAX_PROCESSES,
    MAX_CALL_TIMEOUT_SECONDS,
    MAX_CONCURRENCY_LIMIT,
    MAX_DESCRIPTORS_LIMIT,
    MAX_MEMORY_BYTES_LIMIT,
    MAX_PROCESSES_LIMIT,
    ProcfsWorkerUsageProbe,
    WorkerBudgetCode,
    WorkerBudgetEnforcer,
    WorkerBudgetExceededError,
    WorkerBudgetLimits,
    WorkerBudgetSnapshot,
    WorkerResourceUsage,
)
from omnitensor.plugins.protocol import PluginResult, PluginResultStatus

T = TypeVar("T")


def run_scenario(coroutine: Awaitable[T]) -> T:
    async def bounded() -> T:
        return await asyncio.wait_for(coroutine, timeout=0.5)

    return asyncio.run(bounded())


def usage(processes=1, memory_bytes=1024, descriptors=3):
    return WorkerResourceUsage(processes, memory_bytes, descriptors)


def test_default_limits_are_explicit_and_bounded():
    limits = WorkerBudgetLimits()

    assert limits == WorkerBudgetLimits(
        DEFAULT_CALL_TIMEOUT_SECONDS,
        DEFAULT_MAX_PROCESSES,
        DEFAULT_MAX_MEMORY_BYTES,
        DEFAULT_MAX_DESCRIPTORS,
        DEFAULT_MAX_CONCURRENCY,
    )


@pytest.mark.parametrize(
    "changes, detail",
    [
        ({"call_timeout_seconds": 0}, "call_timeout_seconds"),
        ({"call_timeout_seconds": True}, "call_timeout_seconds"),
        ({"call_timeout_seconds": math.inf}, "call_timeout_seconds"),
        ({"call_timeout_seconds": MAX_CALL_TIMEOUT_SECONDS + 1}, "call_timeout_seconds"),
        ({"max_processes": 0}, "max_processes"),
        ({"max_processes": MAX_PROCESSES_LIMIT + 1}, "max_processes"),
        ({"max_memory_bytes": True}, "max_memory_bytes"),
        ({"max_memory_bytes": MAX_MEMORY_BYTES_LIMIT + 1}, "max_memory_bytes"),
        ({"max_descriptors": 0}, "max_descriptors"),
        ({"max_descriptors": MAX_DESCRIPTORS_LIMIT + 1}, "max_descriptors"),
        ({"max_concurrency": MAX_CONCURRENCY_LIMIT + 1}, "max_concurrency"),
    ],
)
def test_limits_reject_invalid_values(changes, detail):
    with pytest.raises(ValueError, match=detail):
        WorkerBudgetLimits(**changes)


@pytest.mark.parametrize(
    "values, detail",
    [
        ((-1, 0, 0), "processes"),
        ((0, -1, 0), "memory_bytes"),
        ((0, 0, -1), "descriptors"),
        ((True, 0, 0), "processes"),
    ],
)
def test_resource_usage_requires_non_negative_integers(values, detail):
    with pytest.raises(ValueError, match=f"{detail} must be a non-negative integer"):
        WorkerResourceUsage(*values)


def test_successful_call_reports_its_output_and_accounting():
    async def scenario():
        enforcer = WorkerBudgetEnforcer(WorkerBudgetLimits())
        result = await enforcer.run(lambda: asyncio.sleep(0, result={"ok": True}))

        assert result == {"ok": True}
        assert enforcer.snapshot() == WorkerBudgetSnapshot(0, 1, 1, 0)

    run_scenario(scenario())


# Workers run unbounded. These ceilings refused real work: a Qwen worker that
# had already loaded its model and was answering was killed for holding 831 MB
# against a 512 MiB limit, and the cgroup meant to enforce the same numbers
# could never be created in the first place. They now describe the worker's
# cgroup and nothing else; the enforcer never reads them.
def test_declared_ceilings_do_not_bound_the_call():
    async def scenario():
        called = False

        async def operation():
            nonlocal called
            called = True
            return "answer"

        enforcer = WorkerBudgetEnforcer(
            WorkerBudgetLimits(max_processes=1, max_memory_bytes=10, max_descriptors=4)
        )

        assert await enforcer.run(operation) == "answer"
        assert called is True
        assert enforcer.snapshot() == WorkerBudgetSnapshot(0, 1, 1, 0)

    run_scenario(scenario())


def test_deadline_cancels_inflight_call_with_stable_failure():
    async def scenario():
        cancelled = asyncio.Event()

        async def operation():
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        enforcer = WorkerBudgetEnforcer(WorkerBudgetLimits(call_timeout_seconds=0.005))
        with pytest.raises(WorkerBudgetExceededError) as error:
            await enforcer.run(operation)

        assert error.value.code is WorkerBudgetCode.DEADLINE
        assert error.value.detail == "call exceeded 0.005 seconds"
        assert cancelled.is_set()
        assert enforcer.snapshot() == WorkerBudgetSnapshot(0, 1, 0, 1)

    run_scenario(scenario())


def test_concurrency_rejects_without_constructing_an_operation():
    async def scenario():
        release = asyncio.Event()
        entered = asyncio.Event()
        second_called = False

        async def first():
            entered.set()
            await release.wait()
            return {"first": True}

        def second():
            nonlocal second_called
            second_called = True
            return asyncio.sleep(0)

        enforcer = WorkerBudgetEnforcer(WorkerBudgetLimits(call_timeout_seconds=1))
        running = asyncio.create_task(enforcer.run(first))
        await entered.wait()
        with pytest.raises(WorkerBudgetExceededError) as error:
            await enforcer.run(second)
        assert error.value.code is WorkerBudgetCode.CONCURRENCY
        assert error.value.detail == "at most 1 calls may run concurrently"
        assert second_called is False
        release.set()
        assert await running == {"first": True}
        assert enforcer.snapshot() == WorkerBudgetSnapshot(0, 1, 1, 1)

    run_scenario(scenario())


@given(st.text(max_size=64))
def test_no_answer_is_refused_for_its_size(value):
    """A long answer is not a budget violation. It used to be discarded here
    and to cost a model reload; the transport splits it across frames now."""
    result = PluginResult(
        "job",
        PluginResultStatus.SUCCEEDED,
        {"value": value * 64},
        "",
        1,
    )

    async def scenario():
        enforcer = WorkerBudgetEnforcer(WorkerBudgetLimits())
        assert await enforcer.run(lambda: asyncio.sleep(0, result=result)) == result
        assert enforcer.snapshot() == WorkerBudgetSnapshot(0, 1, 1, 0)

    run_scenario(scenario())


def test_call_contract_and_caller_cancellation_cleanup():
    async def scenario():
        enforcer = WorkerBudgetEnforcer(WorkerBudgetLimits())
        with pytest.raises(TypeError, match="operation must be callable"):
            await enforcer.run(object())

        entered = asyncio.Event()

        async def operation():
            entered.set()
            await asyncio.Event().wait()

        running = asyncio.create_task(enforcer.run(operation))
        await entered.wait()
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert enforcer.snapshot() == WorkerBudgetSnapshot(0, 1, 0, 0)

        async def fail():
            raise RuntimeError("operation failure")

        with pytest.raises(RuntimeError, match="operation failure"):
            await enforcer.run(fail)
        assert enforcer.snapshot() == WorkerBudgetSnapshot(0, 1, 0, 0)

    run_scenario(scenario())


def _write_process(proc_root, pid, *, children="", rss_kb=1, descriptors=1):
    task = proc_root / str(pid) / "task" / str(pid)
    task.mkdir(parents=True)
    (task / "children").write_text(children, encoding="ascii")
    (proc_root / str(pid) / "status").write_text(
        f"Name:\tfixture\nVmRSS:\t{rss_kb} kB\n",
        encoding="ascii",
    )
    fd = proc_root / str(pid) / "fd"
    fd.mkdir()
    for index in range(descriptors):
        (fd / str(index)).touch()


def test_procfs_probe_counts_recursive_process_resources_and_cycles(tmp_path):
    _write_process(tmp_path, 10, children="11 12", rss_kb=2, descriptors=1)
    _write_process(tmp_path, 11, children="12", rss_kb=3, descriptors=2)
    _write_process(tmp_path, 12, children="10", rss_kb=5, descriptors=3)

    probe = ProcfsWorkerUsageProbe(10, proc_root=tmp_path)

    assert probe() == WorkerResourceUsage(3, 10 * 1024, 6)


def test_procfs_probe_validates_pid_and_missing_memory_field(tmp_path):
    with pytest.raises(ValueError, match="pid must be a positive integer"):
        ProcfsWorkerUsageProbe(True)
    _write_process(tmp_path, 10)
    (tmp_path / "10/status").write_text("Name:\tfixture\n", encoding="ascii")
    with pytest.raises(OSError, match="VmRSS is unavailable"):
        ProcfsWorkerUsageProbe(10, proc_root=tmp_path)()


def test_cancelling_a_call_does_not_leak_its_concurrency_slot():
    """The decrement sat behind two awaits, so cancellation there leaked a slot."""

    async def scenario():
        enforcer = WorkerBudgetEnforcer(WorkerBudgetLimits(max_concurrency=1))
        started = asyncio.Event()

        async def slow_to_cancel():
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                # Cleanup that outlives the first cancellation, so run()'s
                # own cleanup is still suspended when the second one lands.
                await asyncio.sleep(0.05)
                raise

        call = asyncio.create_task(enforcer.run(slow_to_cancel))
        await started.wait()
        assert enforcer.snapshot().active_calls == 1

        call.cancel()
        await asyncio.sleep(0.01)
        call.cancel()
        await asyncio.gather(call, return_exceptions=True)

        assert enforcer.snapshot().active_calls == 0, "a concurrency slot was leaked"
        assert await enforcer.run(lambda: _completed_value("second call")) == "second call"

    asyncio.run(asyncio.wait_for(scenario(), timeout=5))


async def _completed_value(value):
    return value


def _write_thread(proc_root, pid, tid, *, children=""):
    """A non-main thread of ``pid``, which is where forks are often recorded."""
    task = proc_root / str(pid) / "task" / str(tid)
    task.mkdir(parents=True)
    (task / "children").write_text(children, encoding="ascii")


def test_procfs_probe_counts_children_forked_from_a_non_main_thread(tmp_path):
    """Reading only task/<pid>/children let a thread's fork escape every limit."""
    _write_process(tmp_path, 10, children="", rss_kb=2, descriptors=1)
    _write_thread(tmp_path, 10, 17, children="11")
    _write_process(tmp_path, 11, rss_kb=3, descriptors=2)

    usage = ProcfsWorkerUsageProbe(10, proc_root=tmp_path)()

    assert usage == WorkerResourceUsage(2, 5 * 1024, 3)


def test_procfs_probe_treats_a_pid_that_exits_mid_probe_as_no_usage(tmp_path):
    _write_process(tmp_path, 10, children="11", rss_kb=2, descriptors=1)

    usage = ProcfsWorkerUsageProbe(10, proc_root=tmp_path)()

    assert usage == WorkerResourceUsage(2, 2 * 1024, 1)


def test_procfs_probe_tolerates_a_thread_that_exits_between_listing_and_read(tmp_path, monkeypatch):
    _write_process(tmp_path, 10, children="", rss_kb=2, descriptors=1)
    _write_thread(tmp_path, 10, 17, children="11")
    original = Path.read_text

    def vanish(self, *args, **kwargs):
        if self.parent.name == "17":
            raise OSError(errno.ESRCH, "no such process")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", vanish)
    assert ProcfsWorkerUsageProbe(10, proc_root=tmp_path)() == WorkerResourceUsage(1, 2 * 1024, 1)


def test_procfs_probe_still_reports_unexpected_errors(tmp_path, monkeypatch):
    _write_process(tmp_path, 10, children="", rss_kb=2, descriptors=1)
    original = Path.read_text

    def refuse(self, *args, **kwargs):
        if self.name == "children":
            raise OSError(errno.EACCES, "permission denied")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", refuse)
    with pytest.raises(OSError):
        ProcfsWorkerUsageProbe(10, proc_root=tmp_path)()


def _write_cgroup(root, *, pids=(), memory_bytes=0):
    root.mkdir(parents=True, exist_ok=True)
    (root / "cgroup.procs").write_text("".join(f"{pid}\n" for pid in pids), encoding="ascii")
    (root / "memory.current").write_text(f"{memory_bytes}\n", encoding="ascii")
    return root
