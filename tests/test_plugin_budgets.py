from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable
from typing import TypeVar

import pytest

from omnitensor.plugins import (
    DEFAULT_CALL_TIMEOUT_SECONDS,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_DESCRIPTORS,
    DEFAULT_MAX_MEMORY_BYTES,
    DEFAULT_MAX_OUTPUT_BYTES,
    DEFAULT_MAX_PROCESSES,
    DEFAULT_RESOURCE_POLL_SECONDS,
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
        DEFAULT_MAX_OUTPUT_BYTES,
        DEFAULT_MAX_CONCURRENCY,
        DEFAULT_RESOURCE_POLL_SECONDS,
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
        ({"max_output_bytes": DEFAULT_MAX_OUTPUT_BYTES + 1}, "max_output_bytes"),
        ({"max_concurrency": MAX_CONCURRENCY_LIMIT + 1}, "max_concurrency"),
        ({"resource_poll_seconds": 0}, "resource_poll_seconds"),
        (
            {"call_timeout_seconds": 0.5, "resource_poll_seconds": 0.6},
            "resource_poll_seconds",
        ),
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


def test_successful_call_checks_resources_output_and_accounting():
    async def scenario():
        probes = []

        def probe():
            probes.append(True)
            return usage()

        enforcer = WorkerBudgetEnforcer(WorkerBudgetLimits(), probe)
        result = await enforcer.run(lambda: asyncio.sleep(0, result={"ok": True}))

        assert result == {"ok": True}
        assert len(probes) == 2
        assert enforcer.snapshot() == WorkerBudgetSnapshot(0, 1, 1, 0)

    run_scenario(scenario())


@pytest.mark.parametrize(
    "observed, code, detail",
    [
        (
            usage(processes=2, memory_bytes=1),
            WorkerBudgetCode.PROCESS,
            "2 processes; limit is 1",
        ),
        (usage(memory_bytes=11), WorkerBudgetCode.MEMORY, "11 memory bytes; limit is 10"),
        (
            usage(memory_bytes=1, descriptors=5),
            WorkerBudgetCode.DESCRIPTOR,
            "5 descriptors; limit is 4",
        ),
    ],
)
def test_resource_budgets_fail_before_call(observed, code, detail):
    async def scenario():
        called = False

        async def operation():
            nonlocal called
            called = True

        enforcer = WorkerBudgetEnforcer(
            WorkerBudgetLimits(
                max_memory_bytes=10,
                max_descriptors=4,
            ),
            lambda: observed,
        )
        with pytest.raises(WorkerBudgetExceededError) as error:
            await enforcer.run(operation)

        assert error.value.code is code
        assert error.value.detail == f"worker reported {detail}"
        assert str(error.value) == f"{code}: worker reported {detail}"
        assert called is False
        assert enforcer.snapshot() == WorkerBudgetSnapshot(0, 1, 0, 1)

    run_scenario(scenario())


def test_dynamic_resource_violation_cancels_inflight_call():
    async def scenario():
        probes = iter((usage(memory_bytes=1), usage(memory_bytes=20)))
        cancelled = asyncio.Event()

        async def operation():
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        enforcer = WorkerBudgetEnforcer(
            WorkerBudgetLimits(
                call_timeout_seconds=1,
                max_memory_bytes=10,
                resource_poll_seconds=0.001,
            ),
            lambda: next(probes),
        )
        with pytest.raises(WorkerBudgetExceededError) as error:
            await enforcer.run(operation)

        assert error.value.code is WorkerBudgetCode.MEMORY
        assert cancelled.is_set()
        assert enforcer.snapshot() == WorkerBudgetSnapshot(0, 1, 0, 1)

    run_scenario(scenario())


def test_deadline_cancels_inflight_call_with_stable_failure():
    async def scenario():
        cancelled = asyncio.Event()

        async def operation():
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        enforcer = WorkerBudgetEnforcer(
            WorkerBudgetLimits(
                call_timeout_seconds=0.005,
                resource_poll_seconds=0.001,
            ),
            usage,
        )
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

        enforcer = WorkerBudgetEnforcer(
            WorkerBudgetLimits(call_timeout_seconds=1, resource_poll_seconds=0.01),
            usage,
        )
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


@pytest.mark.parametrize(
    "result, detail",
    [
        ({"value": math.nan}, "worker output is not finite JSON"),
        ({"value": object()}, "worker output is not finite JSON"),
        ({"value": "12345"}, "worker output is 17 bytes; limit is 16"),
    ],
)
def test_output_budget_rejects_invalid_or_oversized_results(result, detail):
    async def scenario():
        enforcer = WorkerBudgetEnforcer(
            WorkerBudgetLimits(max_output_bytes=16),
            usage,
        )
        with pytest.raises(WorkerBudgetExceededError) as error:
            await enforcer.run(lambda: asyncio.sleep(0, result=result))
        assert error.value.code is WorkerBudgetCode.OUTPUT
        assert error.value.detail == detail
        assert enforcer.snapshot() == WorkerBudgetSnapshot(0, 1, 0, 1)

    run_scenario(scenario())


def test_probe_failures_are_redacted_and_fail_closed():
    async def scenario():
        def fail():
            raise OSError("private path")

        for probe, detail in (
            (fail, "resource usage probe failed: OSError"),
            (lambda: object(), "resource usage probe returned an invalid value"),
        ):
            enforcer = WorkerBudgetEnforcer(WorkerBudgetLimits(), probe)
            with pytest.raises(WorkerBudgetExceededError) as error:
                await enforcer.run(lambda: asyncio.sleep(0))
            assert error.value.code is WorkerBudgetCode.USAGE_UNAVAILABLE
            assert error.value.detail == detail
            assert "private" not in str(error.value)

    run_scenario(scenario())


def test_call_contract_and_caller_cancellation_cleanup():
    with pytest.raises(TypeError, match="usage_probe must be callable"):
        WorkerBudgetEnforcer(WorkerBudgetLimits(), object())

    async def scenario():
        enforcer = WorkerBudgetEnforcer(WorkerBudgetLimits(), usage)
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
