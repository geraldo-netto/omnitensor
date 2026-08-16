"""The weighted pool that makes ``weight`` mean something for plugin jobs."""

from __future__ import annotations

import asyncio

import pytest

from omnitensor.dispatch_routing import PluginAwareDispatcher, admit_plugin_job
from omnitensor.job_ports import JobDispatchError
from omnitensor.plugin_admission import PluginAdmissionQueue
from omnitensor.profile_selection import plugin_profile_statuses
from omnitensor.state import PolicyState, ProfilePolicy


def pool(weights=None, admits=None, **kwargs) -> PluginAdmissionQueue:
    weights = weights or {}
    return PluginAdmissionQueue(
        weight_of=lambda profile_id: weights.get(profile_id, 1),
        admits=admits,
        **kwargs,
    )


def returning(value):
    async def body():
        return value

    return body


class TestBounds:
    def test_a_pool_size_the_host_could_not_hold_is_refused(self):
        with pytest.raises(ValueError, match="max_concurrent"):
            pool(max_concurrent=0)
        with pytest.raises(ValueError, match="max_concurrent"):
            pool(max_concurrent=33)

    def test_the_waiting_bound_is_validated_too(self):
        with pytest.raises(ValueError, match="max_waiting"):
            pool(max_waiting=0)

    async def test_an_uncontended_job_never_waits(self):
        queue = pool(max_concurrent=1)
        assert await queue.run("media-transcription", returning(7)) == 7
        assert queue.running() == 0
        assert queue.waiting() == 0


class TestWeighting:
    async def test_a_full_pool_serves_by_weight_not_by_arrival(self):
        """Stride: the weight-5 profile overtakes the weight-1 job queued before it."""
        queue = pool(weights={"heavy": 5, "light": 1}, max_concurrent=1)
        served: list[str] = []
        released = asyncio.Event()

        async def blocker():
            await released.wait()

        def record(profile_id):
            async def body():
                served.append(profile_id)

            return body

        held = asyncio.ensure_future(queue.run("blocker", blocker))
        await asyncio.sleep(0)

        waiting = [
            asyncio.ensure_future(queue.run("light", record("light"))),
            asyncio.ensure_future(queue.run("light", record("light"))),
            asyncio.ensure_future(queue.run("heavy", record("heavy"))),
        ]
        await asyncio.sleep(0)
        assert queue.waiting() == 3

        released.set()
        await held
        await asyncio.gather(*waiting)
        assert served == ["light", "heavy", "light"]

    async def test_weight_orders_nothing_when_the_pool_is_not_contended(self):
        queue = pool(weights={"heavy": 5, "light": 1}, max_concurrent=2)
        served: list[str] = []

        def record(profile_id):
            async def body():
                served.append(profile_id)

            return body

        await asyncio.gather(
            queue.run("light", record("light")),
            queue.run("heavy", record("heavy")),
        )
        assert served == ["light", "heavy"]


class TestPolicy:
    async def test_a_disabled_profile_is_held_and_runs_once_re_enabled(self):
        enabled = {"file-organizer": False}
        queue = pool(admits=lambda profile_id: enabled.get(profile_id, True), max_concurrent=1)
        done = asyncio.Event()

        async def body():
            done.set()

        waiter = asyncio.ensure_future(queue.run("file-organizer", body))
        await asyncio.sleep(0)
        assert queue.waiting() == 1
        assert not done.is_set()

        enabled["file-organizer"] = True
        queue.kick()
        await waiter
        assert done.is_set()
        assert queue.waiting() == 0

    async def test_a_held_profile_does_not_bank_credit_while_it_waits(self):
        """The clamp: being disabled must not buy a burst on re-enable."""
        enabled = {"held": False, "other": True}
        queue = pool(
            weights={"held": 1, "other": 1},
            admits=lambda profile_id: enabled.get(profile_id, True),
            max_concurrent=1,
        )
        served: list[str] = []
        released = asyncio.Event()

        async def blocker():
            await released.wait()

        def record(profile_id):
            async def body():
                served.append(profile_id)

            return body

        held_slot = asyncio.ensure_future(queue.run("blocker", blocker))
        await asyncio.sleep(0)
        waiting = [
            asyncio.ensure_future(queue.run("held", record("held"))),
            asyncio.ensure_future(queue.run("other", record("other"))),
            asyncio.ensure_future(queue.run("other", record("other"))),
        ]
        await asyncio.sleep(0)

        released.set()
        await held_slot
        await asyncio.sleep(0)
        enabled["held"] = True
        queue.kick()
        await asyncio.gather(*waiting)
        # "held" queued first but was not eligible, so it is served level with
        # the others rather than ahead of both of them.
        assert served.count("other") == 2
        assert served[0] == "other"

    async def test_a_held_profile_does_not_block_an_eligible_one_from_a_free_slot(self):
        enabled = {"held": False}
        queue = pool(admits=lambda profile_id: enabled.get(profile_id, True), max_concurrent=1)
        blocked = asyncio.ensure_future(queue.run("held", returning(1)))
        await asyncio.sleep(0)
        assert queue.waiting() == 1

        assert await queue.run("media-transcription", returning(2)) == 2

        blocked.cancel()
        await asyncio.gather(blocked, return_exceptions=True)

    async def test_cancelling_a_queued_job_leaves_no_slot_behind(self):
        queue = pool(max_concurrent=1)
        released = asyncio.Event()

        async def blocker():
            await released.wait()

        held = asyncio.ensure_future(queue.run("a", blocker))
        await asyncio.sleep(0)
        waiter = asyncio.ensure_future(queue.run("b", returning(1)))
        await asyncio.sleep(0)
        assert queue.waiting() == 1

        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert queue.waiting() == 0

        released.set()
        await held
        assert queue.running() == 0
        assert await queue.run("c", returning(2)) == 2


class TestAdmission:
    async def test_a_full_waiting_list_refuses_before_a_job_id_exists(self):
        queue = pool(max_concurrent=1, max_waiting=1)
        # The state a full pool is in: one job waiting, the bound reached.
        queue._push("a", asyncio.get_running_loop().create_future())  # noqa: SLF001
        with pytest.raises(JobDispatchError) as error:
            admit_plugin_job(
                _NoInference(),
                _RecordingPlugins(),
                lambda: frozenset({"file-organizer"}),
                queue,
                "file-organizer",
                {},
            )
        assert error.value.code == "plugin-queue-full"

    def test_a_plugin_job_that_can_wait_is_admitted(self):
        plugins = _RecordingPlugins()
        admit_plugin_job(
            _NoInference(),
            plugins,
            lambda: frozenset({"file-organizer"}),
            pool(),
            "file-organizer",
            {},
        )
        assert plugins.admitted == ["file-organizer"]

    async def test_the_worker_call_starts_only_once_a_slot_is_held(self):
        plugins = _RecordingPlugins()
        queue = pool(max_concurrent=1)
        dispatcher = PluginAwareDispatcher(_NoInference(), plugins, queue)
        released = asyncio.Event()

        async def blocker():
            await released.wait()

        held = asyncio.ensure_future(queue.run("busy", blocker))
        await asyncio.sleep(0)

        operation = asyncio.ensure_future(dispatcher.dispatch("job-1", "file-organizer", {}))
        await asyncio.sleep(0)
        assert plugins.dispatched == []

        released.set()
        await held
        await operation
        assert plugins.dispatched == [("job-1", "file-organizer")]


class TestPublishedStatus:
    def test_a_plugin_profile_reports_what_policy_does_to_it(self):
        policy = PolicyState(profiles={"file-organizer": ProfilePolicy(enabled=False, weight=3)})
        entry = plugin_profile_statuses(["file-organizer"], {}, policy)["file-organizer"]
        assert entry["status"] == "paused"
        assert entry["reason"] == "profile-disabled"

    def test_a_running_plugin_names_the_weight_its_queue_waits_at(self):
        policy = PolicyState(profiles={"file-organizer": ProfilePolicy(enabled=True, weight=4)})
        entry = plugin_profile_statuses(
            ["file-organizer"], {"file-organizer": {"queued": 2, "running": 1}}, policy
        )["file-organizer"]
        assert entry["status"] == "running"
        assert entry["queued"] == 2
        assert "weight 4" in entry["detail"]

    def test_an_idle_plugin_is_published_as_governed_rather_than_absent(self):
        entry = plugin_profile_statuses(["media-transcription"], {}, PolicyState())
        assert entry["media-transcription"]["status"] == "idle"

    def test_more_plugins_than_the_contract_publishes_are_dropped_not_invalid(self):
        statuses = plugin_profile_statuses(
            [f"plugin-{index:03d}" for index in range(200)], {}, PolicyState()
        )
        assert len(statuses) == 128


class _RecordingPlugins:
    def __init__(self) -> None:
        self.admitted: list[str] = []
        self.dispatched: list[tuple[str, str]] = []

    def plugin_ids(self) -> frozenset[str]:
        return frozenset({"file-organizer", "media-transcription"})

    def admit(self, workload_id: str, payload: dict) -> None:
        self.admitted.append(workload_id)

    async def dispatch(self, job_id: str, workload_id: str, payload: dict) -> dict:
        self.dispatched.append((job_id, workload_id))
        return {"ok": True}


class _NoInference:
    async def dispatch(self, job_id: str, workload_id: str, payload: dict) -> dict:
        raise AssertionError("inference dispatch is not part of the plugin pool")
