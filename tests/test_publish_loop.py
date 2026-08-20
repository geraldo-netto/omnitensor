"""OMNI-0515: the publish policy, on its own, without a service around it.

Every one of these was reachable only through `OmniTensorService` before, which
is the reason for the extraction: deciding whether a document is worth writing
is not a fact about being a service.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from omnitensor.publish_loop import SnapshotPublishLoop


class Publisher:
    def __init__(self, fail=None):
        self.published = []
        self.retracted = 0
        self.fail = fail

    def publish(self, snapshot):
        if self.fail is not None:
            raise self.fail
        self.published.append(snapshot)

    def retract(self):
        self.retracted += 1


def loop_for(publisher, *, devices=True, build=None, **kwargs):
    counter = {"n": 0}

    def default_build(_view):
        counter["n"] += 1
        return {"generatedAt": counter["n"], "devices": ["tpu"]}

    made = SnapshotPublishLoop(
        build or default_build,
        publisher,
        stopping=kwargs.pop("stopping", None) or asyncio.Event(),
        interval_s=kwargs.pop("interval_s", 0.001),
        observe=lambda: None,
        devices_present=lambda: devices,
        **kwargs,
    )
    made.builds = counter
    return made


def test_a_document_that_says_nothing_new_is_not_a_change():
    """`generatedAt` moves on every build and is not a change in its own right."""
    subject = loop_for(Publisher())

    first = {"generatedAt": 1, "devices": ["tpu"]}
    assert subject.changed(first) is True
    subject.record_published(first)
    assert subject.changed({"generatedAt": 2, "devices": ["tpu"]}) is False
    assert subject.changed({"generatedAt": 2, "devices": ["tpu", "npu"]}) is True


def test_publish_once_runs_the_builder_directly_when_no_off_loop_is_given():
    """The service hands it `run_off_loop`; anything else may not have one."""
    publisher = Publisher()
    subject = loop_for(publisher)

    published = asyncio.run(_one_tick(subject))

    assert publisher.published == [published]
    assert subject.idle is True  # nothing changed on the second tick


async def _one_tick(subject):
    published = subject.publish_once()
    await subject._publish_tick()
    return published


def test_a_publish_failure_is_an_outage_of_one_output_not_of_the_service(caplog):
    """An OSError escaping here takes down job dispatch and D-Bus with it."""
    publisher = Publisher(fail=OSError("no space left on device"))
    stopping = asyncio.Event()
    subject = loop_for(publisher, stopping=stopping, logger=logging.getLogger("publish-test"))

    async def scenario():
        subject.idle = True
        task = asyncio.get_running_loop().create_task(subject.run())
        await asyncio.sleep(0.02)
        stopping.set()
        subject.wake.set()
        await asyncio.wait_for(task, timeout=1)

    with caplog.at_level("ERROR", logger="publish-test"):
        asyncio.run(scenario())

    assert publisher.published == []
    # A failed tick is not an idle tick: backing off after one would delay the
    # retry by the idle interval.
    assert subject.idle is False
    assert "retrying next tick" in caplog.text


def test_a_wake_request_before_a_loop_is_bound_is_dropped_rather_than_raising():
    subject = loop_for(Publisher())

    subject.request_publish()
    assert subject.wake.is_set() is False

    class ClosedLoop:
        def call_soon_threadsafe(self, _callback):
            raise RuntimeError("Event loop is closed")

    subject.bind(ClosedLoop())
    subject.request_publish()
    assert subject.wake.is_set() is False


def test_a_wake_request_is_not_repeated_while_one_is_already_pending():
    subject = loop_for(Publisher())
    calls = []

    class Loop:
        def call_soon_threadsafe(self, callback):
            calls.append(callback)
            callback()

    subject.bind(Loop())
    subject.request_publish()
    subject.request_publish()

    assert len(calls) == 1
    assert subject.wake.is_set() is True


def test_no_devices_retracts_once_and_forgets_what_was_published():
    """With zero devices no schema-valid snapshot exists, so absence is honest."""
    publisher = Publisher()
    stopping = asyncio.Event()
    subject = loop_for(publisher, devices=False, stopping=stopping)
    subject.record_published({"generatedAt": 1, "devices": ["tpu"]})

    async def scenario():
        task = asyncio.get_running_loop().create_task(subject.run())
        await asyncio.sleep(0.02)
        stopping.set()
        subject.wake.set()
        await asyncio.wait_for(task, timeout=1)

    asyncio.run(scenario())

    assert publisher.retracted == 1
    assert subject.retracted is True
    assert subject.last_published is None
    assert subject.next_heartbeat == 0.0


def test_an_idle_loop_waits_the_idle_interval_and_a_request_ends_the_wait():
    subject = loop_for(Publisher(), interval_s=30.0, idle_interval_s=30.0)

    async def scenario():
        subject.bind(asyncio.get_running_loop())
        subject.idle = True
        waiting = asyncio.get_running_loop().create_task(subject.await_next())
        await asyncio.sleep(0)
        subject.request_publish()
        await asyncio.wait_for(waiting, timeout=1)
        # Cleared on the way out, so the next wait is a real wait.
        assert subject.wake.is_set() is False

    asyncio.run(scenario())


def test_the_heartbeat_is_due_only_once_the_interval_has_passed():
    subject = loop_for(Publisher(), heartbeat_interval_s=30.0)

    assert subject.heartbeat_due() is True
    subject.record_published({"generatedAt": 1, "devices": ["tpu"]})
    assert subject.heartbeat_due() is False


@pytest.mark.asyncio
async def test_the_off_loop_hop_is_used_for_every_call_when_one_is_supplied():
    hops = []

    async def off_loop(call, *arguments):
        hops.append(call)
        return call(*arguments)

    publisher = Publisher()
    subject = loop_for(publisher, off_loop=off_loop)

    await subject._publish_tick()

    assert hops == [subject._build, publisher.publish]
