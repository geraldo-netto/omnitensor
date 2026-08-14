from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.control import (
    INTERNAL_ERROR_MESSAGE,
    PERSIST_FAILURE_MESSAGE,
    REVISION_MISMATCH_MESSAGE,
    ControlService,
    build_control_service,
)
from omnitensor.registry import validate_document
from omnitensor.state import PolicyState, PolicyStore, ProfilePolicy


@pytest.fixture
def control(tmp_path):
    defaults = {
        "hardware-health": ProfilePolicy(enabled=True, weight=2),
        "visual-library": ProfilePolicy(enabled=False, weight=3),
    }
    return build_control_service(tmp_path / "policy.json", defaults)


def command(operation, profile_id, value, revision=0, command_id="xpuwlm-1"):
    return json.dumps({
        "version": 1,
        "id": command_id,
        "issuedAt": 1_700_000_000_000,
        "expectedRevision": revision,
        "operation": operation,
        "profileId": profile_id,
        "value": value,
    })


def apply(control, text):
    """ApplyCommand persists off the event loop, so it is a coroutine."""
    acknowledgement = json.loads(asyncio.run(control.apply_command_text(text)))
    assert validate_document("runtime-acknowledgement.schema.json", acknowledgement) == []
    return acknowledgement


def test_apply_enable_and_weight_increment_revision_and_persist(control, tmp_path):
    first = apply(control, command("set-profile-enabled", "visual-library", True))
    assert first["status"] == "applied"
    assert first["revision"] == 1
    assert first["portfolio"]["profiles"]["visual-library"]["enabled"] is True

    second = apply(control, command("set-profile-weight", "visual-library", 5, revision=1))
    assert second["revision"] == 2
    assert second["portfolio"]["profiles"]["visual-library"]["weight"] == 5

    reloaded = PolicyStore(tmp_path / "policy.json", {
        "hardware-health": ProfilePolicy(enabled=True, weight=2),
        "visual-library": ProfilePolicy(enabled=False, weight=3),
    }).load()
    assert reloaded.revision == 2
    assert reloaded.profiles["visual-library"].weight == 5


def test_revision_mismatch_rejects_without_applying(control):
    stale = apply(control, command("set-paused", None, True, revision=7))
    assert stale["status"] == "rejected"
    assert stale["message"] == REVISION_MISMATCH_MESSAGE
    assert stale["revision"] == 0
    assert control.state.paused is False


def test_pause_toggles_globally(control):
    acknowledgement = apply(control, command("set-paused", None, True))
    assert acknowledgement["status"] == "applied"
    assert acknowledgement["portfolio"]["paused"] is True


def test_unknown_profile_and_malformed_commands_reject(control):
    unknown = apply(control, command("set-profile-enabled", "missing", True))
    assert unknown["status"] == "rejected"
    assert "Unknown workload profile" in unknown["message"]

    not_json = apply(control, "{nope")
    assert not_json["status"] == "rejected"
    assert not_json["commandId"] == "invalid"

    not_object = apply(control, json.dumps(["array"]))
    assert not_object["status"] == "rejected"

    bad_contract = apply(control, json.dumps({"version": 2, "id": "x-1"}))
    assert bad_contract["status"] == "rejected"
    assert bad_contract["commandId"] == "x-1"
    assert "version 1 contract" in bad_contract["message"]


def test_rejection_leaves_revision_untouched(control):
    apply(control, command("set-profile-enabled", "missing", True))
    assert control.state.revision == 0
    applied = apply(control, command("set-paused", None, True))
    assert applied["revision"] == 1


@pytest.mark.parametrize("bad_id", ["bad id with spaces", "", "a" * 200, 7, None, "tab\tchar"])
def test_unechoable_command_ids_never_break_the_acknowledgement(control, bad_id):
    acknowledgement = apply(control, json.dumps({"id": bad_id}))
    assert acknowledgement["status"] == "rejected"
    assert acknowledgement["commandId"] == "invalid"


class FailingStorage:
    """PolicyStorage whose save always fails."""

    def __init__(self, error):
        self._error = error

    def load(self):
        return PolicyState(profiles={"visual-library": ProfilePolicy(enabled=False, weight=3)})

    def save(self, state):
        raise self._error


def test_save_failure_rejects_and_keeps_served_state_consistent():
    control = ControlService(FailingStorage(OSError("disk full")))
    acknowledgement = apply(control, command("set-profile-enabled", "visual-library", True))
    assert acknowledgement["status"] == "rejected"
    assert acknowledgement["message"] == PERSIST_FAILURE_MESSAGE
    assert acknowledgement["revision"] == 0
    assert control.state.revision == 0
    assert control.state.profiles["visual-library"].enabled is False


def test_unexpected_handler_failure_returns_internal_error_rejection():
    control = ControlService(FailingStorage(RuntimeError("boom")))
    acknowledgement = apply(control, command("set-paused", None, True))
    assert acknowledgement["status"] == "rejected"
    assert acknowledgement["message"] == INTERNAL_ERROR_MESSAGE
    assert control.state.paused is False


class MemoryStore:
    def __init__(self):
        self.saved = []

    def load(self):
        return PolicyState(profiles={"visual-library": ProfilePolicy(enabled=True, weight=3)})

    def save(self, state):
        self.saved.append(state)


def test_on_applied_listener_fires_only_for_applied_commands():
    calls = []
    control = ControlService(MemoryStore(), on_applied=lambda: calls.append(True))
    applied = apply(control, command("set-paused", None, True))
    assert applied["status"] == "applied"
    assert calls == [True]

    rejected = apply(control, command("set-paused", None, True, revision=99))
    assert rejected["status"] == "rejected"
    assert calls == [True]

    invalid = apply(control, "not json")
    assert invalid["status"] == "rejected"
    assert calls == [True]


def test_raising_on_applied_listener_never_breaks_the_acknowledgement():
    def explode():
        raise RuntimeError("listener bug")

    control = ControlService(MemoryStore(), on_applied=explode)
    acknowledgement = apply(control, command("set-paused", None, True))
    assert acknowledgement["status"] == "applied"
    assert control.state.paused is True


def test_control_service_holds_no_dead_defaults_state():
    """Regression (OMNI-0022): defaults belong to PolicyStore; ControlService
    takes only the storage port and the on_applied listener."""
    import inspect

    parameters = list(inspect.signature(ControlService.__init__).parameters)
    assert parameters == ["self", "store", "on_applied"]


def test_policy_persistence_does_not_block_the_event_loop(tmp_path):
    """A save is a write plus two fsyncs; on the loop it stalls every other task."""
    started = threading.Event()
    release = threading.Event()

    class BlockingStore:
        def __init__(self):
            self.saved = []

        def load(self):
            return PolicyState(profiles={"visual-library": ProfilePolicy(True, 1)})

        def save(self, state):
            started.set()
            assert release.wait(timeout=5)
            self.saved.append(state)

    async def scenario():
        control = ControlService(BlockingStore())
        applying = asyncio.create_task(
            control.apply_command_text(
                command("set-profile-enabled", "visual-library", False)
            )
        )
        await asyncio.to_thread(started.wait, 5)
        # The loop is still live while the store blocks in its own thread.
        ticks = 0
        for _ in range(3):
            await asyncio.sleep(0)
            ticks += 1
        release.set()
        return ticks, json.loads(await applying)

    ticks, acknowledgement = asyncio.run(asyncio.wait_for(scenario(), timeout=10))

    assert ticks == 3, "the event loop was blocked by the policy save"
    assert acknowledgement["status"] == "applied"


def test_concurrent_commands_cannot_both_commit_the_same_revision(tmp_path):
    """Persisting off the loop opens a window the revision check must close."""

    class SlowStore:
        def __init__(self):
            self.saved = []

        def load(self):
            return PolicyState(profiles={"visual-library": ProfilePolicy(True, 1)})

        def save(self, state):
            time.sleep(0.05)
            self.saved.append(state)

    store = SlowStore()

    async def scenario():
        control = ControlService(store)
        first, second = await asyncio.gather(
            control.apply_command_text(
                command("set-profile-enabled", "visual-library", False, 0, "cmd-a")
            ),
            control.apply_command_text(
                command("set-profile-weight", "visual-library", 4, 0, "cmd-b")
            ),
        )
        return json.loads(first), json.loads(second)

    first, second = asyncio.run(asyncio.wait_for(scenario(), timeout=10))

    statuses = sorted((first["status"], second["status"]))
    assert statuses == ["applied", "rejected"]
    assert len(store.saved) == 1
    assert store.saved[0].revision == 1


@pytest.mark.parametrize("value", [2.0, 5.0])
def test_an_integral_float_weight_is_stored_as_an_integer(tmp_path, value):
    """Draft 2020-12 accepts 2.0 as an integer; stored as a float the loader
    would silently reset the profile to its default on the next start."""
    path = tmp_path / "policy.json"
    defaults = {"visual-library": ProfilePolicy(enabled=True, weight=1)}
    control = ControlService(PolicyStore(path, defaults))

    acknowledgement = apply(
        control, command("set-profile-weight", "visual-library", value)
    )

    assert acknowledgement["status"] == "applied"
    stored = json.loads(path.read_text())["profiles"]["visual-library"]["weight"]
    assert stored == int(value)
    assert isinstance(stored, int)
    assert PolicyStore(path, defaults).load().profiles["visual-library"].weight == int(value)


def test_a_boolean_is_never_accepted_as_a_weight(tmp_path):
    control = ControlService(PolicyStore(tmp_path / "policy.json", {
        "visual-library": ProfilePolicy(enabled=True, weight=1),
    }))
    acknowledgement = apply(control, command("set-profile-weight", "visual-library", True))
    assert acknowledgement["status"] == "rejected"


def batch(changes, revision=0, command_id="xpuwlm-batch"):
    return json.dumps({
        "version": 1,
        "id": command_id,
        "issuedAt": 1_700_000_000_000,
        "expectedRevision": revision,
        "operation": "apply-profiles",
        "profileId": None,
        "value": None,
        "changes": changes,
    })


def test_a_batch_is_one_call_one_revision_and_one_acknowledgement(control):
    """The same settings sent one at a time spend a revision each."""
    acknowledgement = apply(control, batch([
        {"profileId": "visual-library", "enabled": True, "weight": 5},
        {"profileId": "hardware-health", "enabled": False},
    ]))

    assert acknowledgement["status"] == "applied"
    assert acknowledgement["revision"] == 1
    portfolio = acknowledgement["portfolio"]["profiles"]
    assert portfolio["visual-library"] == {"enabled": True, "weight": 5}
    assert portfolio["hardware-health"]["enabled"] is False
    assert portfolio["hardware-health"]["weight"] == 2, "what it did not name is untouched"


@given(st.integers(min_value=1, max_value=5))
def test_a_batch_persists_schema_integer_floats_as_integers(weight):
    store = MemoryStore()
    control = ControlService(store)

    acknowledgement = apply(control, batch([
        {"profileId": "visual-library", "weight": float(weight)},
    ]))

    acknowledged = acknowledgement["portfolio"]["profiles"]["visual-library"]["weight"]
    persisted = store.saved[0].profiles["visual-library"].weight
    assert acknowledged == weight
    assert persisted == weight
    assert isinstance(acknowledged, int)
    assert isinstance(persisted, int)


def test_a_batch_that_cannot_apply_whole_changes_nothing(control):
    """Half-applied policy with nothing to retry is the defect this removes."""
    before = json.loads(asyncio.run(control.apply_command_text(batch([
        {"profileId": "visual-library", "enabled": True},
    ]))))["portfolio"]

    acknowledgement = apply(control, batch(
        [
            {"profileId": "visual-library", "enabled": False, "weight": 1},
            {"profileId": "no-such-profile", "enabled": True},
        ],
        revision=1,
        command_id="xpuwlm-batch-2",
    ))

    assert acknowledgement["status"] == "rejected"
    assert "no-such-profile" in acknowledgement["message"]
    assert acknowledgement["revision"] == 1, "a refused batch spends no revision"
    assert acknowledgement["portfolio"] == before, "the earlier change is untouched"


def test_a_batch_that_changes_nothing_is_named_rather_than_accepted(control):
    acknowledgement = apply(control, batch([{"profileId": "visual-library"}]))

    assert acknowledgement["status"] == "rejected"
    assert "neither enabled nor weight" in acknowledgement["message"]
    assert acknowledgement["revision"] == 0


def test_the_batch_operation_is_bounded_by_the_same_contract_as_the_rest(control):
    """The schema is the gate; the service never sees a malformed batch."""
    for changes in ([], [{"profileId": "visual-library", "weight": 9}], [{"weight": 1}]):
        acknowledgement = apply(control, batch(changes))
        assert acknowledgement["status"] == "rejected"
        assert acknowledgement["revision"] == 0

    assert validate_document("runtime-command.schema.json", json.loads(batch([
        {"profileId": "visual-library", "enabled": True},
    ]))) == []


def test_a_batch_still_obeys_the_revision_compare_and_swap(control):
    apply(control, command("set-paused", None, True))

    stale = apply(control, batch(
        [{"profileId": "visual-library", "enabled": True}], revision=0, command_id="xpuwlm-stale"
    ))

    assert stale["status"] == "rejected"
    assert "revision" in stale["message"].lower()
