from __future__ import annotations

import asyncio
import json
import logging
import threading
import time

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.control import (
    CONTROL_VERSION,
    INTERNAL_ERROR_MESSAGE,
    PERSIST_FAILURE_MESSAGE,
    REVISION_MISMATCH_MESSAGE,
    ControlService,
    build_control_service,
)
from omnitensor.registry import validate_document
from omnitensor.state import (
    MAX_DEVICE_CHOICES,
    PolicyState,
    PolicyStore,
    ProfilePolicy,
)


@pytest.fixture
def control(tmp_path):
    defaults = {
        "hardware-health": ProfilePolicy(enabled=True, weight=2),
        "visual-library": ProfilePolicy(enabled=False, weight=3),
    }
    return build_control_service(tmp_path / "policy.json", defaults)


def command(operation, profile_id, value, revision=0, command_id="xpuwlm-1"):
    return json.dumps(
        {
            "version": CONTROL_VERSION,
            "id": command_id,
            "issuedAt": 1_700_000_000_000,
            "expectedRevision": revision,
            "operation": operation,
            "profileId": profile_id,
            "value": value,
        }
    )


def apply(control, text):
    """ApplyCommand persists off the event loop, so it is a coroutine."""
    acknowledgement = json.loads(asyncio.run(control.apply_command_text(text)))
    assert validate_document("runtime-acknowledgement.schema.json", acknowledgement) == []
    return acknowledgement


def test_apply_enable_and_weight_increment_revision_and_persist(control, tmp_path):
    first = apply(control, command("set-profile-enabled", "visual-library", True))
    assert first["status"] == "applied"
    assert first["message"] == "Policy applied"
    assert first["revision"] == 1
    assert first["portfolio"]["profiles"]["visual-library"]["enabled"] is True

    second = apply(control, command("set-profile-weight", "visual-library", 5, revision=1))
    assert second["revision"] == 2
    assert second["portfolio"]["profiles"]["visual-library"]["weight"] == 5

    reloaded = PolicyStore(
        tmp_path / "policy.json",
        {
            "hardware-health": ProfilePolicy(enabled=True, weight=2),
            "visual-library": ProfilePolicy(enabled=False, weight=3),
        },
    ).load()
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
    assert not_json["message"] == "Command is not valid JSON"

    not_object = apply(control, json.dumps(["array"]))
    assert not_object["status"] == "rejected"
    assert not_object["commandId"] == "invalid"
    assert not_object["message"] == "Command is not an object"

    bad_contract = apply(control, json.dumps({"version": 1, "id": "x-1"}))
    assert bad_contract["status"] == "rejected"
    assert bad_contract["commandId"] == "x-1"
    assert bad_contract["message"] == "Command does not match the version 2 contract"


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
    assert parameters == [
        "self",
        "store",
        "on_applied",
        "profile_exists",
        "gpu_device_ids",
        "profile_models",
    ]


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
            control.apply_command_text(command("set-profile-enabled", "visual-library", False))
        )
        while not started.is_set():
            await asyncio.sleep(0.001)
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

    acknowledgement = apply(control, command("set-profile-weight", "visual-library", value))

    assert acknowledgement["status"] == "applied"
    stored = json.loads(path.read_text())["profiles"]["visual-library"]["weight"]
    assert stored == int(value)
    assert isinstance(stored, int)
    assert PolicyStore(path, defaults).load().profiles["visual-library"].weight == int(value)


def test_a_boolean_is_never_accepted_as_a_weight(tmp_path):
    control = ControlService(
        PolicyStore(
            tmp_path / "policy.json",
            {
                "visual-library": ProfilePolicy(enabled=True, weight=1),
            },
        )
    )
    acknowledgement = apply(control, command("set-profile-weight", "visual-library", True))
    assert acknowledgement["status"] == "rejected"


def batch(changes, revision=0, command_id="xpuwlm-batch"):
    return json.dumps(
        {
            "version": CONTROL_VERSION,
            "id": command_id,
            "issuedAt": 1_700_000_000_000,
            "expectedRevision": revision,
            "operation": "apply-profiles",
            "profileId": None,
            "value": None,
            "changes": changes,
        }
    )


def test_a_batch_is_one_call_one_revision_and_one_acknowledgement(control):
    """The same settings sent one at a time spend a revision each."""
    acknowledgement = apply(
        control,
        batch(
            [
                {"profileId": "visual-library", "enabled": True, "weight": 5},
                {"profileId": "hardware-health", "enabled": False},
            ]
        ),
    )

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

    acknowledgement = apply(
        control,
        batch(
            [
                {"profileId": "visual-library", "weight": float(weight)},
            ]
        ),
    )

    acknowledged = acknowledgement["portfolio"]["profiles"]["visual-library"]["weight"]
    persisted = store.saved[0].profiles["visual-library"].weight
    assert acknowledged == weight
    assert persisted == weight
    assert isinstance(acknowledged, int)
    assert isinstance(persisted, int)


def test_a_batch_that_cannot_apply_whole_changes_nothing(control):
    """Half-applied policy with nothing to retry is the defect this removes."""
    before = json.loads(
        asyncio.run(
            control.apply_command_text(
                batch(
                    [
                        {"profileId": "visual-library", "enabled": True},
                    ]
                )
            )
        )
    )["portfolio"]

    acknowledgement = apply(
        control,
        batch(
            [
                {"profileId": "visual-library", "enabled": False, "weight": 1},
                {"profileId": "no-such-profile", "enabled": True},
            ],
            revision=1,
            command_id="xpuwlm-batch-2",
        ),
    )

    assert acknowledgement["status"] == "rejected"
    assert "no-such-profile" in acknowledgement["message"]
    assert acknowledgement["revision"] == 1, "a refused batch spends no revision"
    assert acknowledgement["portfolio"] == before, "the earlier change is untouched"


def test_a_batch_that_changes_nothing_is_named_rather_than_accepted(control):
    acknowledgement = apply(control, batch([{"profileId": "visual-library"}]))

    assert acknowledgement["status"] == "rejected"
    assert "none of enabled, weight, deviceId or modelId" in acknowledgement["message"]
    assert acknowledgement["revision"] == 0


def test_the_batch_operation_is_bounded_by_the_same_contract_as_the_rest(control):
    """The schema is the gate; the service never sees a malformed batch."""
    for changes in ([], [{"profileId": "visual-library", "weight": 9}], [{"weight": 1}]):
        acknowledgement = apply(control, batch(changes))
        assert acknowledgement["status"] == "rejected"
        assert acknowledgement["revision"] == 0

    assert (
        validate_document(
            "runtime-command.schema.json",
            json.loads(
                batch(
                    [
                        {"profileId": "visual-library", "enabled": True},
                    ]
                )
            ),
        )
        == []
    )


def selectable_control(store=None):
    return ControlService(
        store or MemoryStore(),
        profile_exists=lambda profile_id: profile_id == "external-plugin",
        gpu_device_ids=lambda: ("gpu-renderD128", "gpu-renderD129"),
    )


def test_device_choice_is_validated_persisted_acknowledged_and_cleared():
    store = MemoryStore()
    control = selectable_control(store)

    selected = apply(
        control,
        command("set-profile-device", "visual-library", "gpu-renderD129"),
    )
    assert selected["status"] == "applied"
    assert selected["portfolio"]["deviceChoices"] == {"visual-library": "gpu-renderD129"}
    assert store.saved[-1].device_choices == {"visual-library": "gpu-renderD129"}

    cleared = apply(
        control,
        command("set-profile-device", "visual-library", None, revision=1),
    )
    assert cleared["status"] == "applied"
    assert cleared["portfolio"]["deviceChoices"] == {}


def test_device_choice_refuses_unknown_profile_or_unavailable_gpu_atomically():
    control = selectable_control()

    missing_profile = apply(
        control,
        command("set-profile-device", "missing", "gpu-renderD128"),
    )
    missing_gpu = apply(
        control,
        command("set-profile-device", "visual-library", "gpu-renderD999"),
    )

    assert missing_profile["status"] == "rejected"
    assert missing_profile["message"] == "Unknown workload profile: missing"
    assert missing_gpu["status"] == "rejected"
    assert missing_gpu["message"] == "GPU device is unavailable: gpu-renderD999"
    assert control.state.device_choices == {}
    assert control.state.revision == 0


@pytest.mark.parametrize(
    "device_id",
    ["gpu-card0", "gpu-renderD", "gpu-renderD1234567", 128, True, {}],
)
def test_device_choice_wire_identity_is_closed_before_policy_lookup(device_id):
    control = selectable_control()

    acknowledgement = apply(
        control,
        command("set-profile-device", "visual-library", device_id),
    )

    assert acknowledgement["status"] == "rejected"
    assert acknowledgement["message"] == "Command does not match the version 2 contract"
    assert control.state.device_choices == {}


def test_device_choice_count_bound_rejects_new_profile_but_allows_replace_and_clear():
    store = MemoryStore()
    control = ControlService(
        store,
        profile_exists=lambda _profile_id: True,
        gpu_device_ids=lambda: ("gpu-renderD128", "gpu-renderD129"),
    )
    control.state.device_choices.update(
        {f"external-{index:03d}": "gpu-renderD128" for index in range(MAX_DEVICE_CHOICES)}
    )

    rejected = apply(
        control,
        command("set-profile-device", "overflow", "gpu-renderD128"),
    )
    assert rejected["status"] == "rejected"
    assert rejected["message"] == "GPU device choices are limited to 128 profiles"
    assert rejected["revision"] == 0

    replaced = apply(
        control,
        command("set-profile-device", "external-000", "gpu-renderD129"),
    )
    assert replaced["status"] == "applied"
    assert replaced["portfolio"]["deviceChoices"]["external-000"] == "gpu-renderD129"

    cleared = apply(
        control,
        command("set-profile-device", "external-001", None, revision=1),
    )
    assert cleared["status"] == "applied"
    assert len(cleared["portfolio"]["deviceChoices"]) == MAX_DEVICE_CHOICES - 1


def test_external_plugin_device_choice_and_batch_use_the_same_contract():
    control = selectable_control()

    selected = apply(
        control,
        command("set-profile-device", "external-plugin", "gpu-renderD128"),
    )
    assert selected["portfolio"]["deviceChoices"] == {"external-plugin": "gpu-renderD128"}

    changed = apply(
        control,
        batch(
            [
                {"profileId": "visual-library", "weight": 4},
                {"profileId": "external-plugin", "deviceId": "gpu-renderD129"},
            ],
            revision=1,
        ),
    )
    assert changed["status"] == "applied"
    assert changed["portfolio"]["deviceChoices"] == {"external-plugin": "gpu-renderD129"}


def test_a_batch_still_obeys_the_revision_compare_and_swap(control):
    apply(control, command("set-paused", None, True))

    stale = apply(
        control,
        batch(
            [{"profileId": "visual-library", "enabled": True}],
            revision=0,
            command_id="xpuwlm-stale",
        ),
    )

    assert stale["status"] == "rejected"
    assert "revision" in stale["message"].lower()


def test_adopting_a_plugin_profile_makes_its_policy_settable(control):
    """The refusal F4 named: a plugin has no policy until the runtime adopts it."""
    refusal = apply(control, command("set-profile-enabled", "file-organizer", False))
    assert refusal["status"] == "rejected"
    assert refusal["message"] == "Unknown workload profile: file-organizer"

    adopted = asyncio.run(control.adopt_profiles(["file-organizer"]))

    assert adopted == ("file-organizer",)
    assert control.state.profiles["file-organizer"] == ProfilePolicy(enabled=True, weight=1)
    applied = apply(
        control,
        command("set-profile-enabled", "file-organizer", False, revision=control.state.revision),
    )
    assert applied["status"] == "applied"
    assert applied["portfolio"]["profiles"]["file-organizer"]["enabled"] is False


def test_adoption_never_overwrites_a_decision_already_made(control):
    asyncio.run(control.adopt_profiles(["file-organizer"]))
    apply(
        control,
        command("set-profile-weight", "file-organizer", 4, revision=control.state.revision),
    )

    again = asyncio.run(control.adopt_profiles(["file-organizer", "media-transcription"]))

    assert again == ("media-transcription",)
    assert control.state.profiles["file-organizer"].weight == 4


def test_adoption_is_persisted_and_counted_in_the_revision(control, tmp_path):
    before = control.state.revision

    asyncio.run(control.adopt_profiles(["file-organizer"]))

    assert control.state.revision == before + 1
    stored = json.loads((tmp_path / "policy.json").read_text(encoding="utf-8"))
    assert stored["profiles"]["file-organizer"] == {"enabled": True, "weight": 1}


def test_adopting_nothing_new_spends_no_revision(control):
    asyncio.run(control.adopt_profiles(["file-organizer"]))
    revision = control.state.revision

    assert asyncio.run(control.adopt_profiles(["file-organizer"])) == ()
    assert control.state.revision == revision


def test_unpersistable_adoption_is_logged_rather_than_silent(control, monkeypatch, caplog):
    def refuse(_state):
        raise OSError("read-only file system")

    monkeypatch.setattr(control._store, "save", refuse)

    with caplog.at_level(logging.WARNING, logger="omnitensor.control"):
        assert asyncio.run(control.adopt_profiles(["file-organizer"])) == ()

    assert "file-organizer" not in control.state.profiles
    assert any("file-organizer" in record.getMessage() for record in caplog.records)
