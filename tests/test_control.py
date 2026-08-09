from __future__ import annotations

import json

import pytest

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


def command(operation, profile_id, value, revision=0, command_id="tpuwm-1"):
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
    acknowledgement = json.loads(control.apply_command_text(text))
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
