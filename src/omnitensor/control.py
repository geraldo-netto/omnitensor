"""Runtime command handling with optimistic concurrency.

Transport-neutral: :class:`ControlService` consumes command JSON text and
returns acknowledgement JSON text, both validated against the canonical
schemas.  The D-Bus wiring in :mod:`omnitensor.service` stays a thin shim
around this class so the policy logic is fully testable off the bus.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from .ports import PolicyStorage
from .registry import validate_document
from .state import MAX_WEIGHT, MIN_WEIGHT, PolicyState, PolicyStore, ProfilePolicy

CONTROL_VERSION = 1
REVISION_MISMATCH_MESSAGE = "Runtime policy revision changed; refresh and retry"


def _now_ms() -> int:
    return max(1, int(time.time() * 1000))


class ControlService:
    def __init__(self, store: PolicyStorage, defaults: dict[str, ProfilePolicy]):
        self._store = store
        self._defaults = defaults
        self._state: PolicyState = store.load()

    @property
    def state(self) -> PolicyState:
        return self._state

    def apply_command_text(self, text: str) -> str:
        acknowledgement = self._apply(text)
        violations = validate_document("runtime-acknowledgement.schema.json", acknowledgement)
        if violations:  # pragma: no cover - contract self-check, never expected
            raise RuntimeError(f"acknowledgement violates contract: {violations}")
        return json.dumps(acknowledgement, separators=(",", ":"))

    def _apply(self, text: str) -> dict:
        try:
            command = json.loads(text)
        except ValueError:
            return self._rejection("invalid", "Command is not valid JSON")
        if not isinstance(command, dict):
            return self._rejection("invalid", "Command is not an object")
        command_id = command.get("id")
        command_id = command_id if isinstance(command_id, str) and command_id else "invalid"
        violations = validate_document("runtime-command.schema.json", command)
        if violations:
            return self._rejection(command_id, "Command does not match the version 1 contract")
        if command["expectedRevision"] != self._state.revision:
            return self._rejection(command_id, REVISION_MISMATCH_MESSAGE)
        message = self._execute(command)
        if message is not None:
            return self._rejection(command_id, message)
        self._state.revision += 1
        self._store.save(self._state)
        return self._acknowledgement(command_id, "applied", "Policy applied")

    def _execute(self, command: dict) -> str | None:
        operation = command["operation"]
        if operation == "set-paused":
            self._state.paused = command["value"] is True
            return None
        profile_id = command["profileId"]
        policy = self._state.profiles.get(profile_id)
        if policy is None:
            return f"Unknown workload profile: {profile_id}"
        if operation == "set-profile-enabled":
            policy.enabled = command["value"] is True
            return None
        weight = command["value"]
        if not MIN_WEIGHT <= weight <= MAX_WEIGHT:
            return f"Weight must be between {MIN_WEIGHT} and {MAX_WEIGHT}"
        policy.weight = weight
        return None

    def _rejection(self, command_id: str, message: str) -> dict:
        return self._acknowledgement(command_id, "rejected", message)

    def _acknowledgement(self, command_id: str, status: str, message: str) -> dict:
        return {
            "version": CONTROL_VERSION,
            "commandId": command_id,
            "status": status,
            "revision": self._state.revision,
            "appliedAt": _now_ms(),
            "message": message,
            "portfolio": self._state.portfolio(),
        }


def build_control_service(state_path: Path, defaults: dict[str, ProfilePolicy]) -> ControlService:
    return ControlService(PolicyStore(state_path, defaults), defaults)
