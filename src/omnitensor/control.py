"""Runtime command handling with optimistic concurrency.

Transport-neutral: :class:`ControlService` consumes command JSON text and
returns acknowledgement JSON text, both validated against the canonical
schemas.  The D-Bus wiring in :mod:`omnitensor.service` stays a thin shim
around this class so the policy logic is fully testable off the bus.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import json
import re
import time
from pathlib import Path

from .ports import PolicyStorage
from .registry import validate_document
from .state import MAX_WEIGHT, MIN_WEIGHT, PolicyState, PolicyStore, ProfilePolicy

CONTROL_VERSION = 1
REVISION_MISMATCH_MESSAGE = "Runtime policy revision changed; refresh and retry"
PERSIST_FAILURE_MESSAGE = "Could not persist the policy; nothing was applied"
INTERNAL_ERROR_MESSAGE = "Internal error while applying the command"
COMMAND_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,120}$")
MAX_MESSAGE_LENGTH = 240


def _now_ms() -> int:
    return max(1, int(time.time() * 1000))


def _sanitized_command_id(command: dict) -> str:
    """The command's ``id`` when it is echoable inside a contract-valid
    acknowledgement, ``"invalid"`` otherwise."""
    candidate = command.get("id")
    if isinstance(candidate, str) and COMMAND_ID_PATTERN.fullmatch(candidate):
        return candidate
    return "invalid"


class ControlService:
    def __init__(self, store: PolicyStorage, on_applied=None):
        self._store = store
        self._on_applied = on_applied
        self._state: PolicyState = store.load()
        self._lock = asyncio.Lock()

    @property
    def state(self) -> PolicyState:
        return self._state

    async def apply_command_text(self, text: str) -> str:
        try:
            acknowledgement = await self._apply(text)
        except Exception:  # noqa: BLE001 - transport must always get a contract-valid reply
            acknowledgement = self._rejection("invalid", INTERNAL_ERROR_MESSAGE)
        violations = validate_document("runtime-acknowledgement.schema.json", acknowledgement)
        if violations:  # pragma: no cover - contract self-check, never expected
            raise RuntimeError(f"acknowledgement violates contract: {violations}")
        return json.dumps(acknowledgement, separators=(",", ":"))

    async def _apply(self, text: str) -> dict:
        try:
            command = json.loads(text)
        except ValueError:
            return self._rejection("invalid", "Command is not valid JSON")
        if not isinstance(command, dict):
            return self._rejection("invalid", "Command is not an object")
        command_id = _sanitized_command_id(command)
        violations = validate_document("runtime-command.schema.json", command)
        if violations:
            return self._rejection(command_id, "Command does not match the version 1 contract")
        # The lock spans the revision check, the persist, and the publish.
        # Persisting off the loop means another command can run between the
        # check and the commit, and two commands that both saw revision N
        # would both commit N+1 — the compare-and-swap the contract promises
        # would silently stop holding.
        async with self._lock:
            if command["expectedRevision"] != self._state.revision:
                return self._rejection(command_id, REVISION_MISMATCH_MESSAGE)
            # Commit protocol: mutate a copy, persist it, only then publish it
            # in memory — a failed save leaves the served state untouched.
            candidate = copy.deepcopy(self._state)
            message = self._execute(candidate, command)
            if message is not None:
                return self._rejection(command_id, message)
            candidate.revision += 1
            try:
                # A policy save is a write plus a file and a directory fsync.
                # On the event loop that stalls snapshot publishing and job
                # dispatch for as long as the disk takes.
                await asyncio.to_thread(self._store.save, candidate)
            except OSError:
                return self._rejection(command_id, PERSIST_FAILURE_MESSAGE)
            self._state = candidate
        if self._on_applied is not None:
            # The listener is a runtime nudge (e.g. wake scheduler workers);
            # it must never break the acknowledgement contract for an already
            # committed command.
            with contextlib.suppress(Exception):
                self._on_applied()
        return self._acknowledgement(command_id, "applied", "Policy applied")

    def _execute(self, state: PolicyState, command: dict) -> str | None:
        operation = command["operation"]
        if operation == "set-paused":
            state.paused = command["value"] is True
            return None
        profile_id = command["profileId"]
        policy = state.profiles.get(profile_id)
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
            "message": message[:MAX_MESSAGE_LENGTH],
            "portfolio": self._state.portfolio(),
        }


def build_control_service(state_path: Path, defaults: dict[str, ProfilePolicy]) -> ControlService:
    return ControlService(PolicyStore(state_path, defaults))
