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

from .plugins.offloop import run_off_loop
from .ports import PolicyStorage
from .registry import validate_document
from .state import (
    MAX_DEVICE_CHOICES,
    MAX_MODEL_CHOICES,
    MAX_WEIGHT,
    MIN_WEIGHT,
    PolicyState,
    PolicyStore,
    ProfilePolicy,
)

CONTROL_VERSION = 2
REVISION_MISMATCH_MESSAGE = "Runtime policy revision changed; refresh and retry"
PERSIST_FAILURE_MESSAGE = "Could not persist the policy; nothing was applied"
INTERNAL_ERROR_MESSAGE = "Internal error while applying the command"
COMMAND_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,120}$")
MAX_MESSAGE_LENGTH = 240


def _now_ms() -> int:
    return max(1, int(time.time() * 1000))


_CHOOSERS = {
    "set-profile-device": lambda control, state, profile_id, value: (
        control._set_profile_device(state, profile_id, value)
    ),
    "set-profile-model": lambda control, state, profile_id, value: (
        control._set_profile_model(state, profile_id, value)
    ),
}


def _integral_weight(value: object) -> int | None:
    """Coerce a schema-valid weight to the ``int`` the policy stores.

    Draft 2020-12 counts ``2.0`` as an integer, so a float reaches this code
    having passed validation.  Assigned as a float it persists as ``2.0``,
    which the loader's integer check then rejects — silently resetting the
    profile to its default the next time the service starts.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _sanitized_command_id(command: dict) -> str:
    """The command's ``id`` when it is echoable inside a contract-valid
    acknowledgement, ``"invalid"`` otherwise."""
    candidate = command.get("id")
    if isinstance(candidate, str) and COMMAND_ID_PATTERN.fullmatch(candidate):
        return candidate
    return "invalid"


class ControlService:
    def __init__(
        self,
        store: PolicyStorage,
        on_applied=None,
        *,
        profile_exists=None,
        gpu_device_ids=None,
        profile_models=None,
    ):
        self._store = store
        self._on_applied = on_applied
        self._profile_exists = profile_exists
        self._gpu_device_ids = gpu_device_ids or (lambda: ())
        # Which models a profile may be given, from the artifacts its manifest
        # pins. A model no manifest declares has no verified digest here, so
        # choosing it could only ever end at a worker that refuses to start.
        self._profile_models = profile_models or (lambda _profile_id: ())
        self._state: PolicyState = store.load()
        self._lock = asyncio.Lock()

    @property
    def state(self) -> PolicyState:
        return self._state

    async def adopt_profiles(
        self, profile_ids, default: ProfilePolicy | None = None
    ) -> tuple[str, ...]:
        """Give governed profiles a policy the stored state does not hold yet.

        Policy is seeded from the workload catalog, which is complete before
        the first command arrives.  Plugins are not in it: they are discovered
        after the store is loaded, so every ``set-profile-enabled`` naming one
        was answered ``Unknown workload profile`` and the surfaces that read
        the published profile set drew three controls that could not work.

        Adoption is a policy change like any other — persisted, and counted in
        the revision, so a caller holding the previous revision refreshes
        rather than committing against a profile set it never saw.  Profiles
        that already have policy are untouched: this never overwrites what
        somebody chose.
        """
        default = default or ProfilePolicy(enabled=True, weight=MIN_WEIGHT)
        async with self._lock:
            candidate = copy.deepcopy(self._state)
            adopted = tuple(
                profile_id
                for profile_id in sorted(set(profile_ids))
                if isinstance(profile_id, str) and profile_id not in candidate.profiles
            )
            if not adopted:
                return ()
            for profile_id in adopted:
                candidate.profiles[profile_id] = ProfilePolicy(
                    enabled=default.enabled, weight=default.weight
                )
            candidate.revision += 1
            try:
                await run_off_loop(self._store.save, candidate)
            except OSError:
                # Unpersisted adoption would be forgotten on the next start
                # while the running process believed it had happened.
                return ()
            self._state = candidate
        if self._on_applied is not None:
            with contextlib.suppress(Exception):
                self._on_applied()
        return adopted

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
            return self._rejection(command_id, "Command does not match the version 2 contract")
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
                await run_off_loop(self._store.save, candidate)
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
        if operation == "apply-profiles":
            return self._apply_batch(state, command["changes"])
        if operation == "set-paused":
            state.paused = command["value"] is True
            return None
        profile_id = command["profileId"]
        # Device and model are choices *about* a profile rather than settings
        # inside its policy, so they are answerable for a profile that has no
        # stored policy yet — hence one gate for both, before the policy lookup
        # the rest need.
        chooser = _CHOOSERS.get(operation)
        if chooser is not None:
            if not self._known_profile(state, profile_id):
                return f"Unknown workload profile: {profile_id}"
            return chooser(self, state, profile_id, command["value"])
        policy = state.profiles.get(profile_id)
        if policy is None:
            return f"Unknown workload profile: {profile_id}"
        if operation == "set-profile-enabled":
            policy.enabled = command["value"] is True
            return None
        weight = _integral_weight(command["value"])
        if weight is None or not MIN_WEIGHT <= weight <= MAX_WEIGHT:
            return f"Weight must be between {MIN_WEIGHT} and {MAX_WEIGHT}"
        policy.weight = weight
        return None

    def _apply_batch(self, state: PolicyState, changes: list) -> str | None:
        """Apply every change or none of them.

        The caller already mutates a copy and commits only on success, so a
        refusal anywhere here discards the whole candidate — which is the point:
        the same settings sent one at a time spend a revision each, and a
        failure part-way leaves policy half-applied with nothing to retry as a
        unit.

        Every change is checked before any is made, so a batch naming one
        unknown profile does not apply the ones before it and then stop.
        """
        for change in changes:
            unknown = self._batch_error(state, change)
            if unknown is not None:
                return unknown
        for change in changes:
            error = self._apply_batch_change(state, change)
            if error is not None:
                return error
        return None

    def _apply_batch_change(self, state: PolicyState, change: dict) -> str | None:
        policy = state.profiles.get(change["profileId"])
        if "enabled" in change:
            assert policy is not None
            policy.enabled = change["enabled"] is True
        if "weight" in change:
            assert policy is not None
            weight = _integral_weight(change["weight"])
            if weight is None:  # Contract validation makes this unreachable.
                return f"Weight must be between {MIN_WEIGHT} and {MAX_WEIGHT}"
            policy.weight = weight
        if "deviceId" in change:
            error = self._set_profile_device(
                state, change["profileId"], change["deviceId"]
            )
            if error is not None:
                return error
        if "modelId" in change:
            return self._set_profile_model(
                state, change["profileId"], change["modelId"]
            )
        return None

    def _batch_error(self, state: PolicyState, change: dict) -> str | None:
        profile_id = change["profileId"]
        if not self._known_profile(state, profile_id):
            return f"Unknown workload profile: {profile_id}"
        if not {"enabled", "weight", "deviceId", "modelId"} & set(change):
            # A change that changes nothing is a caller mistake worth naming:
            # silently accepting it would spend a revision and alter nothing.
            return f"Change for {profile_id} sets neither enabled nor weight"
        if ("enabled" in change or "weight" in change) and profile_id not in state.profiles:
            return f"Profile policy unavailable: {profile_id}"
        if "deviceId" in change:
            error = self._device_choice_error(change["deviceId"])
            if error is not None:
                return error
        if "modelId" in change:
            return self._model_choice_error(change["profileId"], change["modelId"])
        return None

    def _known_profile(self, state: PolicyState, profile_id: str) -> bool:
        return (
            profile_id in state.profiles
            or profile_id in state.device_choices
            or profile_id in state.model_choices
            or (
                self._profile_exists is not None
                and self._profile_exists(profile_id) is True
            )
        )

    def _device_choice_error(self, device_id: str | None) -> str | None:
        if device_id is None:
            return None
        if device_id not in set(self._gpu_device_ids()):
            return f"GPU device is unavailable: {device_id}"
        return None

    def _model_choice_error(self, profile_id: str, model_id: str | None) -> str | None:
        if model_id is None:
            return None
        declared = tuple(self._profile_models(profile_id))
        if model_id not in declared:
            # Named rather than silently ignored: a person who chose a model
            # and got the old one back would rightly not believe the window.
            return f"Model is not declared by this workload: {model_id}"
        return None

    def _set_profile_model(
        self, state: PolicyState, profile_id: str, model_id: str | None
    ) -> str | None:
        error = self._model_choice_error(profile_id, model_id)
        if error is not None:
            return error
        if model_id is None:
            # Back to whatever the qualification receipt defaults to, which is
            # a different fact from having chosen that model.
            state.model_choices.pop(profile_id, None)
        else:
            if (
                profile_id not in state.model_choices
                and len(state.model_choices) >= MAX_MODEL_CHOICES
            ):
                return f"Model choices are limited to {MAX_MODEL_CHOICES} profiles"
            state.model_choices[profile_id] = model_id
        return None

    def _set_profile_device(
        self, state: PolicyState, profile_id: str, device_id: str | None
    ) -> str | None:
        error = self._device_choice_error(device_id)
        if error is not None:
            return error
        if device_id is None:
            state.device_choices.pop(profile_id, None)
        else:
            if (
                profile_id not in state.device_choices
                and len(state.device_choices) >= MAX_DEVICE_CHOICES
            ):
                return f"GPU device choices are limited to {MAX_DEVICE_CHOICES} profiles"
            state.device_choices[profile_id] = device_id
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


def build_control_service(
    state_path: Path,
    defaults: dict[str, ProfilePolicy],
    *,
    profile_exists=None,
    gpu_device_ids=None,
) -> ControlService:
    return ControlService(
        PolicyStore(state_path, defaults),
        profile_exists=profile_exists,
        gpu_device_ids=gpu_device_ids,
    )
