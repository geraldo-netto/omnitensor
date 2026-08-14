"""Bounded trigger scheduling and asynchronous source collection."""

from __future__ import annotations

import asyncio
import json
import re
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from ..telemetry_types import SourceStatus
from .pipeline import CollectedOutput
from .protocol import JsonObject

DEFAULT_MAX_TRIGGER_BYTES = 256 * 1024
DEFAULT_MAX_PENDING_TRIGGERS = 256
DEFAULT_COLLECTION_TIMEOUT_SECONDS = 30.0
_IDENTIFIER = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_MAX_CORRELATION_CHARS = 128


class TriggerKind(StrEnum):
    MANUAL = "manual"
    EVENT = "event"
    PERIODIC = "periodic"


class TriggerDisposition(StrEnum):
    ACCEPTED = "accepted"
    COALESCED = "coalesced"
    QUEUE_FULL = "queue-full"


class CollectionStatus(StrEnum):
    SUCCEEDED = "succeeded"
    TIMED_OUT = "timed-out"
    FAILED = "failed"


class TriggerValidationError(ValueError):
    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class Trigger:
    """One manual, event, or periodic collection request."""

    plugin_id: str
    trigger_id: str
    kind: TriggerKind
    payload: JsonObject
    created_at_ms: int


@dataclass(frozen=True, slots=True)
class TriggerReceipt:
    trigger: Trigger
    disposition: TriggerDisposition
    pending_count: int


@dataclass(frozen=True, slots=True)
class CollectorReadiness:
    status: SourceStatus
    detail: str
    checked_at_ms: int


@dataclass(frozen=True, slots=True)
class CollectionOutcome:
    trigger: Trigger
    status: CollectionStatus
    output: CollectedOutput | None
    detail: str


@runtime_checkable
class Collector(Protocol):
    """External source adapter; every operation must remain asynchronous."""

    async def readiness(self) -> CollectorReadiness: ...

    async def collect(self, trigger: Trigger) -> CollectedOutput: ...


@dataclass(slots=True)
class _PeriodicRegistration:
    plugin_id: str
    interval_ms: int
    next_due_ms: int
    payload: JsonObject
    sequence: int = 0


class TriggerCoordinator:
    """Coalesce bounded work and execute one source collection at a time."""

    def __init__(
        self,
        *,
        max_trigger_bytes: int = DEFAULT_MAX_TRIGGER_BYTES,
        max_pending: int = DEFAULT_MAX_PENDING_TRIGGERS,
        collection_timeout_seconds: float = DEFAULT_COLLECTION_TIMEOUT_SECONDS,
    ):
        if isinstance(max_trigger_bytes, bool) or not isinstance(max_trigger_bytes, int):
            raise ValueError("max_trigger_bytes must be a positive integer")
        if max_trigger_bytes < 1:
            raise ValueError("max_trigger_bytes must be a positive integer")
        if isinstance(max_pending, bool) or not isinstance(max_pending, int) or max_pending < 1:
            raise ValueError("max_pending must be a positive integer")
        if (
            isinstance(collection_timeout_seconds, bool)
            or not isinstance(collection_timeout_seconds, (int, float))
            or collection_timeout_seconds <= 0
        ):
            raise ValueError("collection_timeout_seconds must be positive")
        self._max_trigger_bytes = max_trigger_bytes
        self._max_pending = max_pending
        self._collection_timeout_seconds = float(collection_timeout_seconds)
        self._collectors: dict[str, Collector] = {}
        self._pending: deque[Trigger] = deque()
        self._pending_keys: set[tuple[str, TriggerKind, str]] = set()
        self._periodic: dict[str, _PeriodicRegistration] = {}

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    def register_collector(self, plugin_id: str, collector: Collector) -> None:
        _validate_plugin_id(plugin_id)
        if not isinstance(collector, Collector):
            raise TypeError("collector must implement readiness and collect")
        if plugin_id in self._collectors:
            raise ValueError(f"collector already registered: {plugin_id}")
        self._collectors[plugin_id] = collector

    def register_periodic(
        self,
        plugin_id: str,
        *,
        interval_ms: int,
        first_due_ms: int,
        payload: JsonObject,
    ) -> None:
        self._require_collector(plugin_id)
        _validate_non_negative_integer(first_due_ms, "first_due_ms")
        if isinstance(interval_ms, bool) or not isinstance(interval_ms, int) or interval_ms < 1:
            raise ValueError("interval_ms must be a positive integer")
        self._validate_payload(payload)
        if plugin_id in self._periodic:
            raise ValueError(f"periodic trigger already registered: {plugin_id}")
        self._periodic[plugin_id] = _PeriodicRegistration(
            plugin_id,
            interval_ms,
            first_due_ms,
            payload,
        )

    def submit_manual(
        self,
        plugin_id: str,
        trigger_id: str,
        payload: JsonObject,
        *,
        created_at_ms: int,
    ) -> TriggerReceipt:
        return self.submit(
            Trigger(plugin_id, trigger_id, TriggerKind.MANUAL, payload, created_at_ms)
        )

    def submit_event(
        self,
        plugin_id: str,
        event_id: str,
        payload: JsonObject,
        *,
        created_at_ms: int,
    ) -> TriggerReceipt:
        return self.submit(Trigger(plugin_id, event_id, TriggerKind.EVENT, payload, created_at_ms))

    def tick(self, now_ms: int) -> tuple[TriggerReceipt, ...]:
        """Submit at most one trigger per overdue periodic source."""
        _validate_non_negative_integer(now_ms, "now_ms")
        receipts = []
        for registration in sorted(self._periodic.values(), key=lambda value: value.plugin_id):
            if registration.next_due_ms > now_ms:
                continue
            registration.sequence += 1
            trigger = Trigger(
                registration.plugin_id,
                f"periodic-{registration.sequence}",
                TriggerKind.PERIODIC,
                registration.payload,
                now_ms,
            )
            receipts.append(self.submit(trigger))
            registration.next_due_ms = now_ms + registration.interval_ms
        return tuple(receipts)

    async def readiness(self, plugin_id: str) -> CollectorReadiness:
        """Query readiness with the same deadline as collection."""
        collector = self._require_collector(plugin_id)
        try:
            readiness = await asyncio.wait_for(
                collector.readiness(),
                timeout=self._collection_timeout_seconds,
            )
        except TimeoutError:
            return CollectorReadiness(SourceStatus.UNAVAILABLE, "readiness timed out", 0)
        except Exception as error:
            return CollectorReadiness(
                SourceStatus.UNAVAILABLE,
                f"readiness failed: {type(error).__name__}",
                0,
            )
        if not isinstance(readiness, CollectorReadiness):
            return CollectorReadiness(SourceStatus.UNAVAILABLE, "invalid readiness response", 0)
        return readiness

    async def collect_next(self) -> CollectionOutcome | None:
        """Collect the oldest trigger; return immediately when no work exists."""
        if not self._pending:
            return None
        trigger = self._pending.popleft()
        self._pending_keys.remove(self._key(trigger))
        collector = self._collectors[trigger.plugin_id]
        try:
            output = await asyncio.wait_for(
                collector.collect(trigger),
                timeout=self._collection_timeout_seconds,
            )
        except TimeoutError:
            return CollectionOutcome(
                trigger,
                CollectionStatus.TIMED_OUT,
                None,
                "collection timed out",
            )
        except Exception as error:
            return CollectionOutcome(
                trigger,
                CollectionStatus.FAILED,
                None,
                f"collection failed: {type(error).__name__}",
            )
        if not isinstance(output, CollectedOutput):
            return CollectionOutcome(
                trigger,
                CollectionStatus.FAILED,
                None,
                "collector returned an invalid output",
            )
        try:
            self._validate_payload(output.payload)
        except TriggerValidationError as error:
            return CollectionOutcome(trigger, CollectionStatus.FAILED, None, error.detail)
        return CollectionOutcome(trigger, CollectionStatus.SUCCEEDED, output, "")

    def submit(self, trigger: Trigger) -> TriggerReceipt:
        """Submit a typed trigger after enforcing all boundary limits."""
        self._require_collector(trigger.plugin_id)
        _validate_correlation_id(trigger.trigger_id)
        if not isinstance(trigger.kind, TriggerKind):
            raise TriggerValidationError("invalid-trigger-kind", "unsupported trigger kind")
        _validate_non_negative_integer(trigger.created_at_ms, "created_at_ms")
        self._validate_payload(trigger.payload)
        key = self._key(trigger)
        if key in self._pending_keys:
            return TriggerReceipt(trigger, TriggerDisposition.COALESCED, len(self._pending))
        if len(self._pending) >= self._max_pending:
            return TriggerReceipt(trigger, TriggerDisposition.QUEUE_FULL, len(self._pending))
        self._pending.append(trigger)
        self._pending_keys.add(key)
        return TriggerReceipt(trigger, TriggerDisposition.ACCEPTED, len(self._pending))

    def _require_collector(self, plugin_id: str) -> Collector:
        _validate_plugin_id(plugin_id)
        try:
            return self._collectors[plugin_id]
        except KeyError as error:
            raise KeyError(f"collector not registered: {plugin_id}") from error

    def _validate_payload(self, payload: JsonObject) -> None:
        if not isinstance(payload, dict):
            raise TriggerValidationError("invalid-payload", "payload must be a JSON object")
        try:
            encoded = json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        except (TypeError, ValueError) as error:
            raise TriggerValidationError(
                "invalid-payload",
                "payload must contain JSON values",
            ) from error
        if len(encoded) > self._max_trigger_bytes:
            raise TriggerValidationError(
                "payload-too-large",
                f"payload is {len(encoded)} bytes; limit is {self._max_trigger_bytes}",
            )

    @staticmethod
    def _key(trigger: Trigger) -> tuple[str, TriggerKind, str]:
        return trigger.plugin_id, trigger.kind, trigger.trigger_id


def _validate_plugin_id(plugin_id: str) -> None:
    if not isinstance(plugin_id, str) or not _IDENTIFIER.fullmatch(plugin_id):
        raise TriggerValidationError("invalid-plugin-id", f"invalid plugin id: {plugin_id!r}")


def _validate_correlation_id(trigger_id: str) -> None:
    if (
        not isinstance(trigger_id, str)
        or not trigger_id
        or len(trigger_id) > _MAX_CORRELATION_CHARS
    ):
        raise TriggerValidationError(
            "invalid-trigger-id",
            f"trigger id must contain 1-{_MAX_CORRELATION_CHARS} characters",
        )


def _validate_non_negative_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
