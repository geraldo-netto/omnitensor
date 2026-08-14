"""The shape every bounded host collector shares.

The network and peripheral collectors arrived at the same structure
independently, and every remaining profile needs it too: consent before
anything is read, a source that can be replayed instead of touching hardware,
an allowlist of what may be emitted, a hard cap on how much, and churn reported
against what was *eligible* rather than what happened to fit.

Writing that seven more times would be seven more places for one of those rules
to be forgotten — and the ones that get forgotten are the privacy rules, because
nothing fails when they are missing.  So the skeleton lives here once and each
profile supplies only what is genuinely specific to it: what a sample is, how
to validate it, and how it becomes a document.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable

from ..registry import validate_document
from ..telemetry_types import STABLE_ID
from .pipeline import CollectedOutput
from .triggers import CollectorReadiness, SourceStatus, Trigger

MAX_COLLECTED_ITEMS = 1024
COLLECTED_OUTPUT_SCHEMA = "collected-output.schema.json"

Sample = TypeVar("Sample")


class CollectionError(ValueError):
    """Stable collection rejection safe to surface through orchestration."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@runtime_checkable
class CollectionPermissionGate(Protocol):
    """Read-only declared/granted permission intersection."""

    def allows(self, permission: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class SourceSnapshot(Generic[Sample]):
    """One bounded observation of a host source."""

    status: SourceStatus
    observed_at_ms: int
    items: tuple[Sample, ...]


@runtime_checkable
class MetadataSource(Protocol):
    """A trusted host adapter, always replaceable by a replay for tests."""

    async def readiness(self) -> CollectorReadiness: ...

    async def snapshot(self) -> SourceSnapshot: ...


class ReplaySource:
    """Deterministic snapshot replay that touches no hardware.

    Every collector gets one of these for free, because a collector that can
    only be exercised against real hardware cannot be tested for the cases that
    matter — a missing sensor, a device removed mid-scan, a source that goes
    unavailable — which are exactly the ones that reach users.
    """

    def __init__(self, snapshots: Sequence[SourceSnapshot], *, label: str = "source"):
        if isinstance(snapshots, (str, bytes)) or not isinstance(snapshots, Sequence):
            raise TypeError("snapshots must be a sequence")
        self._snapshots = tuple(snapshots)
        self._label = label
        self._index = 0

    async def readiness(self) -> CollectorReadiness:
        if not self._snapshots:
            return CollectorReadiness(
                SourceStatus.UNAVAILABLE, f"{self._label} replay is empty", 0
            )
        snapshot = self._snapshots[self._index]
        return CollectorReadiness(
            snapshot.status,
            source_health_detail(self._label, snapshot.status),
            snapshot.observed_at_ms,
        )

    async def snapshot(self) -> SourceSnapshot:
        if not self._snapshots:
            raise CollectionError("source-unavailable", f"{self._label} replay is empty")
        snapshot = self._snapshots[self._index]
        if self._index < len(self._snapshots) - 1:
            # The final snapshot repeats, so a collector polled more often than
            # the fixture has states keeps observing a stable world instead of
            # falling off the end.
            self._index += 1
        return snapshot


def _check_declared_plugin_id(collector_type: type) -> None:
    plugin_id = collector_type.__dict__.get("plugin_id")
    if not isinstance(plugin_id, str) or not plugin_id:
        raise TypeError(
            "BoundedCollector subclasses must declare a non-empty plugin_id"
        )


class BoundedCollector(Generic[Sample]):
    """Consent, allowlisting, bounding, and churn for one host source.

    Subclasses supply :meth:`identity_of`, :meth:`document_of`, and the
    permission names; everything a privacy or bounding rule depends on is
    enforced here so it cannot be omitted per profile.
    """

    #: Plugin identity accepted by :meth:`collect`; every subclass must declare it.
    plugin_id: str
    #: Error contract retained by profile-specific collector APIs.
    error_type: type[CollectionError] = CollectionError
    #: Permission that must be granted before the source is read at all.
    metadata_permission: str = ""
    #: Human label used in readiness and health details.
    label: str = "source"
    #: Profile-specific wording retained for readiness and trigger diagnostics.
    source_label: str = ""
    readiness_label: str = ""
    trigger_label: str = ""
    #: Schema identifier written into every emitted document.
    source_name: str = "source"
    #: Family-specific collection keys; the rest of the envelope stays shared.
    items_key: str = "items"
    truncated_items_key: str = "truncatedItems"
    #: Privacy-sensitive families require a non-empty explicit allowlist.
    require_allowlist: bool = False
    #: Existing collectors sort changed field names in their churn contract.
    sort_changed_fields: bool = True

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        _check_declared_plugin_id(cls)

    def __init__(
        self,
        source: MetadataSource,
        permissions: CollectionPermissionGate,
        allowed_ids: Collection[str] = (),
        *,
        max_items: int = 64,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        if not isinstance(source, MetadataSource):
            raise TypeError("source must implement MetadataSource")
        if not isinstance(permissions, CollectionPermissionGate):
            raise TypeError("permissions must implement CollectionPermissionGate")
        if type(max_items) is not int or not 1 <= max_items <= MAX_COLLECTED_ITEMS:
            raise ValueError(f"max_items must be an integer from 1 to {MAX_COLLECTED_ITEMS}")
        if clock_ms is not None and not callable(clock_ms):
            raise TypeError("clock_ms must be callable")
        self._source = source
        self._permissions = permissions
        self._allowed_ids = validated_allowlist(allowed_ids)
        self._max_items = max_items
        self._clock_ms = clock_ms or _now_ms
        self._previous: dict[str, Sample] = {}

    # --- hooks a profile supplies -------------------------------------------------

    def identity_of(self, item: Sample) -> str:
        """The stable identity this item is allowlisted and tracked by."""
        raise NotImplementedError

    def document_of(self, item: Sample) -> dict:
        """The redacted, JSON-safe document emitted for this item."""
        raise NotImplementedError

    def changed_fields(self, previous: Sample, current: Sample) -> list[str]:
        """Which fields changed, ignoring timestamps that always change."""
        return [] if previous == current else ["state"]

    def item_permission(self, identity: str) -> str | None:
        """A per-item grant this profile requires, or ``None`` for one gate."""
        return None

    def identity_document(self, item: Sample) -> dict[str, object]:
        """The public churn identity for one eligible item."""
        return {"id": self.identity_of(item)}

    # --- the shared behaviour -----------------------------------------------------

    async def readiness(self) -> CollectorReadiness:
        if not self._permissions.allows(self.metadata_permission):
            return CollectorReadiness(
                SourceStatus.UNAVAILABLE,
                f"{self.label} permission is not granted",
                self._clock_ms(),
            )
        try:
            readiness = await self._source.readiness()
        except Exception as error:  # noqa: BLE001 - host adapters fail arbitrarily
            return CollectorReadiness(
                SourceStatus.UNAVAILABLE,
                f"{self.readiness_label or self.label} readiness failed: "
                f"{type(error).__name__}",
                self._clock_ms(),
            )
        if (
            not isinstance(readiness, CollectorReadiness)
            or not isinstance(readiness.status, SourceStatus)
            or _non_negative_integer_error(readiness.checked_at_ms)
        ):
            return CollectorReadiness(
                SourceStatus.UNAVAILABLE,
                f"{self.readiness_label or self.label} readiness is invalid",
                self._clock_ms(),
            )
        return CollectorReadiness(
            readiness.status,
            source_health_detail(self.source_label or self.label, readiness.status),
            readiness.checked_at_ms,
        )

    async def collect(self, trigger: Trigger) -> CollectedOutput:
        """Read the source once and emit only what consent and bounds allow."""
        self._validate_trigger(trigger)
        if not self._permissions.allows(self.metadata_permission):
            raise self.error_type(
                "permission-denied", f"{self.label} permission is not granted"
            )
        snapshot = await self._source.snapshot()
        self._validate_snapshot(snapshot)
        items = self._snapshot_items(snapshot)
        eligible = sorted(
            (item for item in items if self._may_emit(self.identity_of(item))),
            key=self.identity_of,
        )
        selected = eligible[: self._max_items]
        # Churn is tracked against everything eligible, not the truncated
        # emission list: an item past the cap is still present, and reporting
        # it removed would be churn the user never experienced.
        current = {self.identity_of(item): item for item in eligible}
        authorized_previous = {
            identity: item
            for identity, item in self._previous.items()
            if self._may_emit(identity)
        }
        churn = self._churn(authorized_previous, current)
        document = self._output_document(
            snapshot,
            selected,
            churn,
            len(eligible) - len(selected),
        )
        document.update(await self._output_extensions())
        output = self._collected_output(document)
        # Churn is transactional: invalid snapshots or output construction
        # failures must not replace the last successfully emitted state.
        self._previous = current
        return output

    def _snapshot_items(self, snapshot: object) -> Sequence[Sample]:
        assert isinstance(snapshot, SourceSnapshot)
        return snapshot.items

    def _output_document(
        self,
        snapshot: object,
        selected: Sequence[Sample],
        churn: dict[str, list],
        truncated: int,
    ) -> dict[str, object]:
        return {
            "schemaVersion": 1,
            "source": self.source_name,
            "sourceHealth": str(snapshot.status),
            "observedAtMs": snapshot.observed_at_ms,
            self.items_key: [self.document_of(item) for item in selected],
            "churn": churn,
            self.truncated_items_key: truncated,
        }

    def _collected_output(self, document: dict[str, object]) -> CollectedOutput:
        """Single shared emission boundary for canonical validation."""
        violations = validate_document(COLLECTED_OUTPUT_SCHEMA, document)
        if violations:
            raise self.error_type("output-invalid", violations[0])
        return CollectedOutput(document)

    async def _output_extensions(self) -> dict[str, object]:
        """Family-specific fields merged before validation and churn commit."""
        return {}

    def _churn(
        self, previous: dict[str, Sample], current: dict[str, Sample]
    ) -> dict[str, list]:
        added = [
            self.identity_document(current[identity])
            for identity in sorted(current.keys() - previous.keys())
        ]
        removed = [
            self.identity_document(previous[identity])
            for identity in sorted(previous.keys() - current.keys())
        ]
        changed = []
        for identity in sorted(previous.keys() & current.keys()):
            fields = self.changed_fields(previous[identity], current[identity])
            if fields:
                changed.append(
                    {
                        **self.identity_document(current[identity]),
                        "fields": sorted(fields) if self.sort_changed_fields else fields,
                    }
                )
        return {
            "added": added,
            "removed": removed,
            "changed": changed,
        }

    def _may_emit(self, identity: str) -> bool:
        if (self.require_allowlist or self._allowed_ids) and identity not in self._allowed_ids:
            return False
        permission = self.item_permission(identity)
        return permission is None or self._permissions.allows(permission)

    def _validate_trigger(self, trigger: Trigger) -> None:
        subject = self.trigger_label or self.label
        if not isinstance(trigger, Trigger):
            raise self.error_type("trigger-invalid", f"{subject} requires a Trigger")
        if trigger.plugin_id != self.plugin_id:
            raise self.error_type(
                "trigger-invalid",
                f"{subject} requires a {self.plugin_id} trigger",
            )

    def _validate_snapshot(self, snapshot: object) -> None:
        if not isinstance(snapshot, SourceSnapshot):
            raise self.error_type("source-invalid", f"{self.label} returned no snapshot")
        if not isinstance(snapshot.status, SourceStatus):
            raise self.error_type("source-invalid", f"{self.label} status is invalid")
        if _non_negative_integer_error(snapshot.observed_at_ms):
            raise self.error_type("source-invalid", f"{self.label} timestamp is invalid")
        if len(snapshot.items) > MAX_COLLECTED_ITEMS:
            raise self.error_type(
                "source-invalid",
                f"{self.label} returned more than {MAX_COLLECTED_ITEMS} items",
            )
        for item in snapshot.items:
            try:
                identity = self.identity_of(item)
            except (AttributeError, TypeError) as error:
                # A source that returns the wrong shape is a stable rejection,
                # not a traceback escaping into orchestration.
                raise self.error_type(
                    "source-invalid", f"{self.label} item has an invalid type"
                ) from error
            if not isinstance(identity, str) or not STABLE_ID.fullmatch(identity):
                raise self.error_type(
                    "source-invalid", f"{self.label} item identity is invalid"
                )


def validated_allowlist(allowed_ids: Collection[str]) -> frozenset[str]:
    """An explicit allowlist of identities a collector may ever emit."""
    if isinstance(allowed_ids, (str, bytes)):
        raise TypeError("allowed_ids must be a collection of identities")
    identities = frozenset(allowed_ids)
    for identity in identities:
        if not isinstance(identity, str) or not STABLE_ID.fullmatch(identity):
            raise ValueError(f"allowlisted identity is invalid: {identity!r}")
    if len(identities) > MAX_COLLECTED_ITEMS:
        raise ValueError(f"at most {MAX_COLLECTED_ITEMS} identities may be allowlisted")
    return identities


def source_health_detail(label: str, status: SourceStatus) -> str:
    if status is SourceStatus.READY:
        return f"{label} is ready"
    if status is SourceStatus.DEGRADED:
        return f"{label} is degraded"
    return f"{label} is unavailable"


def bounded_number(value: object, name: str, maximum: int) -> int:
    """One non-negative counter within a declared ceiling."""
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise CollectionError("source-invalid", f"{name} must be an integer in [0, {maximum}]")
    return value


def _non_negative_integer_error(value: object) -> bool:
    return isinstance(value, bool) or not isinstance(value, int) or value < 0


def _strict_non_negative_integer_error(value: object) -> bool:
    """Reject integer subclasses at the native host-adapter boundary."""
    return type(value) is not int or value < 0


def _now_ms() -> int:
    return int(time.time() * 1000)
