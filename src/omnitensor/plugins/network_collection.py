"""Consent-gated aggregate network metadata collection."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from .pipeline import CollectedOutput
from .triggers import CollectorReadiness, SourceStatus, Trigger

NETWORK_METADATA_PERMISSION = "read:network-metadata"
NETWORK_PERIPHERALS_PLUGIN_ID = "network-peripherals"
DEFAULT_MAX_NETWORK_LINKS = 64
MAX_NETWORK_LINKS = 256
MAX_NETWORK_COUNTER = 2**63 - 1
_STABLE_ID = re.compile(r"[a-z0-9](?:[a-z0-9._-]{0,78}[a-z0-9])?")


class NetworkLinkKind(StrEnum):
    ETHERNET = "ethernet"
    WIFI = "wifi"
    CELLULAR = "cellular"
    VPN = "vpn"
    OTHER = "other"


class NetworkLinkState(StrEnum):
    UP = "up"
    DOWN = "down"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class NetworkConnectivity(StrEnum):
    FULL = "full"
    LIMITED = "limited"
    PORTAL = "portal"
    NONE = "none"
    UNKNOWN = "unknown"


class NetworkCollectionError(ValueError):
    """Stable network collection rejection."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class NetworkCounters:
    """Monotonic link counters; never packet contents."""

    received_bytes: int
    transmitted_bytes: int
    received_errors: int
    transmitted_errors: int
    received_drops: int
    transmitted_drops: int


@dataclass(frozen=True, slots=True)
class NetworkLinkSample:
    """One aggregate NetworkManager/link observation."""

    stable_id: str
    kind: NetworkLinkKind
    state: NetworkLinkState
    connectivity: NetworkConnectivity
    carrier: bool
    metered: bool
    default_route: bool
    signal_percent: int | None
    counters: NetworkCounters
    observed_at_ms: int


@dataclass(frozen=True, slots=True)
class NetworkSnapshot:
    """Bounded-source snapshot used by live adapters and replay tests."""

    status: SourceStatus
    observed_at_ms: int
    links: tuple[NetworkLinkSample, ...]


@runtime_checkable
class CollectionPermissionGate(Protocol):
    """Read-only declared/granted permission intersection."""

    def allows(self, permission: str) -> bool: ...


@runtime_checkable
class NetworkMetadataSource(Protocol):
    """Trusted host adapter for NetworkManager and kernel link metadata."""

    async def readiness(self) -> CollectorReadiness: ...

    async def snapshot(self) -> NetworkSnapshot: ...


class ReplayNetworkMetadataSource:
    """Deterministic typed snapshot replay without network or hardware access."""

    def __init__(self, snapshots: Sequence[NetworkSnapshot]):
        if isinstance(snapshots, (str, bytes)) or not isinstance(snapshots, Sequence):
            raise TypeError("snapshots must be a sequence")
        self._snapshots = tuple(snapshots)
        self._index = 0

    async def readiness(self) -> CollectorReadiness:
        if not self._snapshots:
            return CollectorReadiness(SourceStatus.UNAVAILABLE, "network replay is empty", 0)
        snapshot = self._snapshots[self._index]
        error = network_snapshot_error(snapshot)
        if error:
            return CollectorReadiness(SourceStatus.UNAVAILABLE, "network replay is invalid", 0)
        return CollectorReadiness(
            snapshot.status,
            _source_health_detail(snapshot.status),
            snapshot.observed_at_ms,
        )

    async def snapshot(self) -> NetworkSnapshot:
        if not self._snapshots:
            raise NetworkCollectionError("source-unavailable", "network replay is empty")
        snapshot = self._snapshots[self._index]
        if self._index < len(self._snapshots) - 1:
            self._index += 1
        return snapshot


class NetworkMetadataCollector:
    """Emit deterministic aggregate link features after explicit consent."""

    def __init__(
        self,
        source: NetworkMetadataSource,
        permissions: CollectionPermissionGate,
        *,
        max_links: int = DEFAULT_MAX_NETWORK_LINKS,
        clock_ms: Callable[[], int] | None = None,
    ):
        if not isinstance(source, NetworkMetadataSource):
            raise TypeError("source must implement NetworkMetadataSource")
        if not isinstance(permissions, CollectionPermissionGate):
            raise TypeError("permissions must implement CollectionPermissionGate")
        if type(max_links) is not int or not 1 <= max_links <= MAX_NETWORK_LINKS:
            raise ValueError(f"max_links must be an integer from 1 to {MAX_NETWORK_LINKS}")
        if clock_ms is not None and not callable(clock_ms):
            raise TypeError("clock_ms must be callable")
        self._source = source
        self._permissions = permissions
        self._max_links = max_links
        self._clock_ms = clock_ms or _now_ms

    async def readiness(self) -> CollectorReadiness:
        if not self._permissions.allows(NETWORK_METADATA_PERMISSION):
            return CollectorReadiness(
                SourceStatus.UNAVAILABLE,
                "network metadata permission is not granted",
                self._clock_ms(),
            )
        try:
            readiness = await self._source.readiness()
        except Exception as error:  # noqa: BLE001 - host adapters can fail arbitrarily
            return CollectorReadiness(
                SourceStatus.UNAVAILABLE,
                f"network source readiness failed: {type(error).__name__}",
                self._clock_ms(),
            )
        if not isinstance(readiness, CollectorReadiness):
            return CollectorReadiness(
                SourceStatus.UNAVAILABLE,
                "network source readiness is invalid",
                self._clock_ms(),
            )
        if not isinstance(readiness.status, SourceStatus) or _non_negative_integer_error(
            readiness.checked_at_ms
        ):
            return CollectorReadiness(
                SourceStatus.UNAVAILABLE,
                "network source readiness is invalid",
                self._clock_ms(),
            )
        return CollectorReadiness(
            readiness.status,
            _source_health_detail(readiness.status),
            readiness.checked_at_ms,
        )

    async def collect(self, trigger: Trigger) -> CollectedOutput:
        if not isinstance(trigger, Trigger) or trigger.plugin_id != NETWORK_PERIPHERALS_PLUGIN_ID:
            raise NetworkCollectionError(
                "trigger-invalid", "network collector requires a network-peripherals trigger"
            )
        if not self._permissions.allows(NETWORK_METADATA_PERMISSION):
            raise NetworkCollectionError(
                "permission-denied", "network metadata permission is not granted"
            )
        snapshot = await self._source.snapshot()
        error = network_snapshot_error(snapshot)
        if error:
            raise NetworkCollectionError("source-invalid", error)
        ordered = sorted(snapshot.links, key=lambda link: link.stable_id)
        selected = ordered[: self._max_links]
        return CollectedOutput(
            {
                "schemaVersion": 1,
                "source": "network-manager-link-metadata",
                "sourceHealth": str(snapshot.status),
                "observedAtMs": snapshot.observed_at_ms,
                "links": [_link_document(link) for link in selected],
                "truncatedLinks": len(ordered) - len(selected),
            }
        )


def network_snapshot_error(snapshot: object) -> str:
    """Return one stable validation failure for an untrusted source snapshot."""
    error = _network_snapshot_shape_error(snapshot)
    if error:
        return error
    assert isinstance(snapshot, NetworkSnapshot)
    identities: set[str] = set()
    for link in snapshot.links:
        error = _network_link_error(link, snapshot.observed_at_ms)
        if error:
            return error
        if link.stable_id in identities:
            return f"duplicate network link identity: {link.stable_id}"
        identities.add(link.stable_id)
    return ""


def _network_snapshot_shape_error(snapshot: object) -> str:
    if not isinstance(snapshot, NetworkSnapshot):
        return "network snapshot is not typed"
    if not isinstance(snapshot.status, SourceStatus):
        return "network source status is invalid"
    if _non_negative_integer_error(snapshot.observed_at_ms):
        return "network snapshot timestamp is invalid"
    if not isinstance(snapshot.links, tuple):
        return "network links must be a tuple"
    if len(snapshot.links) > MAX_NETWORK_LINKS:
        return f"network snapshot exceeds {MAX_NETWORK_LINKS} links"
    return ""


def _network_link_error(link: object, snapshot_time_ms: int) -> str:
    if not isinstance(link, NetworkLinkSample):
        return "network link sample is not typed"
    if not isinstance(link.stable_id, str) or not _STABLE_ID.fullmatch(link.stable_id):
        return "network link stable identity is invalid"
    enum_error = _network_link_enum_error(link)
    if enum_error:
        return enum_error
    if any(type(value) is not bool for value in (link.carrier, link.metered, link.default_route)):
        return f"network link flags are invalid: {link.stable_id}"
    if link.signal_percent is not None and (
        type(link.signal_percent) is not int or not 0 <= link.signal_percent <= 100
    ):
        return f"network link signal is invalid: {link.stable_id}"
    counter_error = _network_counter_error(link.counters)
    if counter_error:
        return f"{counter_error}: {link.stable_id}"
    if _non_negative_integer_error(link.observed_at_ms) or link.observed_at_ms > snapshot_time_ms:
        return f"network link timestamp is invalid: {link.stable_id}"
    return ""


def _network_link_enum_error(link: NetworkLinkSample) -> str:
    if not isinstance(link.kind, NetworkLinkKind):
        return f"network link kind is invalid: {link.stable_id}"
    if not isinstance(link.state, NetworkLinkState):
        return f"network link state is invalid: {link.stable_id}"
    if not isinstance(link.connectivity, NetworkConnectivity):
        return f"network link connectivity is invalid: {link.stable_id}"
    return ""


def _network_counter_error(counters: object) -> str:
    if not isinstance(counters, NetworkCounters):
        return "network counters are not typed"
    values = (
        counters.received_bytes,
        counters.transmitted_bytes,
        counters.received_errors,
        counters.transmitted_errors,
        counters.received_drops,
        counters.transmitted_drops,
    )
    if any(type(value) is not int or not 0 <= value <= MAX_NETWORK_COUNTER for value in values):
        return "network counters are invalid"
    return ""


def _link_document(link: NetworkLinkSample) -> dict[str, object]:
    return {
        "stableId": link.stable_id,
        "kind": str(link.kind),
        "state": str(link.state),
        "connectivity": str(link.connectivity),
        "carrier": link.carrier,
        "metered": link.metered,
        "defaultRoute": link.default_route,
        "signalPercent": link.signal_percent,
        "counters": {
            "receivedBytes": link.counters.received_bytes,
            "transmittedBytes": link.counters.transmitted_bytes,
            "receivedErrors": link.counters.received_errors,
            "transmittedErrors": link.counters.transmitted_errors,
            "receivedDrops": link.counters.received_drops,
            "transmittedDrops": link.counters.transmitted_drops,
        },
        "observedAtMs": link.observed_at_ms,
    }


def _source_health_detail(status: SourceStatus) -> str:
    if status is SourceStatus.READY:
        return "network metadata source is ready"
    if status is SourceStatus.DEGRADED:
        return "network metadata source is degraded"
    return "network metadata source is unavailable"


def _non_negative_integer_error(value: object) -> bool:
    return type(value) is not int or value < 0


def _now_ms() -> int:
    return time.time_ns() // 1_000_000
