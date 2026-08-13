"""Consent-gated aggregate network metadata collection."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from .collection import (
    BoundedCollector,
    CollectionError,
    CollectionPermissionGate,
    _now_ms,  # noqa: F401 - retained private compatibility alias
)
from .collection import (
    _strict_non_negative_integer_error as _non_negative_integer_error,
)
from .triggers import CollectorReadiness, SourceStatus

NETWORK_METADATA_PERMISSION = "read:network-metadata"
NETWORK_LINK_PERMISSION_PREFIX = "read:network-link/"
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


class NetworkCollectionError(CollectionError):
    """Stable network collection rejection."""


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


class NetworkMetadataCollector(BoundedCollector[NetworkLinkSample]):
    """Emit only explicitly allowlisted links with active per-link grants."""

    plugin_id = NETWORK_PERIPHERALS_PLUGIN_ID
    error_type = NetworkCollectionError
    metadata_permission = NETWORK_METADATA_PERMISSION
    label = "network metadata"
    source_label = "network metadata source"
    readiness_label = "network source"
    trigger_label = "network collector"
    source_name = "network-manager-link-metadata"
    items_key = "links"
    truncated_items_key = "truncatedLinks"
    require_allowlist = True
    sort_changed_fields = False

    def __init__(
        self,
        source: NetworkMetadataSource,
        permissions: CollectionPermissionGate,
        allowed_link_ids: Sequence[str],
        *,
        max_links: int = DEFAULT_MAX_NETWORK_LINKS,
        clock_ms: Callable[[], int] | None = None,
    ):
        if not isinstance(source, NetworkMetadataSource):
            raise TypeError("source must implement NetworkMetadataSource")
        if not isinstance(permissions, CollectionPermissionGate):
            raise TypeError("permissions must implement CollectionPermissionGate")
        allowed = _validated_link_allowlist(allowed_link_ids)
        if type(max_links) is not int or not 1 <= max_links <= MAX_NETWORK_LINKS:
            raise ValueError(f"max_links must be an integer from 1 to {MAX_NETWORK_LINKS}")
        super().__init__(
            source,
            permissions,
            allowed,
            max_items=max_links,
            clock_ms=clock_ms,
        )

    def identity_of(self, item: NetworkLinkSample) -> str:
        return item.stable_id

    def item_permission(self, identity: str) -> str:
        return network_link_permission(identity)

    def identity_document(self, item: NetworkLinkSample) -> dict[str, object]:
        return {"stableId": item.stable_id, "kind": str(item.kind)}

    def document_of(self, item: NetworkLinkSample) -> dict[str, object]:
        return _link_document(item)

    def changed_fields(
        self, previous: NetworkLinkSample, current: NetworkLinkSample
    ) -> list[str]:
        fields = (
            ("kind", previous.kind, current.kind),
            ("state", previous.state, current.state),
            ("connectivity", previous.connectivity, current.connectivity),
            ("carrier", previous.carrier, current.carrier),
            ("metered", previous.metered, current.metered),
            ("defaultRoute", previous.default_route, current.default_route),
            ("signalPercent", previous.signal_percent, current.signal_percent),
            ("counters", previous.counters, current.counters),
        )
        return [name for name, before, after in fields if before != after]

    def _snapshot_items(self, snapshot: object) -> Sequence[NetworkLinkSample]:
        assert isinstance(snapshot, NetworkSnapshot)
        return snapshot.links

    def _validate_snapshot(self, snapshot: object) -> None:
        error = network_snapshot_error(snapshot)
        if error:
            raise NetworkCollectionError("source-invalid", error)


def network_link_permission(stable_id: str) -> str:
    """Return the scoped grant name for one validated stable link identity."""
    if not isinstance(stable_id, str) or not _STABLE_ID.fullmatch(stable_id):
        raise ValueError("network link stable identity is invalid")
    return f"{NETWORK_LINK_PERMISSION_PREFIX}{stable_id}"


def _validated_link_allowlist(identities: object) -> frozenset[str]:
    if isinstance(identities, (str, bytes)) or not isinstance(identities, Sequence):
        raise TypeError("allowed_link_ids must be a sequence")
    if len(identities) > MAX_NETWORK_LINKS:
        raise ValueError(f"at most {MAX_NETWORK_LINKS} link identities may be allowed")
    validated: set[str] = set()
    for stable_id in identities:
        network_link_permission(stable_id)
        if stable_id in validated:
            raise ValueError("allowed link identities must be unique")
        validated.add(stable_id)
    return frozenset(validated)


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
