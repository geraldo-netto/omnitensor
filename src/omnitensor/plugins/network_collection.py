"""Consent-gated aggregate network metadata collection."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

from .. import telemetry_types as _telemetry_types
from .collection import (
    BoundedCollector,
    CollectionError,
    CollectionPermissionGate,
    _now_ms,  # noqa: F401 - retained private compatibility alias
)
from .triggers import CollectorReadiness

NETWORK_METADATA_PERMISSION = "read:network-metadata"
NETWORK_LINK_PERMISSION_PREFIX = "read:network-link/"
NETWORK_PERIPHERALS_PLUGIN_ID = "network-peripherals"
DEFAULT_MAX_NETWORK_LINKS = 64
MAX_NETWORK_COUNTER = _telemetry_types.MAX_NETWORK_COUNTER
MAX_NETWORK_LINKS = _telemetry_types.MAX_NETWORK_LINKS
NetworkConnectivity = _telemetry_types.NetworkConnectivity
NetworkCounters = _telemetry_types.NetworkCounters
NetworkLinkKind = _telemetry_types.NetworkLinkKind
NetworkLinkSample = _telemetry_types.NetworkLinkSample
NetworkLinkState = _telemetry_types.NetworkLinkState
NetworkSnapshot = _telemetry_types.NetworkSnapshot
SourceStatus = _telemetry_types.SourceStatus
network_snapshot_error = _telemetry_types.network_snapshot_error
_STABLE_ID = re.compile(r"[a-z0-9](?:[a-z0-9._-]{0,78}[a-z0-9])?")


class NetworkCollectionError(CollectionError):
    """Stable network collection rejection."""


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
