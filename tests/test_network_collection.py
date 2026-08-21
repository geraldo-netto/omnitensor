from __future__ import annotations

import asyncio
import dataclasses
import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.plugins import (
    MAX_NETWORK_COUNTER,
    MAX_NETWORK_LINKS,
    NETWORK_LINK_PERMISSION_PREFIX,
    NETWORK_METADATA_PERMISSION,
    BoundedCollector,
    CollectedOutput,
    CollectionStatus,
    Collector,
    CollectorReadiness,
    NetworkCollectionError,
    NetworkConnectivity,
    NetworkCounters,
    NetworkLinkKind,
    NetworkLinkSample,
    NetworkLinkState,
    NetworkMetadataCollector,
    NetworkMetadataSource,
    NetworkSnapshot,
    ReplayNetworkMetadataSource,
    SourceStatus,
    Trigger,
    TriggerCoordinator,
    TriggerKind,
    network_link_permission,
    network_snapshot_error,
)
from omnitensor.sdk import PermissionView


def permission_view(*link_ids: str, metadata: bool = True) -> PermissionView:
    link_permissions = {network_link_permission(stable_id) for stable_id in link_ids}
    declared = frozenset({NETWORK_METADATA_PERMISSION, *link_permissions})
    granted = frozenset(
        {
            *link_permissions,
            *({NETWORK_METADATA_PERMISSION} if metadata else set()),
        }
    )
    return PermissionView(declared, granted)


def counters(**changes) -> NetworkCounters:
    values = {
        "received_bytes": 100,
        "transmitted_bytes": 200,
        "received_errors": 1,
        "transmitted_errors": 2,
        "received_drops": 3,
        "transmitted_drops": 4,
    }
    values.update(changes)
    return NetworkCounters(**values)


def link(stable_id: str = "link-a", **changes) -> NetworkLinkSample:
    values = {
        "stable_id": stable_id,
        "kind": NetworkLinkKind.ETHERNET,
        "state": NetworkLinkState.UP,
        "connectivity": NetworkConnectivity.FULL,
        "carrier": True,
        "metered": False,
        "default_route": True,
        "signal_percent": None,
        "counters": counters(),
        "observed_at_ms": 9,
    }
    values.update(changes)
    return NetworkLinkSample(**values)


def snapshot(*links: NetworkLinkSample, **changes) -> NetworkSnapshot:
    values = {
        "status": SourceStatus.READY,
        "observed_at_ms": 10,
        "links": tuple(links) or (link(),),
    }
    values.update(changes)
    return NetworkSnapshot(**values)


def test_zero_timestamps_are_valid_native_values():
    assert network_snapshot_error(snapshot(link(observed_at_ms=0), observed_at_ms=0)) == ""


def trigger(payload=None) -> Trigger:
    return Trigger(
        "network-peripherals",
        "network-sample-1",
        TriggerKind.PERIODIC,
        payload or {},
        10,
    )


def test_network_collector_runs_end_to_end_through_trigger_coordinator():
    source = ReplayNetworkMetadataSource(
        [
            snapshot(
                link("link-z", kind=NetworkLinkKind.WIFI, signal_percent=80),
                link("link-a"),
                link("link-m", state=NetworkLinkState.DEGRADED),
            )
        ]
    )
    collector = NetworkMetadataCollector(
        source,
        permission_view("link-a", "link-m", "link-z"),
        ("link-z", "link-a", "link-m"),
    )
    coordinator = TriggerCoordinator()
    coordinator.register_collector("network-peripherals", collector)
    secret_payload = {"packetPayload": "must-not-escape", "ssid": "private"}
    coordinator.submit(trigger(secret_payload))

    outcome = asyncio.run(coordinator.collect_next())

    assert isinstance(collector, Collector)
    assert isinstance(collector, BoundedCollector)
    assert isinstance(source, NetworkMetadataSource)
    assert outcome.status is CollectionStatus.SUCCEEDED
    assert outcome.output == CollectedOutput(
        {
            "schemaVersion": 2,
            "source": "network-manager-link-metadata",
            "sourceHealth": "ready",
            "observedAtMs": 10,
            "links": [
                {
                    "stableId": "link-a",
                    "kind": "ethernet",
                    "state": "up",
                    "connectivity": "full",
                    "carrier": True,
                    "metered": False,
                    "defaultRoute": True,
                    "signalPercent": None,
                    "counters": {
                        "receivedBytes": 100,
                        "transmittedBytes": 200,
                        "receivedErrors": 1,
                        "transmittedErrors": 2,
                        "receivedDrops": 3,
                        "transmittedDrops": 4,
                    },
                    "observedAtMs": 9,
                },
                {
                    "stableId": "link-m",
                    "kind": "ethernet",
                    "state": "degraded",
                    "connectivity": "full",
                    "carrier": True,
                    "metered": False,
                    "defaultRoute": True,
                    "signalPercent": None,
                    "counters": {
                        "receivedBytes": 100,
                        "transmittedBytes": 200,
                        "receivedErrors": 1,
                        "transmittedErrors": 2,
                        "receivedDrops": 3,
                        "transmittedDrops": 4,
                    },
                    "observedAtMs": 9,
                },
                {
                    "stableId": "link-z",
                    "kind": "wifi",
                    "state": "up",
                    "connectivity": "full",
                    "carrier": True,
                    "metered": False,
                    "defaultRoute": True,
                    "signalPercent": 80,
                    "counters": {
                        "receivedBytes": 100,
                        "transmittedBytes": 200,
                        "receivedErrors": 1,
                        "transmittedErrors": 2,
                        "receivedDrops": 3,
                        "transmittedDrops": 4,
                    },
                    "observedAtMs": 9,
                },
            ],
            "churn": {
                "added": [
                    {"stableId": "link-a", "kind": "ethernet"},
                    {"stableId": "link-m", "kind": "ethernet"},
                    {"stableId": "link-z", "kind": "wifi"},
                ],
                "removed": [],
                "changed": [],
            },
        }
    )
    encoded = json.dumps(outcome.output.payload)
    assert "must-not-escape" not in encoded
    assert "private" not in encoded
    assert all(
        set(item)
        == {
            "stableId",
            "kind",
            "state",
            "connectivity",
            "carrier",
            "metered",
            "defaultRoute",
            "signalPercent",
            "counters",
            "observedAtMs",
        }
        for item in outcome.output.payload["links"]
    )


def test_missing_consent_blocks_readiness_and_collection_before_source_access():
    class Source:
        def __init__(self):
            self.calls = 0

        async def readiness(self):
            self.calls += 1
            return CollectorReadiness(SourceStatus.READY, "ready", 1)

        async def snapshot(self):
            self.calls += 1
            return snapshot()

    source = Source()
    collector = NetworkMetadataCollector(
        source,
        permission_view("link-a", metadata=False),
        ("link-a",),
        clock_ms=lambda: 77,
    )

    assert asyncio.run(collector.readiness()) == CollectorReadiness(
        SourceStatus.UNAVAILABLE,
        "network metadata permission is not granted",
        77,
    )
    with pytest.raises(NetworkCollectionError) as excinfo:
        asyncio.run(collector.collect(trigger()))
    assert (excinfo.value.code, excinfo.value.detail) == (
        "permission-denied",
        "network metadata permission is not granted",
    )
    assert source.calls == 0


def test_allowlist_and_per_link_grants_filter_before_bounding_and_churn():
    collector = NetworkMetadataCollector(
        ReplayNetworkMetadataSource(
            [snapshot(link("link-a"), link("link-b"), link("link-unlisted"))]
        ),
        permission_view("link-a"),
        ("link-a", "link-b"),
    )

    payload = asyncio.run(collector.collect(trigger())).payload

    assert [item["stableId"] for item in payload["links"]] == ["link-a"]
    assert payload["churn"] == {
        "added": [{"stableId": "link-a", "kind": "ethernet"}],
        "removed": [],
        "changed": [],
    }


def test_empty_allowlist_emits_nothing_even_when_link_grant_exists():
    collector = NetworkMetadataCollector(
        ReplayNetworkMetadataSource([snapshot()]),
        permission_view("link-a"),
        (),
    )

    payload = asyncio.run(collector.collect(trigger())).payload

    assert payload["links"] == []
    assert payload["churn"] == {"added": [], "removed": [], "changed": []}


def test_network_churn_is_stable_complete_and_ignores_timestamps():
    first = snapshot(link("link-a"), link("link-b"), observed_at_ms=10)
    second = snapshot(
        link(
            "link-a",
            kind=NetworkLinkKind.WIFI,
            state=NetworkLinkState.DEGRADED,
            connectivity=NetworkConnectivity.PORTAL,
            carrier=False,
            metered=True,
            default_route=False,
            signal_percent=50,
            counters=counters(received_bytes=101),
            observed_at_ms=19,
        ),
        link("link-c", observed_at_ms=19),
        observed_at_ms=20,
    )
    collector = NetworkMetadataCollector(
        ReplayNetworkMetadataSource([first, second]),
        permission_view("link-a", "link-b", "link-c"),
        ("link-c", "link-b", "link-a"),
    )

    initial = asyncio.run(collector.collect(trigger())).payload
    changed = asyncio.run(collector.collect(trigger())).payload
    repeated = asyncio.run(collector.collect(trigger())).payload

    assert initial["churn"] == {
        "added": [
            {"stableId": "link-a", "kind": "ethernet"},
            {"stableId": "link-b", "kind": "ethernet"},
        ],
        "removed": [],
        "changed": [],
    }
    assert changed["churn"] == {
        "added": [{"stableId": "link-c", "kind": "ethernet"}],
        "removed": [{"stableId": "link-b", "kind": "ethernet"}],
        "changed": [
            {
                "stableId": "link-a",
                "kind": "wifi",
                "fields": [
                    "kind",
                    "state",
                    "connectivity",
                    "carrier",
                    "metered",
                    "defaultRoute",
                    "signalPercent",
                    "counters",
                ],
            }
        ],
    }
    assert repeated["churn"] == {"added": [], "removed": [], "changed": []}


def test_invalid_snapshot_does_not_replace_last_successful_churn_state():
    collector = NetworkMetadataCollector(
        ReplayNetworkMetadataSource(
            [
                snapshot(link("link-a")),
                snapshot(link("link-b"), observed_at_ms=-1),
                snapshot(links=(), observed_at_ms=30),
            ]
        ),
        permission_view("link-a", "link-b"),
        ("link-a", "link-b"),
    )
    asyncio.run(collector.collect(trigger()))

    with pytest.raises(NetworkCollectionError, match="source-invalid"):
        asyncio.run(collector.collect(trigger()))
    recovered = asyncio.run(collector.collect(trigger())).payload

    assert recovered["churn"] == {
        "added": [],
        "removed": [{"stableId": "link-a", "kind": "ethernet"}],
        "changed": [],
    }


def test_invalid_output_does_not_replace_last_successful_churn_state():
    second = snapshot(link("link-b"), observed_at_ms=20)
    collector = NetworkMetadataCollector(
        ReplayNetworkMetadataSource([snapshot(link("link-a")), second, second]),
        permission_view("link-a", "link-b"),
        ("link-a", "link-b"),
    )
    asyncio.run(collector.collect(trigger()))
    build_document = collector._output_document

    def invalid_document(*arguments):
        document = build_document(*arguments)
        document["schemaVersion"] = 1
        return document

    collector._output_document = invalid_document
    with pytest.raises(NetworkCollectionError) as caught:
        asyncio.run(collector.collect(trigger()))
    assert caught.value.code == "output-invalid"
    assert caught.value.detail == "schemaVersion: 2 was expected"

    collector._output_document = build_document
    recovered = asyncio.run(collector.collect(trigger())).payload
    assert recovered["churn"] == {
        "added": [{"stableId": "link-b", "kind": "ethernet"}],
        "removed": [{"stableId": "link-a", "kind": "ethernet"}],
        "changed": [],
    }


def test_revoked_link_is_forgotten_without_post_revocation_identity_output():
    class MutableGate:
        def __init__(self):
            self.grants = {
                NETWORK_METADATA_PERMISSION,
                network_link_permission("link-a"),
            }

        def allows(self, permission):
            return permission in self.grants

    gate = MutableGate()
    collector = NetworkMetadataCollector(
        ReplayNetworkMetadataSource([snapshot(), snapshot()]),
        gate,
        ("link-a",),
    )
    assert asyncio.run(collector.collect(trigger())).payload["links"]

    gate.grants.remove(network_link_permission("link-a"))
    after_revoke = asyncio.run(collector.collect(trigger())).payload

    assert after_revoke["links"] == []
    assert after_revoke["churn"] == {"added": [], "removed": [], "changed": []}


def test_every_allowlisted_link_is_emitted_and_tracked_until_it_really_disappears():
    """OMNI-0392: link-b is emitted alongside link-a, never silently omitted."""
    collector = NetworkMetadataCollector(
        ReplayNetworkMetadataSource(
            [
                snapshot(link("link-a"), link("link-b")),
                snapshot(link("link-a"), link("link-b", metered=True)),
                snapshot(link("link-a")),
            ]
        ),
        permission_view("link-a", "link-b"),
        ("link-a", "link-b"),
    )

    initial = asyncio.run(collector.collect(trigger())).payload
    changed = asyncio.run(collector.collect(trigger())).payload
    removed = asyncio.run(collector.collect(trigger())).payload

    assert [item["stableId"] for item in initial["links"]] == ["link-a", "link-b"]
    assert [item["stableId"] for item in initial["churn"]["added"]] == [
        "link-a",
        "link-b",
    ]
    assert [item["stableId"] for item in changed["churn"]["changed"]] == ["link-b"]
    assert [item["stableId"] for item in removed["churn"]["removed"]] == ["link-b"]


def test_more_links_than_the_old_default_are_all_emitted():
    """OMNI-0392 regression: 100 allowlisted links exceed the old default of 64."""
    identities = [f"link-{index:03d}" for index in range(100)]
    collector = NetworkMetadataCollector(
        ReplayNetworkMetadataSource(
            [snapshot(*(link(identity) for identity in identities))]
        ),
        permission_view(*identities),
        tuple(identities),
    )

    payload = asyncio.run(collector.collect(trigger())).payload

    assert [item["stableId"] for item in payload["links"]] == identities
    assert "truncatedLinks" not in payload


@pytest.mark.parametrize(
    ("behavior", "expected"),
    [
        (
            "ready",
            CollectorReadiness(
                SourceStatus.DEGRADED,
                "network metadata source is degraded",
                5,
            ),
        ),
        (
            "error",
            CollectorReadiness(
                SourceStatus.UNAVAILABLE,
                "network source readiness failed: RuntimeError",
                88,
            ),
        ),
        (
            "wrong-type",
            CollectorReadiness(
                SourceStatus.UNAVAILABLE,
                "network source readiness is invalid",
                88,
            ),
        ),
        (
            "bad-fields",
            CollectorReadiness(
                SourceStatus.UNAVAILABLE,
                "network source readiness is invalid",
                88,
            ),
        ),
    ],
)
def test_readiness_contains_source_failures_and_redacts_adapter_detail(behavior, expected):
    class Source:
        async def readiness(self):
            if behavior == "error":
                raise RuntimeError("SSID and address must stay private")
            if behavior == "wrong-type":
                return "ready"
            if behavior == "bad-fields":
                return CollectorReadiness("ready", "secret", -1)
            return CollectorReadiness(SourceStatus.DEGRADED, "private SSID", 5)

        async def snapshot(self):
            return snapshot()

    collector = NetworkMetadataCollector(Source(), permission_view(), (), clock_ms=lambda: 88)
    assert asyncio.run(collector.readiness()) == expected


def test_replay_source_advances_then_repeats_last_snapshot():
    first = snapshot(link("first"), observed_at_ms=10)
    second = snapshot(
        link("second", observed_at_ms=19),
        status=SourceStatus.DEGRADED,
        observed_at_ms=20,
    )
    source = ReplayNetworkMetadataSource([first, second])

    assert asyncio.run(source.readiness()).status is SourceStatus.READY
    assert asyncio.run(source.snapshot()) is first
    assert asyncio.run(source.readiness()) == CollectorReadiness(
        SourceStatus.DEGRADED,
        "network metadata source is degraded",
        20,
    )
    assert asyncio.run(source.snapshot()) is second
    assert asyncio.run(source.snapshot()) is second


def test_empty_or_invalid_replay_is_unavailable():
    source = ReplayNetworkMetadataSource([])
    assert asyncio.run(source.readiness()) == CollectorReadiness(
        SourceStatus.UNAVAILABLE,
        "network replay is empty",
        0,
    )
    with pytest.raises(NetworkCollectionError) as excinfo:
        asyncio.run(source.snapshot())
    assert str(excinfo.value) == "source-unavailable: network replay is empty"

    invalid = ReplayNetworkMetadataSource([snapshot(observed_at_ms=-1)])
    assert asyncio.run(invalid.readiness()).detail == "network replay is invalid"


@pytest.mark.parametrize("value", [None, 1, {}, "snapshots", b"snapshots"])
def test_replay_requires_a_sequence(value):
    with pytest.raises(TypeError, match="snapshots must be a sequence"):
        ReplayNetworkMetadataSource(value)


@pytest.mark.parametrize(
    ("source", "permissions", "allowlist", "clock", "reason"),
    [
        (
            object(),
            permission_view(),
            (),
            None,
            "source must implement NetworkMetadataSource",
        ),
        (
            ReplayNetworkMetadataSource([]),
            object(),
            (),
            None,
            "permissions must implement CollectionPermissionGate",
        ),
        (
            ReplayNetworkMetadataSource([]),
            permission_view(),
            "link-a",
            None,
            "allowed_link_ids must be a sequence",
        ),
        (
            ReplayNetworkMetadataSource([]),
            permission_view(),
            ("link-a", "link-a"),
            None,
            "allowed link identities must be unique",
        ),
        (
            ReplayNetworkMetadataSource([]),
            permission_view(),
            (["unhashable"],),
            None,
            "network link stable identity is invalid",
        ),
        (
            ReplayNetworkMetadataSource([]),
            permission_view(),
            tuple(f"link-{index}" for index in range(MAX_NETWORK_LINKS + 1)),
            None,
            f"at most {MAX_NETWORK_LINKS} link identities may be allowed",
        ),
        (
            ReplayNetworkMetadataSource([]),
            permission_view(),
            (),
            1,
            "clock_ms must be callable",
        ),
    ],
)
def test_collector_configuration_is_strict(source, permissions, allowlist, clock, reason):
    with pytest.raises((TypeError, ValueError)) as excinfo:
        NetworkMetadataCollector(
            source,
            permissions,
            allowlist,
            clock_ms=clock,
        )
    assert str(excinfo.value) == reason


def test_the_removed_link_emission_cap_is_not_silently_accepted():
    """OMNI-0392: the max_links ceiling is gone, not ignored."""
    with pytest.raises(TypeError):
        NetworkMetadataCollector(
            ReplayNetworkMetadataSource([]), permission_view(), (), max_links=8
        )


@pytest.mark.parametrize("stable_id", [None, 1, "", "Bad identity", "-bad", "bad-"])
def test_link_permission_rejects_invalid_stable_identity(stable_id):
    with pytest.raises(ValueError, match="network link stable identity is invalid"):
        network_link_permission(stable_id)


def test_link_permission_is_scoped_to_one_opaque_identity():
    assert network_link_permission("wifi.link-1") == (
        f"{NETWORK_LINK_PERMISSION_PREFIX}wifi.link-1"
    )


@pytest.mark.parametrize(
    ("candidate", "reason"),
    [
        (None, "network snapshot is not typed"),
        (snapshot(status="ready"), "network source status is invalid"),
        (snapshot(observed_at_ms=True), "network snapshot timestamp is invalid"),
        (snapshot(observed_at_ms=-1), "network snapshot timestamp is invalid"),
        (snapshot(links=[link()]), "network links must be a tuple"),
        (
            snapshot(links=tuple(link(f"link-{index}") for index in range(257))),
            f"network snapshot exceeds {MAX_NETWORK_LINKS} links",
        ),
        (snapshot(links=("bad",)), "network link sample is not typed"),
        (snapshot(link(stable_id="Bad identity")), "network link stable identity is invalid"),
        (snapshot(link(kind="wifi")), "network link kind is invalid: link-a"),
        (snapshot(link(state="up")), "network link state is invalid: link-a"),
        (
            snapshot(link(connectivity="full")),
            "network link connectivity is invalid: link-a",
        ),
        (snapshot(link(carrier=1)), "network link flags are invalid: link-a"),
        (snapshot(link(signal_percent=True)), "network link signal is invalid: link-a"),
        (snapshot(link(signal_percent=101)), "network link signal is invalid: link-a"),
        (snapshot(link(counters=None)), "network counters are not typed: link-a"),
        (
            snapshot(link(counters=counters(received_bytes=-1))),
            "network counters are invalid: link-a",
        ),
        (
            snapshot(link(counters=counters(transmitted_bytes=True))),
            "network counters are invalid: link-a",
        ),
        (
            snapshot(link(counters=counters(received_errors=MAX_NETWORK_COUNTER + 1))),
            "network counters are invalid: link-a",
        ),
        (
            snapshot(link(observed_at_ms=11)),
            "network link timestamp is invalid: link-a",
        ),
        (
            snapshot(link("same"), link("same")),
            "duplicate network link identity: same",
        ),
    ],
)
def test_snapshot_boundary_has_one_stable_failure(candidate, reason):
    assert network_snapshot_error(candidate) == reason


def test_collector_rejects_invalid_source_and_wrong_plugin_trigger():
    collector = NetworkMetadataCollector(
        ReplayNetworkMetadataSource([snapshot(observed_at_ms=-1)]),
        permission_view("link-a"),
        ("link-a",),
    )
    with pytest.raises(NetworkCollectionError) as source_error:
        asyncio.run(collector.collect(trigger()))
    assert source_error.value.code == "source-invalid"
    assert source_error.value.detail == "network snapshot timestamp is invalid"

    wrong = dataclasses.replace(trigger(), plugin_id="other-plugin")
    with pytest.raises(NetworkCollectionError) as trigger_error:
        asyncio.run(collector.collect(wrong))
    assert trigger_error.value.code == "trigger-invalid"


@given(
    stable_id=st.one_of(st.none(), st.integers(), st.text(max_size=100)),
    signal=st.one_of(st.none(), st.booleans(), st.integers()),
    received=st.one_of(st.none(), st.booleans(), st.integers()),
    observed=st.one_of(st.none(), st.booleans(), st.integers()),
)
def test_arbitrary_network_metadata_never_raises_unexpected_errors(
    stable_id, signal, received, observed
):
    candidate = snapshot(
        link(
            stable_id=stable_id,
            signal_percent=signal,
            counters=counters(received_bytes=received),
            observed_at_ms=observed,
        )
    )
    assert isinstance(network_snapshot_error(candidate), str)


@given(
    allowed=st.sets(st.sampled_from(("link-a", "link-b", "link-c"))),
    granted=st.sets(st.sampled_from(("link-a", "link-b", "link-c"))),
)
def test_arbitrary_allowlist_and_grant_intersections_never_leak(allowed, granted):
    collector = NetworkMetadataCollector(
        ReplayNetworkMetadataSource([snapshot(link("link-a"), link("link-b"), link("link-c"))]),
        permission_view(*granted),
        tuple(allowed),
    )

    payload = asyncio.run(collector.collect(trigger())).payload

    expected = sorted(allowed & granted)
    assert [item["stableId"] for item in payload["links"]] == expected
    assert [item["stableId"] for item in payload["churn"]["added"]] == expected
    assert "truncatedLinks" not in payload


def test_network_collector_documentation_freezes_privacy_and_bounds():
    guide = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "docs"
        / "network-peripherals.md"
    ).read_text(encoding="utf-8")
    assert "read:network-metadata" in guide
    assert "read:network-link/<stable-id>" in guide
    assert "does not capture packets, addresses, SSIDs" in guide
    assert "contains every allowlisted, granted" in guide
    assert "at most 256 identities" in guide
    assert "repeated replay snapshot produces empty churn" in guide
