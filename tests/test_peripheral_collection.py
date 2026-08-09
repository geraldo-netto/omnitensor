from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.plugins import (
    MAX_PERIPHERAL_DEVICES,
    MAX_PERIPHERAL_ERROR_COUNT,
    MAX_PERIPHERAL_SOURCE_DEVICES,
    PERIPHERAL_METADATA_PERMISSION,
    CollectedOutput,
    CollectionStatus,
    Collector,
    CollectorReadiness,
    PeripheralBus,
    PeripheralClass,
    PeripheralCollectionError,
    PeripheralHealth,
    PeripheralMetadataCollector,
    PeripheralMetadataSource,
    PeripheralSample,
    PeripheralSnapshot,
    ReplayPeripheralMetadataSource,
    SourceStatus,
    Trigger,
    TriggerCoordinator,
    TriggerKind,
    peripheral_device_permission,
    peripheral_snapshot_error,
)
from omnitensor.sdk import PermissionView


def permission_view(*device_ids: str, metadata: bool = True) -> PermissionView:
    device_permissions = {peripheral_device_permission(stable_id) for stable_id in device_ids}
    declared = frozenset({PERIPHERAL_METADATA_PERMISSION, *device_permissions})
    granted = frozenset(
        {
            *device_permissions,
            *({PERIPHERAL_METADATA_PERMISSION} if metadata else set()),
        }
    )
    return PermissionView(declared, granted)


def device(stable_id: str = "device-a", **changes) -> PeripheralSample:
    values = {
        "stable_id": stable_id,
        "bus": PeripheralBus.USB,
        "device_class": PeripheralClass.INPUT,
        "health": PeripheralHealth.READY,
        "connected": True,
        "authorized": True,
        "paired": False,
        "trusted": False,
        "battery_percent": None,
        "error_count": 0,
        "observed_at_ms": 9,
    }
    values.update(changes)
    return PeripheralSample(**values)


def snapshot(*devices: PeripheralSample, **changes) -> PeripheralSnapshot:
    values = {
        "status": SourceStatus.READY,
        "observed_at_ms": 10,
        "devices": tuple(devices) or (device(),),
    }
    values.update(changes)
    return PeripheralSnapshot(**values)


def trigger(payload=None) -> Trigger:
    return Trigger(
        "network-peripherals",
        "peripheral-sample-1",
        TriggerKind.EVENT,
        payload or {},
        10,
    )


def collect(collector: PeripheralMetadataCollector) -> CollectedOutput:
    return asyncio.run(collector.collect(trigger()))


def test_peripheral_collector_filters_grants_and_runs_through_coordinator():
    source = ReplayPeripheralMetadataSource(
        [
            snapshot(
                device("device-z", bus=PeripheralBus.BLUETOOTH, paired=True),
                device("device-a"),
                device("device-m", health=PeripheralHealth.DEGRADED),
                device("device-unlisted"),
                device("device-ungranted"),
            )
        ]
    )
    permissions = permission_view("device-z", "device-a", "device-m")
    collector = PeripheralMetadataCollector(
        source,
        permissions,
        ("device-z", "device-a", "device-m", "device-ungranted"),
        max_devices=2,
    )
    coordinator = TriggerCoordinator()
    coordinator.register_collector("network-peripherals", collector)
    coordinator.submit(trigger({"bluetoothName": "private", "hidEvent": "secret"}))

    outcome = asyncio.run(coordinator.collect_next())

    assert isinstance(collector, Collector)
    assert isinstance(source, PeripheralMetadataSource)
    assert outcome.status is CollectionStatus.SUCCEEDED
    assert outcome.output == CollectedOutput(
        {
            "schemaVersion": 1,
            "source": "usb-bluetooth-health-metadata",
            "sourceHealth": "ready",
            "observedAtMs": 10,
            "devices": [
                {
                    "stableId": "device-a",
                    "bus": "usb",
                    "class": "input",
                    "health": "ready",
                    "connected": True,
                    "authorized": True,
                    "paired": False,
                    "trusted": False,
                    "batteryPercent": None,
                    "errorCount": 0,
                    "observedAtMs": 9,
                },
                {
                    "stableId": "device-m",
                    "bus": "usb",
                    "class": "input",
                    "health": "degraded",
                    "connected": True,
                    "authorized": True,
                    "paired": False,
                    "trusted": False,
                    "batteryPercent": None,
                    "errorCount": 0,
                    "observedAtMs": 9,
                },
            ],
            # Churn covers every eligible device, including the one past
            # max_devices whose metadata is not emitted (OMNI-0155).
            "churn": {
                "added": [
                    {"stableId": "device-a", "bus": "usb"},
                    {"stableId": "device-m", "bus": "usb"},
                    {"stableId": "device-z", "bus": "bluetooth"},
                ],
                "removed": [],
                "changed": [],
            },
            "truncatedDevices": 1,
        }
    )
    encoded = json.dumps(outcome.output.payload)
    assert "private" not in encoded
    assert "secret" not in encoded
    assert "device-unlisted" not in encoded
    assert "device-ungranted" not in encoded


def test_hotplug_churn_is_sorted_stable_and_ignores_timestamp_only_changes():
    first = snapshot(
        device("usb-b"),
        device(
            "bt-a",
            bus=PeripheralBus.BLUETOOTH,
            device_class=PeripheralClass.AUDIO,
            paired=True,
            trusted=True,
            battery_percent=80,
        ),
        observed_at_ms=10,
    )
    second = snapshot(
        device(
            "new-c",
            bus=PeripheralBus.BLUETOOTH,
            device_class=PeripheralClass.SENSOR,
            observed_at_ms=19,
        ),
        device(
            "bt-a",
            bus=PeripheralBus.BLUETOOTH,
            device_class=PeripheralClass.AUDIO,
            health=PeripheralHealth.DEGRADED,
            paired=True,
            trusted=False,
            battery_percent=75,
            error_count=2,
            observed_at_ms=19,
        ),
        observed_at_ms=20,
    )
    collector = PeripheralMetadataCollector(
        ReplayPeripheralMetadataSource([first, second]),
        permission_view("usb-b", "bt-a", "new-c"),
        ("new-c", "bt-a", "usb-b"),
    )

    initial = collect(collector).payload
    changed = collect(collector).payload
    repeated = collect(collector).payload

    assert initial["churn"] == {
        "added": [
            {"stableId": "bt-a", "bus": "bluetooth"},
            {"stableId": "usb-b", "bus": "usb"},
        ],
        "removed": [],
        "changed": [],
    }
    assert changed["churn"] == {
        "added": [{"stableId": "new-c", "bus": "bluetooth"}],
        "removed": [{"stableId": "usb-b", "bus": "usb"}],
        "changed": [
            {
                "stableId": "bt-a",
                "bus": "bluetooth",
                "fields": ["health", "trusted", "batteryPercent", "errorCount"],
            }
        ],
    }
    assert repeated["churn"] == {"added": [], "removed": [], "changed": []}


def test_all_public_device_fields_have_deterministic_change_names():
    old = device("device-a")
    new = device(
        "device-a",
        bus=PeripheralBus.BLUETOOTH,
        device_class=PeripheralClass.STORAGE,
        health=PeripheralHealth.UNAVAILABLE,
        connected=False,
        authorized=False,
        paired=True,
        trusted=True,
        battery_percent=1,
        error_count=1,
        observed_at_ms=19,
    )
    collector = PeripheralMetadataCollector(
        ReplayPeripheralMetadataSource([snapshot(old), snapshot(new, observed_at_ms=20)]),
        permission_view("device-a"),
        ("device-a",),
    )
    collect(collector)

    change = collect(collector).payload["churn"]["changed"]

    assert change == [
        {
            "stableId": "device-a",
            "bus": "bluetooth",
            "fields": [
                "bus",
                "class",
                "health",
                "connected",
                "authorized",
                "paired",
                "trusted",
                "batteryPercent",
                "errorCount",
            ],
        }
    ]


def test_revoked_device_is_forgotten_without_post_revocation_identity_output():
    class MutableGate:
        def __init__(self):
            self.grants = {
                PERIPHERAL_METADATA_PERMISSION,
                peripheral_device_permission("device-a"),
            }

        def allows(self, permission):
            return permission in self.grants

    gate = MutableGate()
    collector = PeripheralMetadataCollector(
        ReplayPeripheralMetadataSource([snapshot(), snapshot()]),
        gate,
        ("device-a",),
    )
    assert collect(collector).payload["devices"]

    gate.grants.remove(peripheral_device_permission("device-a"))
    after_revoke = collect(collector).payload

    assert after_revoke["devices"] == []
    assert after_revoke["churn"] == {"added": [], "removed": [], "changed": []}


def test_missing_metadata_consent_blocks_source_access():
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
    collector = PeripheralMetadataCollector(
        source,
        permission_view("device-a", metadata=False),
        ("device-a",),
        clock_ms=lambda: 77,
    )

    assert asyncio.run(collector.readiness()) == CollectorReadiness(
        SourceStatus.UNAVAILABLE,
        "peripheral metadata permission is not granted",
        77,
    )
    with pytest.raises(PeripheralCollectionError) as excinfo:
        collect(collector)
    assert (excinfo.value.code, excinfo.value.detail) == (
        "permission-denied",
        "peripheral metadata permission is not granted",
    )
    assert source.calls == 0


@pytest.mark.parametrize(
    ("behavior", "expected"),
    [
        (
            "ready",
            CollectorReadiness(
                SourceStatus.DEGRADED,
                "peripheral metadata source is degraded",
                5,
            ),
        ),
        (
            "unavailable",
            CollectorReadiness(
                SourceStatus.UNAVAILABLE,
                "peripheral metadata source is unavailable",
                5,
            ),
        ),
        (
            "error",
            CollectorReadiness(
                SourceStatus.UNAVAILABLE,
                "peripheral source readiness failed: RuntimeError",
                88,
            ),
        ),
        (
            "wrong-type",
            CollectorReadiness(
                SourceStatus.UNAVAILABLE,
                "peripheral source readiness is invalid",
                88,
            ),
        ),
        (
            "bad-fields",
            CollectorReadiness(
                SourceStatus.UNAVAILABLE,
                "peripheral source readiness is invalid",
                88,
            ),
        ),
    ],
)
def test_readiness_is_typed_and_redacts_adapter_details(behavior, expected):
    class Source:
        async def readiness(self):
            if behavior == "error":
                raise RuntimeError("device serial and name must stay private")
            if behavior == "wrong-type":
                return "ready"
            if behavior == "bad-fields":
                return CollectorReadiness("ready", "private name", -1)
            status = (
                SourceStatus.UNAVAILABLE if behavior == "unavailable" else SourceStatus.DEGRADED
            )
            return CollectorReadiness(status, "private device name", 5)

        async def snapshot(self):
            return snapshot()

    collector = PeripheralMetadataCollector(
        Source(),
        permission_view(),
        (),
        clock_ms=lambda: 88,
    )
    assert asyncio.run(collector.readiness()) == expected


def test_replay_advances_then_repeats_and_reports_health():
    first = snapshot(observed_at_ms=10)
    second = snapshot(
        device(observed_at_ms=19),
        status=SourceStatus.DEGRADED,
        observed_at_ms=20,
    )
    source = ReplayPeripheralMetadataSource([first, second])

    assert asyncio.run(source.readiness()) == CollectorReadiness(
        SourceStatus.READY,
        "peripheral metadata source is ready",
        10,
    )
    assert asyncio.run(source.snapshot()) is first
    assert asyncio.run(source.readiness()).status is SourceStatus.DEGRADED
    assert asyncio.run(source.snapshot()) is second
    assert asyncio.run(source.snapshot()) is second


def test_empty_or_invalid_replay_is_unavailable():
    source = ReplayPeripheralMetadataSource([])
    assert asyncio.run(source.readiness()) == CollectorReadiness(
        SourceStatus.UNAVAILABLE,
        "peripheral replay is empty",
        0,
    )
    with pytest.raises(PeripheralCollectionError) as excinfo:
        asyncio.run(source.snapshot())
    assert str(excinfo.value) == "source-unavailable: peripheral replay is empty"

    invalid = ReplayPeripheralMetadataSource([snapshot(observed_at_ms=-1)])
    assert asyncio.run(invalid.readiness()).detail == "peripheral replay is invalid"


@pytest.mark.parametrize("value", [None, 1, {}, "snapshots", b"snapshots"])
def test_replay_requires_a_sequence(value):
    with pytest.raises(TypeError, match="snapshots must be a sequence"):
        ReplayPeripheralMetadataSource(value)


@pytest.mark.parametrize(
    ("source", "permissions", "allowlist", "max_devices", "clock", "reason"),
    [
        (
            object(),
            permission_view(),
            (),
            1,
            None,
            "source must implement PeripheralMetadataSource",
        ),
        (
            ReplayPeripheralMetadataSource([]),
            object(),
            (),
            1,
            None,
            "permissions must implement CollectionPermissionGate",
        ),
        (
            ReplayPeripheralMetadataSource([]),
            permission_view(),
            "device-a",
            1,
            None,
            "allowed_device_ids must be a sequence",
        ),
        (
            ReplayPeripheralMetadataSource([]),
            permission_view(),
            ("device-a", "device-a"),
            1,
            None,
            "allowed device identities must be unique",
        ),
        (
            ReplayPeripheralMetadataSource([]),
            permission_view(),
            tuple(f"device-{index}" for index in range(MAX_PERIPHERAL_DEVICES + 1)),
            1,
            None,
            f"at most {MAX_PERIPHERAL_DEVICES} device identities may be allowed",
        ),
        (
            ReplayPeripheralMetadataSource([]),
            permission_view(),
            (),
            0,
            None,
            f"max_devices must be an integer from 1 to {MAX_PERIPHERAL_DEVICES}",
        ),
        (
            ReplayPeripheralMetadataSource([]),
            permission_view(),
            (),
            True,
            None,
            f"max_devices must be an integer from 1 to {MAX_PERIPHERAL_DEVICES}",
        ),
        (
            ReplayPeripheralMetadataSource([]),
            permission_view(),
            (),
            MAX_PERIPHERAL_DEVICES + 1,
            None,
            f"max_devices must be an integer from 1 to {MAX_PERIPHERAL_DEVICES}",
        ),
        (
            ReplayPeripheralMetadataSource([]),
            permission_view(),
            (),
            1,
            1,
            "clock_ms must be callable",
        ),
    ],
)
def test_collector_configuration_is_strict(
    source, permissions, allowlist, max_devices, clock, reason
):
    with pytest.raises((TypeError, ValueError)) as excinfo:
        PeripheralMetadataCollector(
            source,
            permissions,
            allowlist,
            max_devices=max_devices,
            clock_ms=clock,
        )
    assert str(excinfo.value) == reason


@pytest.mark.parametrize("stable_id", [None, 1, "", "Bad identity", "-bad", "bad-"])
def test_device_permission_rejects_invalid_stable_identity(stable_id):
    with pytest.raises(ValueError, match="peripheral stable identity is invalid"):
        peripheral_device_permission(stable_id)


def test_device_permission_is_scoped_to_one_opaque_identity():
    assert peripheral_device_permission("usb.keyboard-1") == (
        "read:peripheral-device/usb.keyboard-1"
    )


@pytest.mark.parametrize(
    ("candidate", "reason"),
    [
        (None, "peripheral snapshot is not typed"),
        (snapshot(status="ready"), "peripheral source status is invalid"),
        (snapshot(observed_at_ms=True), "peripheral snapshot timestamp is invalid"),
        (snapshot(observed_at_ms=-1), "peripheral snapshot timestamp is invalid"),
        (snapshot(devices=[device()]), "peripheral devices must be a tuple"),
        (
            snapshot(
                devices=tuple(
                    device(f"device-{index}") for index in range(MAX_PERIPHERAL_SOURCE_DEVICES + 1)
                )
            ),
            f"peripheral snapshot exceeds {MAX_PERIPHERAL_SOURCE_DEVICES} devices",
        ),
        (snapshot(devices=("bad",)), "peripheral sample is not typed"),
        (snapshot(device(stable_id="Bad identity")), "peripheral stable identity is invalid"),
        (snapshot(device(bus="usb")), "peripheral bus is invalid: device-a"),
        (snapshot(device(device_class="input")), "peripheral class is invalid: device-a"),
        (snapshot(device(health="ready")), "peripheral health is invalid: device-a"),
        (snapshot(device(connected=1)), "peripheral flags are invalid: device-a"),
        (snapshot(device(battery_percent=True)), "peripheral battery is invalid: device-a"),
        (snapshot(device(battery_percent=-1)), "peripheral battery is invalid: device-a"),
        (snapshot(device(battery_percent=101)), "peripheral battery is invalid: device-a"),
        (snapshot(device(error_count=True)), "peripheral error count is invalid: device-a"),
        (snapshot(device(error_count=-1)), "peripheral error count is invalid: device-a"),
        (
            snapshot(device(error_count=MAX_PERIPHERAL_ERROR_COUNT + 1)),
            "peripheral error count is invalid: device-a",
        ),
        (snapshot(device(observed_at_ms=11)), "peripheral timestamp is invalid: device-a"),
        (
            snapshot(device("same"), device("same")),
            "duplicate peripheral identity: same",
        ),
    ],
)
def test_snapshot_boundary_has_one_stable_failure(candidate, reason):
    assert peripheral_snapshot_error(candidate) == reason


def test_collector_rejects_invalid_source_and_wrong_plugin_trigger():
    collector = PeripheralMetadataCollector(
        ReplayPeripheralMetadataSource([snapshot(observed_at_ms=-1)]),
        permission_view("device-a"),
        ("device-a",),
    )
    with pytest.raises(PeripheralCollectionError) as source_error:
        collect(collector)
    assert source_error.value.code == "source-invalid"
    assert source_error.value.detail == "peripheral snapshot timestamp is invalid"

    wrong = dataclasses.replace(trigger(), plugin_id="other-plugin")
    with pytest.raises(PeripheralCollectionError) as trigger_error:
        asyncio.run(collector.collect(wrong))
    assert trigger_error.value.code == "trigger-invalid"


@given(
    stable_id=st.one_of(st.none(), st.integers(), st.text(max_size=100)),
    battery=st.one_of(st.none(), st.booleans(), st.integers()),
    errors=st.one_of(st.none(), st.booleans(), st.integers()),
    observed=st.one_of(st.none(), st.booleans(), st.integers()),
)
def test_arbitrary_peripheral_metadata_never_raises_unexpected_errors(
    stable_id, battery, errors, observed
):
    candidate = snapshot(
        device(
            stable_id=stable_id,
            battery_percent=battery,
            error_count=errors,
            observed_at_ms=observed,
        )
    )
    assert isinstance(peripheral_snapshot_error(candidate), str)


def test_peripheral_documentation_freezes_privacy_grants_churn_and_bounds():
    guide = (Path(__file__).resolve().parents[1] / "docs" / "network-peripherals.md").read_text(
        encoding="utf-8"
    )
    assert "read:peripheral-metadata" in guide
    assert "read:peripheral-device/<stable-id>" in guide
    assert "serials, Bluetooth addresses and names" in guide
    assert "32 devices by" in guide
    assert "repeated replay snapshot produces empty churn" in guide


def test_a_device_past_the_emission_limit_is_not_reported_as_removed():
    """Tracking only what fit made a still-attached device churn in and out."""
    source = ReplayPeripheralMetadataSource(
        [
            snapshot(device("device-a"), device("device-b")),
            snapshot(device("device-a"), device("device-b")),
        ]
    )
    collector = PeripheralMetadataCollector(
        source,
        permission_view("device-a", "device-b"),
        ("device-a", "device-b"),
        max_devices=1,
    )

    async def scenario():
        first = await collector.collect(trigger())
        second = await collector.collect(trigger())
        return first.payload, second.payload

    first, second = asyncio.run(scenario())

    assert [entry["stableId"] for entry in first["devices"]] == ["device-a"]
    assert first["truncatedDevices"] == 1
    assert [entry["stableId"] for entry in first["churn"]["added"]] == [
        "device-a",
        "device-b",
    ]
    assert second["churn"] == {"added": [], "removed": [], "changed": []}


def test_a_device_that_really_detaches_is_still_reported_as_removed():
    source = ReplayPeripheralMetadataSource(
        [
            snapshot(device("device-a"), device("device-b")),
            snapshot(device("device-a")),
        ]
    )
    collector = PeripheralMetadataCollector(
        source,
        permission_view("device-a", "device-b"),
        ("device-a", "device-b"),
        max_devices=1,
    )

    async def scenario():
        await collector.collect(trigger())
        return (await collector.collect(trigger())).payload

    second = asyncio.run(scenario())

    assert [entry["stableId"] for entry in second["churn"]["removed"]] == ["device-b"]


def test_a_truncated_device_still_reports_a_real_change():
    source = ReplayPeripheralMetadataSource(
        [
            snapshot(device("device-a"), device("device-b")),
            snapshot(device("device-a"), device("device-b", health=PeripheralHealth.DEGRADED)),
        ]
    )
    collector = PeripheralMetadataCollector(
        source,
        permission_view("device-a", "device-b"),
        ("device-a", "device-b"),
        max_devices=1,
    )

    async def scenario():
        await collector.collect(trigger())
        return (await collector.collect(trigger())).payload

    second = asyncio.run(scenario())

    assert [entry["stableId"] for entry in second["churn"]["changed"]] == ["device-b"]
