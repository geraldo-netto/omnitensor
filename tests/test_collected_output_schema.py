from __future__ import annotations

import copy

import pytest

from omnitensor.plugins.collection import COLLECTED_OUTPUT_SCHEMA
from omnitensor.registry import validate_document


def collected_document(source: str, items_key: str) -> dict:
    document = {
        "schemaVersion": 2,
        "source": source,
        "sourceHealth": "ready",
        "observedAtMs": 10,
        items_key: [{"id": "sample-a", "familyField": None}],
        "churn": {
            "added": [{"id": "sample-a"}],
            "removed": [],
            "changed": [{"id": "sample-a", "fields": ["state"]}],
        },
    }
    if source == "cgroup-systemd-pressure":
        document.update(
            {
                "kernelTelemetry": {
                    "version": 1,
                    "state": "helper-absent",
                    "detail": "helper is not installed",
                    "collectedAtMs": 0,
                    "histograms": [],
                    "counters": [],
                },
                "kernelFeatures": {},
            }
        )
    return document


FAMILIES = [
    ("cgroup-systemd-pressure", "items"),
    ("cinnamon-window-metadata", "items"),
    ("hwmon-edac-power-service", "items"),
    ("smart-nvme-io", "items"),
    ("network-manager-link-metadata", "links"),
    ("usb-bluetooth-health-metadata", "devices"),
]


@pytest.mark.parametrize(("source", "items_key"), FAMILIES)
def test_canonical_schema_accepts_every_collector_family(source, items_key):
    document = collected_document(source, items_key)

    assert validate_document(COLLECTED_OUTPUT_SCHEMA, document) == []


@pytest.mark.parametrize(("source", "items_key"), FAMILIES)
def test_canonical_schema_accepts_the_full_protocol_item_bound(source, items_key):
    """OMNI-0392 regression: every allowlisted item is emitted, up to maxItems."""
    document = collected_document(source, items_key)
    document[items_key] = [{"id": f"sample-{index}"} for index in range(1024)]

    assert validate_document(COLLECTED_OUTPUT_SCHEMA, document) == []

    document[items_key].append({"id": "sample-overflow"})
    assert validate_document(COLLECTED_OUTPUT_SCHEMA, document)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda document: document.update(source="unknown-source"),
        lambda document: document.update(privatePayload="must never escape"),
        lambda document: document.update(items=document.pop("links")),
        lambda document: document.update(schemaVersion=1),
        lambda document: document.update(truncatedLinks=0),
        lambda document: document.update(truncatedItems=0),
        lambda document: document.update(truncatedDevices=0),
        lambda document: document.update(
            kernelTelemetry={
                "version": 1,
                "state": "ready",
                "detail": "",
                "collectedAtMs": 1,
                "histograms": [],
                "counters": [],
            }
        ),
    ],
)
def test_canonical_schema_rejects_wrong_family_and_extension_shapes(mutation):
    document = collected_document("network-manager-link-metadata", "links")
    changed = copy.deepcopy(document)
    mutation(changed)

    assert validate_document(COLLECTED_OUTPUT_SCHEMA, changed)
