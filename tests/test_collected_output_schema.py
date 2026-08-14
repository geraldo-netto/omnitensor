from __future__ import annotations

import copy

import pytest

from omnitensor.plugins.collection import COLLECTED_OUTPUT_SCHEMA
from omnitensor.registry import validate_document


def collected_document(source: str, items_key: str, truncated_key: str) -> dict:
    document = {
        "schemaVersion": 1,
        "source": source,
        "sourceHealth": "ready",
        "observedAtMs": 10,
        items_key: [{"id": "sample-a", "familyField": None}],
        "churn": {
            "added": [{"id": "sample-a"}],
            "removed": [],
            "changed": [{"id": "sample-a", "fields": ["state"]}],
        },
        truncated_key: 0,
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


@pytest.mark.parametrize(
    ("source", "items_key", "truncated_key"),
    [
        ("cgroup-systemd-pressure", "items", "truncatedItems"),
        ("cinnamon-window-metadata", "items", "truncatedItems"),
        ("hwmon-edac-power-service", "items", "truncatedItems"),
        ("smart-nvme-io", "items", "truncatedItems"),
        ("network-manager-link-metadata", "links", "truncatedLinks"),
        ("usb-bluetooth-health-metadata", "devices", "truncatedDevices"),
    ],
)
def test_canonical_schema_accepts_every_collector_family(
    source, items_key, truncated_key
):
    document = collected_document(source, items_key, truncated_key)

    assert validate_document(COLLECTED_OUTPUT_SCHEMA, document) == []


@pytest.mark.parametrize(
    "mutation",
    [
        lambda document: document.update(source="unknown-source"),
        lambda document: document.update(privatePayload="must never escape"),
        lambda document: document.update(items=document.pop("links")),
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
    document = collected_document(
        "network-manager-link-metadata", "links", "truncatedLinks"
    )
    changed = copy.deepcopy(document)
    mutation(changed)

    assert validate_document(COLLECTED_OUTPUT_SCHEMA, changed)
