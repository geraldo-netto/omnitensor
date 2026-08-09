from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import sample_plugin_manifest

from omnitensor.inspection import (
    INVENTORY_SCHEMA,
    PLUGIN_INVENTORY_VERSION,
    ArtifactReadiness,
    artifact_readiness,
    build_plugin_inventory,
    plugin_inventory_entry,
    secret_configuration_keys,
)
from omnitensor.plugins.artifacts import ArtifactResolution
from omnitensor.plugins.discovery import PluginSource
from omnitensor.plugins.identity import ResolvedPlugin
from omnitensor.registry import validate_document

READ_SENSOR = "read:/sys/class/hwmon/*"
RUN_ACTION = "action:notify-user"


def resolved(
    plugin_id="sample-plugin",
    *,
    source=PluginSource.EXTERNAL,
    artifacts=(),
    permissions=(READ_SENSOR, RUN_ACTION),
    configuration=None,
):
    manifest = sample_plugin_manifest(plugin_id)
    manifest["plugin"]["artifacts"] = list(artifacts)
    manifest["plugin"]["permissions"] = list(permissions)
    if configuration is not None:
        manifest["plugin"]["schemas"]["configuration"] = configuration
    return ResolvedPlugin(
        plugin_id,
        manifest["version"],
        source,
        "sample-dist",
        "1.0.0",
        plugin_id,
        f"{plugin_id.replace('-', '_')}:Plugin",
        Path("/tmp/omnitensor-plugin.json"),
        manifest,
    )


def artifact(artifact_id="sample-model", version="1.2.3", fmt="onnx"):
    return {"id": artifact_id, "version": version, "format": fmt, "sha256": "a" * 64}


def ready(_artifact_id):
    return ArtifactResolution(True, Path("/models/sample.onnx"), "", 128)


def unavailable(_artifact_id):
    return ArtifactResolution(False, None, "artifact has no active version", 0)


def test_readiness_is_reported_for_every_declared_artifact():
    plugin = resolved(artifacts=[artifact(), artifact("second-model", "2.0.0", "ncnn")])

    readiness = artifact_readiness(plugin, ready)

    assert readiness == (
        ArtifactReadiness("sample-model", "1.2.3", "onnx", True, ""),
        ArtifactReadiness("second-model", "2.0.0", "ncnn", True, ""),
    )


def test_an_unavailable_artifact_carries_its_reason():
    plugin = resolved(artifacts=[artifact()])
    [readiness] = artifact_readiness(plugin, unavailable)
    assert readiness.ready is False
    assert readiness.reason == "artifact has no active version"


def test_an_unreadable_artifact_store_is_a_readiness_answer_not_an_error():
    """A store that cannot be read means not ready; saying so beats raising."""
    plugin = resolved(artifacts=[artifact()])

    def explode(_artifact_id):
        raise OSError("store is gone")

    [readiness] = artifact_readiness(plugin, explode)

    assert readiness.ready is False
    assert readiness.reason == "artifact store is unreadable: OSError"


def test_a_plugin_with_no_artifacts_reports_none():
    assert artifact_readiness(resolved(), ready) == ()


def test_secret_configuration_keys_are_named():
    schema = {
        "type": "object",
        "properties": {
            "token": {"type": "string", "x-omnitensor-secret": True},
            "mode": {"type": "string"},
            "api_key": {"type": "string", "x-omnitensor-secret": True},
        },
    }
    assert secret_configuration_keys(schema) == ("api_key", "token")


@pytest.mark.parametrize(
    "schema",
    [
        {},
        {"type": "object"},
        {"properties": []},
        "not-a-schema",
        {"properties": {"mode": "not-an-object"}},
        {"properties": {"mode": {"x-omnitensor-secret": "yes"}}},
    ],
)
def test_secret_key_extraction_tolerates_any_schema_shape(schema):
    assert secret_configuration_keys(schema) == ()


def test_an_entry_reports_declared_and_granted_permissions_separately():
    """Declared-but-ungranted is exactly the case a user must act on."""
    entry = plugin_inventory_entry(
        resolved(),
        resolve_artifact=ready,
        granted_permissions={READ_SENSOR},
    )

    assert entry["permissions"] == [
        {"name": RUN_ACTION, "granted": False},
        {"name": READ_SENSOR, "granted": True},
    ]


def test_an_entry_never_contains_configuration_values():
    schema = {
        "type": "object",
        "properties": {"token": {"type": "string", "x-omnitensor-secret": True}},
    }
    entry = plugin_inventory_entry(
        resolved(configuration=schema), resolve_artifact=ready
    )

    assert entry["configurationSchema"] == schema
    assert entry["secretConfigurationKeys"] == ["token"]
    assert "configuration" not in entry
    assert "$secretRef" not in json.dumps(entry)


def test_an_entry_carries_protocol_triggers_and_identity():
    entry = plugin_inventory_entry(resolved(), resolve_artifact=ready, worker_state="ready")

    assert entry["id"] == "sample-plugin"
    assert entry["source"] == "external"
    assert entry["distribution"] == "sample-dist"
    assert entry["workerState"] == "ready"
    assert entry["protocol"]["minimum"] >= 1
    assert entry["protocol"]["maximum"] >= entry["protocol"]["minimum"]
    assert entry["triggers"]


def test_the_inventory_is_contract_valid_and_ordered_by_plugin_id():
    plugins = [
        resolved("zeta-plugin", artifacts=[artifact()]),
        resolved("alpha-plugin"),
    ]

    document = build_plugin_inventory(
        plugins,
        resolve_artifact=unavailable,
        granted_permissions=lambda plugin_id: {READ_SENSOR}
        if plugin_id == "alpha-plugin"
        else set(),
        worker_states=lambda plugin_id: "ready" if plugin_id == "zeta-plugin" else None,
        generated_at_ms=1_700_000_000_000,
    )

    assert validate_document(INVENTORY_SCHEMA, document) == []
    assert document["version"] == PLUGIN_INVENTORY_VERSION
    assert [entry["id"] for entry in document["plugins"]] == [
        "alpha-plugin",
        "zeta-plugin",
    ]
    assert document["plugins"][1]["artifacts"][0]["ready"] is False
    assert document["plugins"][0]["workerState"] is None


def test_an_empty_inventory_is_still_contract_valid():
    document = build_plugin_inventory(
        [], resolve_artifact=ready, generated_at_ms=1_700_000_000_000
    )
    assert validate_document(INVENTORY_SCHEMA, document) == []
    assert document["plugins"] == []


def test_a_document_that_would_violate_the_contract_is_refused():
    """The inventory is a published contract, so it self-checks before it leaves."""
    with pytest.raises(ValueError, match="violates contract"):
        build_plugin_inventory(
            [resolved()], resolve_artifact=ready, generated_at_ms=0
        )


def test_building_the_inventory_never_imports_a_plugin(monkeypatch):
    import importlib

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the inventory must not import plugin modules")

    monkeypatch.setattr(importlib, "import_module", forbidden)
    document = build_plugin_inventory(
        [resolved()], resolve_artifact=ready, generated_at_ms=1_700_000_000_000
    )
    assert document["plugins"][0]["id"] == "sample-plugin"
