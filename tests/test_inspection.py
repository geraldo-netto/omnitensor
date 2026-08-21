from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import sample_plugin_manifest
from hypothesis import given
from hypothesis import strategies as st

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
from omnitensor.plugins.settings import PluginSettings
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


def test_an_entry_given_no_stored_settings_publishes_no_configuration():
    """Absent, never null: a plugin with no configuration contract has no
    stored values and no revision to report."""
    schema = {
        "type": "object",
        "properties": {"token": {"type": "string", "x-omnitensor-secret": True}},
    }
    entry = plugin_inventory_entry(resolved(configuration=schema), resolve_artifact=ready)

    assert entry["configurationSchema"] == schema
    assert entry["secretConfigurationKeys"] == ["token"]
    assert "configuration" not in entry
    assert "settingsRevision" not in entry
    assert "$secretRef" not in json.dumps(entry)


SECRET_SCHEMA = {
    "type": "object",
    "properties": {
        "token": {"type": "string", "x-omnitensor-secret": True},
        "mode": {"type": "string"},
    },
}


def stored(configuration, revision=1, plugin_id="sample-plugin"):
    return PluginSettings(plugin_id, "1.0.0", revision, configuration)


def test_an_entry_publishes_the_manifests_input_schema_verbatim():
    plugin = resolved()
    entry = plugin_inventory_entry(plugin, resolve_artifact=ready)

    assert entry["inputSchema"] == plugin.manifest["plugin"]["schemas"]["input"]
    # A copy, not an alias: mutating the document must not edit the manifest.
    assert entry["inputSchema"] is not plugin.manifest["plugin"]["schemas"]["input"]


def test_an_entry_carries_the_stored_configuration_and_its_revision():
    entry = plugin_inventory_entry(
        resolved(configuration=SECRET_SCHEMA),
        resolve_artifact=ready,
        settings=stored({"mode": "fast"}, revision=4),
    )

    assert entry["configuration"] == {"mode": "fast"}
    assert entry["settingsRevision"] == 4


def test_a_secret_marked_value_never_reaches_the_wire_even_when_stored():
    entry = plugin_inventory_entry(
        resolved(configuration=SECRET_SCHEMA),
        resolve_artifact=ready,
        settings=stored({"mode": "fast", "token": {"$secretRef": "keyring:sample"}}),
    )

    assert entry["configuration"] == {"mode": "fast"}
    assert "token" not in entry["configuration"]
    assert "$secretRef" not in json.dumps(entry)


@given(
    values=st.dictionaries(
        st.text(min_size=1, max_size=12),
        st.none() | st.booleans() | st.integers() | st.text(max_size=8),
        max_size=8,
    ),
    secret_keys=st.sets(st.text(min_size=1, max_size=12), max_size=4),
)
def test_the_published_configuration_is_exactly_the_non_secret_subset(values, secret_keys):
    """The withholding boundary: every secret-marked key is withheld, every
    other stored value is published verbatim."""
    schema = {
        "type": "object",
        "properties": {key: {"x-omnitensor-secret": True} for key in secret_keys},
    }
    entry = plugin_inventory_entry(
        resolved(configuration=schema),
        resolve_artifact=ready,
        settings=stored(values),
    )

    assert entry["configuration"] == {
        key: value for key, value in values.items() if key not in secret_keys
    }
    assert entry["settingsRevision"] == 1


def test_the_inventory_carries_each_plugins_settings_revision():
    """Revision propagation, and absence where no contract is declared."""
    document = build_plugin_inventory(
        [resolved("alpha-plugin"), resolved("zeta-plugin")],
        resolve_artifact=ready,
        stored_settings=lambda plugin_id: (
            stored({"mode": "fast"}, revision=2, plugin_id=plugin_id)
            if plugin_id == "alpha-plugin"
            else None
        ),
        generated_at_ms=1_700_000_000_000,
    )

    assert validate_document(INVENTORY_SCHEMA, document) == []
    alpha, zeta = document["plugins"]
    assert (alpha["configuration"], alpha["settingsRevision"]) == ({"mode": "fast"}, 2)
    assert "configuration" not in zeta
    assert "settingsRevision" not in zeta
    assert alpha["inputSchema"] == {"type": "object", "additionalProperties": False}


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
        granted_permissions=lambda plugin_id: (
            {READ_SENSOR} if plugin_id == "alpha-plugin" else set()
        ),
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
    document = build_plugin_inventory([], resolve_artifact=ready, generated_at_ms=1_700_000_000_000)
    assert validate_document(INVENTORY_SCHEMA, document) == []
    assert document["plugins"] == []


def test_a_document_that_would_violate_the_contract_is_refused():
    """The inventory is a published contract, so it self-checks before it leaves."""
    with pytest.raises(ValueError, match="violates contract"):
        build_plugin_inventory([resolved()], resolve_artifact=ready, generated_at_ms=0)


def test_building_the_inventory_never_imports_a_plugin(monkeypatch):
    import importlib

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the inventory must not import plugin modules")

    monkeypatch.setattr(importlib, "import_module", forbidden)
    document = build_plugin_inventory(
        [resolved()], resolve_artifact=ready, generated_at_ms=1_700_000_000_000
    )
    assert document["plugins"][0]["id"] == "sample-plugin"
