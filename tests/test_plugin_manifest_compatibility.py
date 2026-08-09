from __future__ import annotations

from pathlib import Path

import pytest
from conftest import sample_manifest, sample_plugin_manifest

from omnitensor.plugins import (
    CURRENT_MANIFEST_VERSION,
    MAX_PROTOCOL_CAPABILITIES,
    MAX_PROTOCOL_CAPABILITY_CHARS,
    ManifestCompatibilityCode,
    PluginCatalog,
    PluginManifestCompatibilityGate,
    PluginRejection,
    PluginRejectionCode,
    PluginSource,
    ResolvedPlugin,
    adapt_bundled_manifest_v1,
    resolve_plugin_compatibility,
)
from omnitensor.registry import validate_document
from omnitensor.state import PolicyState, PolicyStore, ProfilePolicy


def resolved(manifest, source=PluginSource.EXTERNAL):
    plugin_id = manifest["id"]
    return ResolvedPlugin(
        plugin_id,
        manifest["version"],
        source,
        "omnitensor" if source is PluginSource.BUNDLED else "external-dist",
        None if source is PluginSource.BUNDLED else "1.0.0",
        plugin_id,
        None if source is PluginSource.BUNDLED else "package:Plugin",
        Path(f"/{plugin_id}/manifest.json"),
        manifest,
    )


def test_bundled_v1_adapter_is_schema_valid_fail_closed_and_non_mutating():
    original = sample_manifest("legacy-plugin")
    before = original.copy()

    adapted = adapt_bundled_manifest_v1(original)

    assert original == before
    assert adapted["manifestVersion"] == CURRENT_MANIFEST_VERSION
    assert adapted["plugin"] == {
        "entryPoint": "legacy-plugin",
        "protocol": {"minimum": 1, "maximum": 1, "capabilities": []},
        "schemas": {
            "configuration": {"type": "object", "additionalProperties": False},
            "input": {"type": "object", "additionalProperties": False},
            "output": {"type": "object", "additionalProperties": False},
        },
        "triggers": ["manual"],
        "artifacts": [],
        "permissions": [],
    }
    for field in ("id", "version", "capabilities", "requirements", "ui", "defaults", "pipeline"):
        assert adapted[field] == original[field]
    assert validate_document("workload-manifest.schema.json", adapted) == []


def test_gate_adapts_only_bundled_v1_and_preserves_policy_fields():
    manifest = sample_manifest("legacy-plugin")
    decision = PluginManifestCompatibilityGate().check(
        resolved(manifest, PluginSource.BUNDLED)
    )

    assert decision.compatible is True
    assert decision.code is ManifestCompatibilityCode.ADAPTED_V1
    assert decision.detail == "bundled manifest v1 adapted without changing policy fields"
    assert decision.protocol_version == 1
    assert decision.capabilities == frozenset()
    assert decision.adapted is True
    assert decision.plugin.manifest["defaults"] == manifest["defaults"]

    external = PluginManifestCompatibilityGate().check(resolved(manifest))
    assert external.compatible is False
    assert external.code is ManifestCompatibilityCode.MANIFEST_VERSION
    assert external.detail == "manifest v1 migration is limited to bundled plugins"


def test_gate_negotiates_highest_protocol_and_capability_intersection():
    manifest = sample_plugin_manifest("modern-plugin")
    manifest["plugin"]["protocol"] = {
        "minimum": 1,
        "maximum": 4,
        "capabilities": ["cancel", "progress"],
    }
    decision = PluginManifestCompatibilityGate(
        protocol_minimum=2,
        protocol_maximum=3,
        capabilities=frozenset({"cancel", "health"}),
    ).check(resolved(manifest))

    assert decision.compatible is True
    assert decision.code is ManifestCompatibilityCode.COMPATIBLE
    assert decision.detail == "manifest and runtime protocols are compatible"
    assert decision.protocol_version == 3
    assert decision.capabilities == frozenset({"cancel"})
    assert decision.adapted is False
    assert decision.plugin is not None
    assert decision.plugin is not resolved(manifest)
    assert decision.plugin.manifest == manifest
    assert decision.plugin.manifest is not manifest


@pytest.mark.parametrize(
    "change, code, detail",
    [
        (
            lambda manifest: manifest.update(manifestVersion=3),
            ManifestCompatibilityCode.MANIFEST_VERSION,
            "runtime requires manifest version 2",
        ),
        (
            lambda manifest: manifest["requirements"].update(runtimeApi=2),
            ManifestCompatibilityCode.RUNTIME_API,
            "runtime API 2 is incompatible with 1",
        ),
        (
            lambda manifest: manifest["plugin"]["protocol"].update(
                minimum=2, maximum=4
            ),
            ManifestCompatibilityCode.PROTOCOL_VERSION,
            "plugin protocol 2-4 does not overlap runtime 1-1",
        ),
        (
            lambda manifest: manifest.pop("ui"),
            ManifestCompatibilityCode.ADAPTER_INVALID,
            "effective manifest violates the runtime schema",
        ),
    ],
)
def test_gate_rejects_incompatible_manifests_with_stable_reasons(change, code, detail):
    manifest = sample_plugin_manifest("incompatible-plugin")
    change(manifest)

    decision = PluginManifestCompatibilityGate().check(resolved(manifest))

    assert decision.compatible is False
    assert decision.code is code
    assert decision.detail == detail
    assert decision.plugin is None
    assert decision.protocol_version is None
    assert decision.capabilities == frozenset()
    assert decision.adapted is False


@pytest.mark.parametrize(
    "arguments, error, detail",
    [
        ({"runtime_api": 0}, ValueError, "runtime_api must be a positive integer"),
        ({"runtime_api": True}, ValueError, "runtime_api must be a positive integer"),
        (
            {"protocol_minimum": 2, "protocol_maximum": 1},
            ValueError,
            "protocol range must be ordered",
        ),
        ({"capabilities": set()}, TypeError, "capabilities must be a frozenset"),
        (
            {"capabilities": frozenset({"x" * (MAX_PROTOCOL_CAPABILITY_CHARS + 1)})},
            ValueError,
            f"capabilities must contain at most {MAX_PROTOCOL_CAPABILITIES}",
        ),
        (
            {
                "capabilities": frozenset(
                    str(index) for index in range(MAX_PROTOCOL_CAPABILITIES + 1)
                )
            },
            ValueError,
            f"capabilities must contain at most {MAX_PROTOCOL_CAPABILITIES}",
        ),
    ],
)
def test_gate_configuration_is_bounded(arguments, error, detail):
    with pytest.raises(error, match=detail):
        PluginManifestCompatibilityGate(**arguments)


def test_catalog_resolution_preserves_identity_rejections_and_adds_compatibility():
    compatible = resolved(sample_plugin_manifest("compatible"))
    incompatible_manifest = sample_plugin_manifest("incompatible")
    incompatible_manifest["requirements"]["runtimeApi"] = 9
    incompatible = resolved(incompatible_manifest)
    identity_rejection = PluginRejection(
        "broken", "broken-dist", PluginRejectionCode.MANIFEST_INVALID, "fixture"
    )

    catalog = resolve_plugin_compatibility(
        PluginCatalog((incompatible, compatible), (identity_rejection,))
    )

    assert [plugin.plugin_id for plugin in catalog.plugins] == ["compatible"]
    assert catalog.rejections[0] == identity_rejection
    assert catalog.rejections[1] == PluginRejection(
        "incompatible",
        "external-dist",
        PluginRejectionCode.INCOMPATIBLE,
        "runtime-api-incompatible: runtime API 9 is incompatible with 1",
    )


def test_manifest_migration_does_not_change_persisted_policy_state(tmp_path):
    manifest = sample_manifest("legacy-plugin")
    defaults = {"legacy-plugin": ProfilePolicy(True, 2)}
    store = PolicyStore(tmp_path / "policy.json", defaults)
    expected = PolicyState(
        paused=True,
        profiles={"legacy-plugin": ProfilePolicy(False, 4)},
        revision=7,
    )
    store.save(expected)

    decision = PluginManifestCompatibilityGate().check(
        resolved(manifest, PluginSource.BUNDLED)
    )

    assert decision.compatible is True
    assert store.load() == expected
    assert manifest["defaults"] == {"enabled": True, "weight": 2}
