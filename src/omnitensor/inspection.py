"""A read-only inventory of what each installed plugin needs and has.

The applet has to tell a user why a profile is not doing anything: an artifact
that never downloaded, a permission never granted, a protocol the service
cannot speak.  All of that is knowable from installed metadata and the artifact
store, so the inventory is built without importing a single plugin module —
answering "what does this plugin require" must never mean running its code.

Two things are deliberately absent from the document.  Configuration *values*
never appear, only the declared schema, because a value can be a resolved
secret.  Secret-marked fields are named as such so a consumer renders a control
rather than a value.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass

from .plugins.artifacts import ArtifactResolution
from .plugins.identity import ResolvedPlugin
from .plugins.secrets import SECRET_SCHEMA_MARKER
from .registry import validate_document

PLUGIN_INVENTORY_VERSION = 1
INVENTORY_SCHEMA = "plugin-inventory.schema.json"


@dataclass(frozen=True, slots=True)
class ArtifactReadiness:
    """Whether one declared artifact can actually be executed right now."""

    artifact_id: str
    version: str
    format: str
    ready: bool
    reason: str


def artifact_readiness(
    plugin: ResolvedPlugin,
    resolve: Callable[[str], ArtifactResolution],
) -> tuple[ArtifactReadiness, ...]:
    """Report every declared artifact's readiness through the digest gate.

    ``resolve`` failing is itself a readiness answer, not an error: an artifact
    store that cannot be read means the artifact is not ready, and saying so is
    more useful than propagating the failure to the caller.
    """
    readiness: list[ArtifactReadiness] = []
    for declared in plugin.manifest["plugin"]["artifacts"]:
        try:
            resolution = resolve(declared["id"])
        except Exception as error:  # noqa: BLE001 - store failures are arbitrary
            readiness.append(
                ArtifactReadiness(
                    declared["id"],
                    declared["version"],
                    declared["format"],
                    False,
                    f"artifact store is unreadable: {type(error).__name__}",
                )
            )
            continue
        readiness.append(
            ArtifactReadiness(
                declared["id"],
                declared["version"],
                declared["format"],
                bool(resolution.ready),
                "" if resolution.ready else resolution.reason,
            )
        )
    return tuple(readiness)


def secret_configuration_keys(schema: object) -> tuple[str, ...]:
    """Top-level configuration keys the plugin declares as secrets."""
    if not isinstance(schema, dict):
        return ()
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return ()
    return tuple(
        sorted(
            key
            for key, definition in properties.items()
            if isinstance(definition, dict) and definition.get(SECRET_SCHEMA_MARKER) is True
        )
    )


def plugin_inventory_entry(
    plugin: ResolvedPlugin,
    *,
    resolve_artifact: Callable[[str], ArtifactResolution],
    granted_permissions: Collection[str] = (),
    worker_state: str | None = None,
) -> dict:
    """One plugin's capabilities, readiness, schema, and permission state."""
    declaration = plugin.manifest["plugin"]
    declared = list(declaration["permissions"])
    granted = set(granted_permissions)
    configuration_schema = declaration["schemas"]["configuration"]
    return {
        "id": plugin.plugin_id,
        "version": plugin.version,
        "source": str(plugin.source),
        "distribution": plugin.distribution_name,
        "workerState": worker_state,
        "protocol": {
            "minimum": declaration["protocol"]["minimum"],
            "maximum": declaration["protocol"]["maximum"],
            "capabilities": sorted(declaration["protocol"].get("capabilities", [])),
        },
        "triggers": sorted(declaration["triggers"]),
        "artifacts": [
            {
                "id": item.artifact_id,
                "version": item.version,
                "format": item.format,
                "ready": item.ready,
                "reason": item.reason,
            }
            for item in artifact_readiness(plugin, resolve_artifact)
        ],
        "permissions": [
            # Declared but ungranted is the case a user needs to act on, so
            # both states are always present rather than only the active set.
            {"name": permission, "granted": permission in granted}
            for permission in sorted(declared)
        ],
        "configurationSchema": configuration_schema,
        "secretConfigurationKeys": list(secret_configuration_keys(configuration_schema)),
    }


def build_plugin_inventory(
    plugins: Sequence[ResolvedPlugin],
    *,
    resolve_artifact: Callable[[str], ArtifactResolution],
    granted_permissions: Callable[[str], Collection[str]] = lambda _plugin_id: (),
    worker_states: Callable[[str], str | None] = lambda _plugin_id: None,
    generated_at_ms: int,
) -> dict:
    """Build the contract-valid inventory document, in stable plugin order."""
    document = {
        "version": PLUGIN_INVENTORY_VERSION,
        "generatedAt": generated_at_ms,
        "plugins": [
            plugin_inventory_entry(
                plugin,
                resolve_artifact=resolve_artifact,
                granted_permissions=granted_permissions(plugin.plugin_id),
                worker_state=worker_states(plugin.plugin_id),
            )
            for plugin in sorted(plugins, key=lambda item: item.plugin_id)
        ],
    }
    violations = validate_document(INVENTORY_SCHEMA, document)
    if violations:
        raise ValueError(f"plugin inventory violates contract: {violations[0]}")
    return document
