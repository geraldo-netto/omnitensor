"""A read-only inventory of what each installed plugin needs and has.

The applet has to tell a user why a profile is not doing anything: an artifact
that never downloaded, a permission never granted, a protocol the service
cannot speak.  All of that is knowable from installed metadata and the artifact
store, so the inventory is built without importing a single plugin module —
answering "what does this plugin require" must never mean running its code.

Stored configuration values are published so a consumer can show what a
workload actually runs on — with every key named in the secret-marked set
withheld, because a stored value can be a resolved secret.  Secret-marked
fields are named as such so a consumer renders a control rather than a value;
their values never reach the wire.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass

from .plugins.artifacts import ArtifactResolution
from .plugins.identity import ResolvedPlugin
from .plugins.secrets import SECRET_SCHEMA_MARKER
from .plugins.settings import PluginSettings
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
    # Whether a person may choose this as the workload's model. A manifest
    # pins every artifact the workload uses, and a client building a dropdown
    # from the list without this offers the embedder behind a question.
    selectable: bool = True


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
        # Absent means selectable, so a manifest written before the field
        # keeps the meaning it had.
        selectable = declared.get("selectable", True) is not False
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
                    selectable,
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
                selectable,
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
    worker_detail: str | None = None,
    settings: PluginSettings | None = None,
) -> dict:
    """One plugin's capabilities, readiness, schema, and permission state."""
    declaration = plugin.manifest["plugin"]
    declared = list(declaration["permissions"])
    granted = set(granted_permissions)
    configuration_schema = declaration["schemas"]["configuration"]
    input_schema = declaration["schemas"].get("input")
    secret_keys = secret_configuration_keys(configuration_schema)
    return {
        "id": plugin.plugin_id,
        "version": plugin.version,
        "source": str(plugin.source),
        "distribution": plugin.distribution_name,
        "workerState": worker_state,
        # Present only when the runtime has something to say: the applet's
        # record is closed, so an always-present null would be a field an
        # older applet rejects, taking the whole catalog down with it.
        **({"workerDetail": worker_detail[:300]} if worker_detail else {}),
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
                "selectable": item.selectable,
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
        "secretConfigurationKeys": list(secret_keys),
        # Verbatim from the manifest, so a consumer can build the payload a
        # job submission is validated against. Optional, like workerDetail:
        # an entry that always carried it would be a field an older consumer
        # rejects.
        **({"inputSchema": copy.deepcopy(input_schema)} if isinstance(input_schema, dict) else {}),
        # The stored values this workload actually runs on, with every
        # secret-marked key withheld — a stored value can be a resolved
        # secret reference. Absent, never null, when the manifest declares
        # no configuration contract.
        **(
            {
                "configuration": {
                    key: copy.deepcopy(value)
                    for key, value in settings.configuration.items()
                    if key not in secret_keys
                },
                "settingsRevision": settings.revision,
            }
            if settings is not None
            else {}
        ),
    }


def build_plugin_inventory(
    plugins: Sequence[ResolvedPlugin],
    *,
    resolve_artifact: Callable[[str], ArtifactResolution],
    granted_permissions: Callable[[str], Collection[str]] = lambda _plugin_id: (),
    worker_states: Callable[[str], str | None] = lambda _plugin_id: None,
    worker_details: Callable[[str], str | None] = lambda _plugin_id: None,
    stored_settings: Callable[[str], PluginSettings | None] = lambda _plugin_id: None,
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
                worker_detail=worker_details(plugin.plugin_id),
                settings=stored_settings(plugin.plugin_id),
            )
            for plugin in sorted(plugins, key=lambda item: item.plugin_id)
        ],
    }
    violations = validate_document(INVENTORY_SCHEMA, document)
    if violations:
        raise ValueError(f"plugin inventory violates contract: {violations[0]}")
    return document
