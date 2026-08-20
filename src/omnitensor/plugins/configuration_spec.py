"""The configuration contract an installed plugin declares in its manifest.

`PluginSettingsStore` holds a plugin's configuration against a
`PluginConfigurationSpec`, and until now the only spec that existed was
`lowlight_settings`'s, written by hand in this repository. An external plugin
cannot have one written for it here, so this module derives it from the
manifest the distribution already ships.

Two things the manifest cannot express are deliberate rather than missing.
Migrations are callables, and a manifest is JSON, so a derived spec has none:
a plugin whose configuration contract changes declares the change as a new
`configurationVersion`, and the settings it left behind refuse to load rather
than being transformed by a guess. The version itself is *not* the plugin's
release version. Binding it there would make `_migrate` raise
`migration-missing` on every upgrade — a plugin that fixed a typo would lose
the configuration somebody chose — so `schemas.configurationVersion` is
declared beside the schema it versions, and a release that leaves the contract
alone keeps the settings.
"""

from __future__ import annotations

from collections.abc import Mapping

from .protocol import JsonObject
from .settings import (
    CONFIGURATION_VERSION_PATTERN,
    PluginConfigurationSpec,
    PluginSettingsError,
)

DEFAULT_CONFIGURATION_VERSION = "1.0.0"


def manifest_configuration_spec(
    plugin_id: str,
    manifest: Mapping[str, object],
) -> PluginConfigurationSpec:
    """Build the configuration spec an installed plugin's manifest declares.

    Raises `PluginSettingsError` when the manifest declares a contract that
    cannot be held: a configuration block that is not an object, a version that
    is not semver, or a required property with no default — the last because a
    plugin whose defaults do not satisfy its own schema has no state a store
    could start from, and finding that out at the first `load` names the store
    rather than the manifest that is actually wrong.
    """
    declaration = manifest.get("plugin")
    if not isinstance(declaration, Mapping):
        raise PluginSettingsError(
            "configuration-undeclared",
            f"{plugin_id} declares no plugin block",
        )
    schemas = declaration.get("schemas")
    if not isinstance(schemas, Mapping):
        raise PluginSettingsError(
            "configuration-undeclared",
            f"{plugin_id} declares no schemas block",
        )
    schema = schemas.get("configuration")
    if not isinstance(schema, Mapping):
        raise PluginSettingsError(
            "invalid-schema",
            f"{plugin_id} configuration schema must be an object",
        )
    version = schemas.get("configurationVersion", DEFAULT_CONFIGURATION_VERSION)
    if not isinstance(version, str) or not CONFIGURATION_VERSION_PATTERN.fullmatch(version):
        raise PluginSettingsError(
            "invalid-version",
            f"{plugin_id} configuration version is invalid: {version!r}",
        )
    defaults = _declared_defaults(plugin_id, schema)
    return PluginConfigurationSpec(
        plugin_id=plugin_id,
        plugin_version=version,
        schema=dict(schema),
        defaults=defaults,
    )


def _declared_defaults(plugin_id: str, schema: Mapping[str, object]) -> JsonObject:
    """Collect the top-level `default` keywords as the starting configuration.

    Only the top level: a nested default is carried whole with the property it
    belongs to, because synthesising an object out of the defaults inside it
    would invent a value the manifest never states.
    """
    properties = schema.get("properties")
    defaults: JsonObject = {}
    if isinstance(properties, Mapping):
        for key, definition in properties.items():
            if isinstance(definition, Mapping) and "default" in definition:
                defaults[key] = definition["default"]
    required = schema.get("required")
    if isinstance(required, (list, tuple)):
        missing = [key for key in required if key not in defaults]
        if missing:
            raise PluginSettingsError(
                "configuration-defaults-missing",
                f"{plugin_id} requires {', '.join(map(str, missing))} and declares no default",
            )
    return defaults
