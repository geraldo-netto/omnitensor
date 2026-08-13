"""Low-light configuration specification and migration route."""

from __future__ import annotations

from .plugins.settings import ConfigurationMigration, PluginConfigurationSpec
from .registry import load_schema

LOW_LIGHT_WORKLOAD_ID = "low-light-enhancement"
LOW_LIGHT_CONFIGURATION_VERSION = "0.2.0"
MAX_PATH_CHARACTERS = 4096


def _migrate_low_light_0_1_to_0_2(configuration: dict) -> dict:
    """Preserve the unchanged folder contract across the GPU profile migration."""
    return dict(configuration)


LOW_LIGHT_CONFIGURATION_SPEC = PluginConfigurationSpec(
    plugin_id=LOW_LIGHT_WORKLOAD_ID,
    plugin_version=LOW_LIGHT_CONFIGURATION_VERSION,
    schema=load_schema("low-light-configuration.schema.json"),
    defaults={"inputFolder": None, "outputFolder": None},
    migrations=(
        ConfigurationMigration(
            "0.1.0",
            LOW_LIGHT_CONFIGURATION_VERSION,
            _migrate_low_light_0_1_to_0_2,
        ),
    ),
)
