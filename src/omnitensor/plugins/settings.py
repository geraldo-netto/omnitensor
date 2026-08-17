"""Revisioned, schema-validated plugin configuration persistence."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import jsonschema

from ..atomicio import JsonTooLargeError, read_json_bounded, write_json_atomic
from ..storelock import store_lock
from .protocol import JsonObject
from .secrets import (
    SecretConfigurationError,
    persisted_configuration_errors,
    validate_secret_references,
)

SETTINGS_DOCUMENT_VERSION = 1
SETTINGS_LOCK_FILE = ".plugin-settings.lock"
DEFAULT_MAX_SETTINGS_BYTES = 256 * 1024
MAX_MIGRATIONS = 32
_PLUGIN_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_VERSION = re.compile(r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$")


class PluginSettingsError(ValueError):
    """Stable configuration or persistence-contract rejection."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class ConfigurationMigration:
    """One explicit, deterministic configuration-version edge."""

    source_version: str
    target_version: str
    transform: Callable[[JsonObject], JsonObject]


@dataclass(frozen=True, slots=True)
class PluginConfigurationSpec:
    """Current plugin configuration contract and upgrade route."""

    plugin_id: str
    plugin_version: str
    schema: Mapping[str, object]
    defaults: JsonObject
    migrations: tuple[ConfigurationMigration, ...] = ()


@dataclass(frozen=True, slots=True)
class PluginSettings:
    plugin_id: str
    plugin_version: str
    revision: int
    configuration: JsonObject


class PluginSettingsStore:
    """Load, migrate, and compare-and-swap per-plugin settings documents."""

    def __init__(
        self,
        root: Path,
        *,
        max_settings_bytes: int = DEFAULT_MAX_SETTINGS_BYTES,
    ):
        if (
            isinstance(max_settings_bytes, bool)
            or not isinstance(max_settings_bytes, int)
            or max_settings_bytes < 1
        ):
            raise ValueError("max_settings_bytes must be a positive integer")
        self._root = Path(root)
        self._max_settings_bytes = max_settings_bytes

    def load(self, spec: PluginConfigurationSpec) -> PluginSettings:
        """Load current settings, atomically committing any required migration."""
        with store_lock(self._root, SETTINGS_LOCK_FILE):
            return self._load_locked(spec)

    def _load_locked(self, spec: PluginConfigurationSpec) -> PluginSettings:
        """Read (and if needed migrate) settings; the store lock is already held."""
        validator, migrations = self._prepare_spec(spec)
        path = self._path(spec.plugin_id)
        if not path.exists():
            return PluginSettings(
                spec.plugin_id,
                spec.plugin_version,
                0,
                copy.deepcopy(dict(spec.defaults)),
            )
        document = self._read(path, spec.plugin_id)
        settings = self._parse(document, spec.plugin_id)
        if settings.plugin_version == spec.plugin_version:
            _validate_persisted_configuration(validator, spec.schema, settings.configuration)
            return _copy_settings(settings)

        migrated = self._migrate(settings, spec, migrations)
        _validate_persisted_configuration(validator, spec.schema, migrated.configuration)
        self._write(path, migrated)
        return _copy_settings(migrated)

    def update(
        self,
        spec: PluginConfigurationSpec,
        *,
        expected_revision: int,
        configuration: JsonObject,
    ) -> PluginSettings:
        """Validate and atomically replace settings when the revision matches.

        The revision is re-read and compared inside the store lock: an unlocked
        check-then-write lets two updaters both observe revision N and both
        commit N+1, silently discarding one of the two updates.
        """
        _validate_revision(expected_revision, "expected_revision")
        validator, _migrations = self._prepare_spec(spec)
        _validate_persisted_configuration(validator, spec.schema, configuration)
        with store_lock(self._root, SETTINGS_LOCK_FILE):
            current = self._load_locked(spec)
            if current.revision != expected_revision:
                raise PluginSettingsError(
                    "revision-mismatch",
                    f"expected {expected_revision}; current revision is {current.revision}",
                )
            updated = PluginSettings(
                spec.plugin_id,
                spec.plugin_version,
                current.revision + 1,
                copy.deepcopy(dict(configuration)),
            )
            self._write(self._path(spec.plugin_id), updated)
        return _copy_settings(updated)

    def _prepare_spec(
        self,
        spec: PluginConfigurationSpec,
    ) -> tuple[jsonschema.Draft202012Validator, dict[str, ConfigurationMigration]]:
        _validate_plugin_id(spec.plugin_id)
        _validate_version(spec.plugin_version, "plugin version")
        if not isinstance(spec.schema, Mapping):
            raise PluginSettingsError("invalid-schema", "configuration schema must be an object")
        try:
            jsonschema.Draft202012Validator.check_schema(dict(spec.schema))
        except jsonschema.SchemaError as error:
            raise PluginSettingsError(
                "invalid-schema",
                "configuration schema is invalid",
            ) from error
        validator = jsonschema.Draft202012Validator(dict(spec.schema))
        _validate_persisted_configuration(validator, spec.schema, spec.defaults)
        if len(spec.migrations) > MAX_MIGRATIONS:
            raise PluginSettingsError(
                "invalid-migrations",
                f"at most {MAX_MIGRATIONS} migrations are allowed",
            )
        by_source = {}
        for migration in spec.migrations:
            _validate_version(migration.source_version, "migration source version")
            _validate_version(migration.target_version, "migration target version")
            if migration.source_version == migration.target_version:
                raise PluginSettingsError("invalid-migrations", "migration cannot target itself")
            if not callable(migration.transform):
                raise PluginSettingsError(
                    "invalid-migrations",
                    "migration transform must be callable",
                )
            if migration.source_version in by_source:
                raise PluginSettingsError(
                    "invalid-migrations",
                    f"ambiguous migration from {migration.source_version}",
                )
            by_source[migration.source_version] = migration
        return validator, by_source

    def _read(self, path: Path, plugin_id: str) -> dict:
        try:
            document = read_json_bounded(path, self._max_settings_bytes)
        except JsonTooLargeError as error:
            raise PluginSettingsError(
                "settings-too-large",
                f"settings exceed the {self._max_settings_bytes} byte limit",
            ) from error
        except (OSError, UnicodeError, ValueError) as error:
            raise PluginSettingsError(
                "settings-unreadable",
                f"cannot read settings for {plugin_id}",
            ) from error
        if not isinstance(document, dict):
            raise PluginSettingsError("invalid-settings", "settings document must be an object")
        return document

    def _parse(self, document: dict, expected_plugin_id: str) -> PluginSettings:
        if set(document) != {
            "documentVersion",
            "pluginId",
            "pluginVersion",
            "revision",
            "configuration",
        }:
            raise PluginSettingsError(
                "invalid-settings",
                "settings fields do not match the contract",
            )
        document_version = document["documentVersion"]
        if isinstance(document_version, bool) or document_version != SETTINGS_DOCUMENT_VERSION:
            raise PluginSettingsError(
                "settings-version-incompatible",
                f"unsupported settings document version: {document_version!r}",
            )
        if document["pluginId"] != expected_plugin_id:
            raise PluginSettingsError(
                "plugin-identity-mismatch",
                f"expected {expected_plugin_id}; received {document['pluginId']!r}",
            )
        _validate_version(document["pluginVersion"], "stored plugin version")
        _validate_revision(document["revision"], "stored revision")
        if not isinstance(document["configuration"], dict):
            raise PluginSettingsError("invalid-settings", "configuration must be an object")
        return PluginSettings(
            expected_plugin_id,
            document["pluginVersion"],
            document["revision"],
            document["configuration"],
        )

    def _migrate(
        self,
        settings: PluginSettings,
        spec: PluginConfigurationSpec,
        migrations: dict[str, ConfigurationMigration],
    ) -> PluginSettings:
        version = settings.plugin_version
        configuration = copy.deepcopy(dict(settings.configuration))
        visited = set()
        steps = 0
        while version != spec.plugin_version:
            if version in visited or (steps > 0 and steps >= len(migrations)):
                raise PluginSettingsError("migration-cycle", f"migration cycle at {version}")
            visited.add(version)
            steps += 1
            try:
                migration = migrations[version]
            except KeyError as error:
                raise PluginSettingsError(
                    "migration-missing",
                    f"no migration from {version} to {spec.plugin_version}",
                ) from error
            try:
                transformed = migration.transform(copy.deepcopy(configuration))
            except Exception as error:
                raise PluginSettingsError(
                    "migration-failed",
                    f"migration {version} to {migration.target_version} raised "
                    f"{type(error).__name__}",
                ) from error
            if not isinstance(transformed, dict):
                raise PluginSettingsError(
                    "migration-failed",
                    f"migration {version} to {migration.target_version} returned a non-object",
                )
            configuration = copy.deepcopy(transformed)
            version = migration.target_version
        return PluginSettings(
            settings.plugin_id,
            spec.plugin_version,
            settings.revision + 1,
            configuration,
        )

    def _write(self, path: Path, settings: PluginSettings) -> None:
        document = {
            "documentVersion": SETTINGS_DOCUMENT_VERSION,
            "pluginId": settings.plugin_id,
            "pluginVersion": settings.plugin_version,
            "revision": settings.revision,
            "configuration": dict(settings.configuration),
        }
        try:
            encoded = json.dumps(document, allow_nan=False, separators=(",", ":")).encode()
        except (TypeError, ValueError) as error:
            raise PluginSettingsError(
                "invalid-configuration",
                "configuration must contain JSON values",
            ) from error
        if len(encoded) > self._max_settings_bytes:
            raise PluginSettingsError(
                "settings-too-large",
                f"settings are {len(encoded)} bytes; limit is {self._max_settings_bytes}",
            )
        write_json_atomic(path, document, prefix=f".{settings.plugin_id}-settings-")

    def _path(self, plugin_id: str) -> Path:
        return self._root / f"{plugin_id}.json"


def _validate_persisted_configuration(
    validator: jsonschema.Draft202012Validator,
    schema: Mapping[str, object],
    configuration: JsonObject,
) -> None:
    if not isinstance(configuration, dict):
        raise PluginSettingsError("invalid-configuration", "configuration must be an object")
    try:
        paths = validate_secret_references(schema, configuration)
    except SecretConfigurationError as error:
        code = "invalid-schema" if error.code == "invalid-secret-schema" else error.code
        raise PluginSettingsError(code, error.detail) from error
    errors = persisted_configuration_errors(validator, configuration, paths)
    if errors:
        error = errors[0]
        raise PluginSettingsError(
            "invalid-configuration",
            f"{error.json_path}: {error.message}",
        )


def _validate_plugin_id(plugin_id: str) -> None:
    if not isinstance(plugin_id, str) or not _PLUGIN_ID.fullmatch(plugin_id):
        raise PluginSettingsError("invalid-plugin-id", f"invalid plugin id: {plugin_id!r}")


def _validate_version(version: str, label: str) -> None:
    if not isinstance(version, str) or not _VERSION.fullmatch(version):
        raise PluginSettingsError("invalid-version", f"{label} is invalid: {version!r}")


def _validate_revision(revision: int, label: str) -> None:
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        raise PluginSettingsError("invalid-revision", f"{label} must be a non-negative integer")


def _copy_settings(settings: PluginSettings) -> PluginSettings:
    return PluginSettings(
        settings.plugin_id,
        settings.plugin_version,
        settings.revision,
        copy.deepcopy(dict(settings.configuration)),
    )
