from __future__ import annotations

import json

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.plugins import (
    MAX_REDACTION_DEPTH,
    MAX_SECRET_FIELDS,
    MAX_SECRET_VALUE_BYTES,
    REDACTED,
    REDACTION_LIMIT,
    SECRET_REFERENCE_FIELD,
    SECRET_SCHEMA_MARKER,
    PluginConfigurationSpec,
    PluginSettingsError,
    PluginSettingsStore,
    SecretConfigurationError,
    SecretRedactor,
    SecretReference,
    SecretResolutionError,
    parse_secret_reference,
    resolve_configuration,
    secret_paths,
    validate_secret_references,
)

SECRET_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["endpoint", "credentials"],
    "properties": {
        "endpoint": {"type": "string", "minLength": 1},
        "credentials": {
            "type": "object",
            "additionalProperties": False,
            "required": ["token"],
            "properties": {
                "token": {
                    "type": "string",
                    "minLength": 8,
                    SECRET_SCHEMA_MARKER: True,
                },
                "optional": {
                    "type": "string",
                    SECRET_SCHEMA_MARKER: True,
                },
            },
        },
    },
}


def reference(provider="secret-service", key="plugins/echo/token", version="v1"):
    return SecretReference(provider, key, version).document()


def persisted_configuration():
    return {
        "endpoint": "local",
        "credentials": {"token": reference()},
    }


class MappingProvider:
    def __init__(self, values=None, error=None):
        self.values = values or {}
        self.error = error
        self.calls = []

    def resolve(self, secret_reference):
        self.calls.append(secret_reference)
        if self.error is not None:
            raise self.error
        return self.values[
            (
                secret_reference.provider,
                secret_reference.key,
                secret_reference.version,
            )
        ]


def test_secret_paths_are_explicit_nested_and_stable():
    assert secret_paths(SECRET_SCHEMA) == (
        ("credentials", "token"),
        ("credentials", "optional"),
    )
    assert secret_paths({"type": "object"}) == ()


@pytest.mark.parametrize(
    "schema, detail",
    [
        ([], "configuration schema must be an object"),
        (
            {SECRET_SCHEMA_MARKER: True, "type": "string"},
            "configuration root cannot be secret",
        ),
        (
            {
                "type": "object",
                "properties": {"token": {SECRET_SCHEMA_MARKER: "yes", "type": "string"}},
            },
            "$.token secret marker must be boolean",
        ),
        (
            {
                "type": "object",
                "properties": {"token": {SECRET_SCHEMA_MARKER: True, "type": "number"}},
            },
            "$.token secret field must have string type",
        ),
        (
            {
                "type": "object",
                "properties": {
                    f"secret-{index}": {
                        SECRET_SCHEMA_MARKER: True,
                        "type": "string",
                    }
                    for index in range(MAX_SECRET_FIELDS + 1)
                },
            },
            "at most 32 secret fields are allowed",
        ),
    ],
)
def test_secret_schema_is_bounded_and_unambiguous(schema, detail):
    with pytest.raises(SecretConfigurationError) as excinfo:
        secret_paths(schema)
    assert excinfo.value.code == "invalid-secret-schema"
    assert excinfo.value.detail == detail


def test_secret_schema_accepts_exact_field_limit():
    schema = {
        "type": "object",
        "properties": {
            f"secret-{index}": {SECRET_SCHEMA_MARKER: True, "type": "string"}
            for index in range(MAX_SECRET_FIELDS)
        },
    }
    assert len(secret_paths(schema)) == MAX_SECRET_FIELDS


def test_secret_reference_round_trip_is_metadata_only():
    document = reference()

    parsed = parse_secret_reference(document)

    assert parsed == SecretReference("secret-service", "plugins/echo/token", "v1")
    assert parsed.document() == document
    assert set(document) == {SECRET_REFERENCE_FIELD}
    assert "material" not in json.dumps(document)
    without_version = SecretReference("keyring", "echo/token").document()
    assert parse_secret_reference(without_version).version is None


@pytest.mark.parametrize(
    "document, detail",
    [
        ("raw-secret", "secret reference fields do not match the contract"),
        ({}, "secret reference fields do not match the contract"),
        ({SECRET_REFERENCE_FIELD: []}, "secret reference body is invalid"),
        (
            {SECRET_REFERENCE_FIELD: {"provider": "keyring"}},
            "secret reference body is invalid",
        ),
        (
            {SECRET_REFERENCE_FIELD: {"provider": "Bad", "key": "key"}},
            "secret provider is invalid",
        ),
        (
            {SECRET_REFERENCE_FIELD: {"provider": "keyring", "key": "bad key"}},
            "secret key is invalid",
        ),
        (
            {
                SECRET_REFERENCE_FIELD: {
                    "provider": "keyring",
                    "key": "key",
                    "version": "bad version",
                }
            },
            "secret version is invalid",
        ),
        (
            {
                SECRET_REFERENCE_FIELD: {
                    "provider": "keyring",
                    "key": "key",
                    "extra": True,
                }
            },
            "secret reference body has unknown fields",
        ),
    ],
)
def test_invalid_secret_references_never_echo_values(document, detail):
    with pytest.raises(SecretConfigurationError, match=detail) as excinfo:
        parse_secret_reference(document)
    assert excinfo.value.detail == detail
    assert "raw-secret" not in excinfo.value.detail
    assert "bad key" not in excinfo.value.detail


def test_raw_secret_is_rejected_at_annotated_path_without_echo():
    configuration = persisted_configuration()
    configuration["credentials"]["token"] = "super-secret-material"

    with pytest.raises(SecretConfigurationError) as excinfo:
        validate_secret_references(SECRET_SCHEMA, configuration)

    assert excinfo.value.code == "secret-reference-required"
    assert excinfo.value.detail == "$.credentials.token must contain a secret reference"
    assert "super-secret-material" not in str(excinfo.value)


@given(material=st.text(max_size=100))
def test_arbitrary_raw_secret_material_is_never_echoed_by_reference_rejection(
    material,
):
    raw = f"private-material:{material}"
    configuration = persisted_configuration()
    configuration["credentials"]["token"] = raw
    with pytest.raises(SecretConfigurationError) as excinfo:
        validate_secret_references(SECRET_SCHEMA, configuration)
    assert raw not in str(excinfo.value)


@given(number=st.integers(min_value=0, max_value=10**20), text=st.text(max_size=100))
def test_redactor_removes_arbitrary_registered_material(number, text):
    secret = f"secret-{number}"
    public = SecretRedactor([secret]).redact_text(f"{text}{secret}{text}")
    assert secret not in public


def test_resolution_uses_injected_provider_and_context_close_scrubs_configuration():
    provider = MappingProvider({("secret-service", "plugins/echo/token", "v1"): "top-secret-token"})

    resolved = resolve_configuration(
        SECRET_SCHEMA,
        persisted_configuration(),
        provider,
    )

    assert provider.calls == [SecretReference("secret-service", "plugins/echo/token", "v1")]
    assert resolved.configuration == {
        "endpoint": "local",
        "credentials": {"token": "top-secret-token"},
    }
    assert repr(resolved) == "ResolvedConfiguration(configuration=[REDACTED])"
    assert repr(resolved.redactor) == "SecretRedactor(values=[REDACTED])"
    assert resolved.redactor.redact_text("token=top-secret-token") == "token=[REDACTED]"

    with resolved as active:
        assert active.configuration["credentials"]["token"] == "top-secret-token"
    assert resolved.configuration["credentials"]["token"] == REDACTED
    assert resolved.redactor.redact_text("top-secret-token") == "top-secret-token"
    resolved.close()
    with pytest.raises(RuntimeError, match="is closed"):
        resolved.__enter__()


def test_bytes_secret_is_decoded_and_duplicate_values_redact_longest_first():
    schema = {
        "type": "object",
        "required": ["short", "long"],
        "properties": {
            "short": {"type": "string", SECRET_SCHEMA_MARKER: True},
            "long": {"type": "string", SECRET_SCHEMA_MARKER: True},
        },
    }
    configuration = {
        "short": SecretReference("keyring", "short").document(),
        "long": SecretReference("keyring", "long").document(),
    }
    provider = MappingProvider(
        {
            ("keyring", "short", None): b"secret",
            ("keyring", "long", None): b"secret-value",
        }
    )

    resolved = resolve_configuration(schema, configuration, provider)

    assert resolved.redactor.redact_text("secret-value secret") == ("[REDACTED] [REDACTED]")


@pytest.mark.parametrize(
    "provider, code, detail",
    [
        (
            MappingProvider(error=RuntimeError("actual secret")),
            "secret-unavailable",
            "secret provider failed with RuntimeError",
        ),
        (
            MappingProvider({("secret-service", "plugins/echo/token", "v1"): 7}),
            "secret-invalid",
            "secret provider returned a non-text value",
        ),
        (
            MappingProvider({("secret-service", "plugins/echo/token", "v1"): b"\xff"}),
            "secret-invalid",
            "secret value is not valid UTF-8",
        ),
        (
            MappingProvider({("secret-service", "plugins/echo/token", "v1"): ""}),
            "secret-invalid",
            "secret value is empty or exceeds the byte limit",
        ),
        (
            MappingProvider(
                {("secret-service", "plugins/echo/token", "v1"): "x" * (MAX_SECRET_VALUE_BYTES + 1)}
            ),
            "secret-invalid",
            "secret value is empty or exceeds the byte limit",
        ),
        (
            MappingProvider({("secret-service", "plugins/echo/token", "v1"): "short"}),
            "resolved-configuration-invalid",
            "resolved configuration does not match schema at $.credentials.token",
        ),
    ],
)
def test_resolution_failures_are_stable_and_never_echo_material(provider, code, detail):
    with pytest.raises(SecretResolutionError) as excinfo:
        resolve_configuration(SECRET_SCHEMA, persisted_configuration(), provider)
    assert excinfo.value.code == code
    assert excinfo.value.detail == detail
    assert "actual secret" not in str(excinfo.value)
    assert "short" not in str(excinfo.value)


def test_redactor_handles_logs_snapshots_results_exceptions_and_unknown_objects():
    redactor = SecretRedactor(["secret-token", b"binary-secret", "secret-token"])
    error = RuntimeError("request used secret-token")
    public = redactor.redact(
        {
            "log": "bearer secret-token",
            "snapshot": [b"binary-secret", {"result": "secret-token suffix"}],
            "tuple": ("safe", "secret-token"),
            "error": error,
            "count": 2,
            "unknown": object(),
        }
    )

    assert public == {
        "log": "bearer [REDACTED]",
        "snapshot": ["[REDACTED]", {"result": "[REDACTED] suffix"}],
        "tuple": ("safe", "[REDACTED]"),
        "error": "RuntimeError: request used [REDACTED]",
        "count": 2,
        "unknown": "<object>",
    }
    with pytest.raises(TypeError, match="redacted text"):
        redactor.redact_text(b"secret-token")
    with pytest.raises(TypeError, match="redaction values"):
        SecretRedactor([object()])


def test_redaction_depth_is_hard_bounded():
    nested = "secret"
    for _ in range(MAX_REDACTION_DEPTH + 2):
        nested = [nested]
    redacted = SecretRedactor(["secret"]).redact(nested)
    current = redacted
    for _ in range(MAX_REDACTION_DEPTH + 1):
        current = current[0]
    assert current == REDACTION_LIMIT


def plugin_spec(defaults=None, schema=SECRET_SCHEMA):
    return PluginConfigurationSpec(
        "secret-plugin",
        "1.0.0",
        schema,
        defaults or persisted_configuration(),
    )


def test_settings_store_persists_only_references_and_resolves_ephemerally(tmp_path):
    store = PluginSettingsStore(tmp_path)
    configuration = persisted_configuration()

    saved = store.update(plugin_spec(), expected_revision=0, configuration=configuration)

    path = tmp_path / "secret-plugin.json"
    persisted = path.read_text()
    assert "top-secret-token" not in persisted
    assert json.loads(persisted)["configuration"] == configuration
    assert store.load(plugin_spec()) == saved
    provider = MappingProvider({("secret-service", "plugins/echo/token", "v1"): "top-secret-token"})
    resolved = resolve_configuration(SECRET_SCHEMA, saved.configuration, provider)
    assert resolved.configuration["credentials"]["token"] == "top-secret-token"


def test_settings_reject_raw_secret_before_write_and_preserve_prior_revision(tmp_path):
    store = PluginSettingsStore(tmp_path)
    store.update(plugin_spec(), expected_revision=0, configuration=persisted_configuration())
    path = tmp_path / "secret-plugin.json"
    before = path.read_bytes()
    raw = persisted_configuration()
    raw["credentials"]["token"] = "raw-secret-value"

    with pytest.raises(PluginSettingsError) as excinfo:
        store.update(plugin_spec(), expected_revision=1, configuration=raw)

    assert excinfo.value.code == "secret-reference-required"
    assert "raw-secret-value" not in str(excinfo.value)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "schema",
    [
        {
            "type": "object",
            "properties": {"token": {"type": "number", SECRET_SCHEMA_MARKER: True}},
        },
        {
            "type": "object",
            "properties": {"token": {"type": "string", SECRET_SCHEMA_MARKER: "true"}},
        },
    ],
)
def test_settings_reject_invalid_secret_schema_before_state_access(tmp_path, schema):
    with pytest.raises(PluginSettingsError) as excinfo:
        PluginSettingsStore(tmp_path).load(plugin_spec(schema=schema, defaults={}))
    assert excinfo.value.code == "invalid-schema"
    assert not (tmp_path / "secret-plugin.json").exists()
