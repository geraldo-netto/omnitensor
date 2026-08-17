"""Secret-reference persistence, injected resolution, and value redaction."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import jsonschema

from .protocol import JsonObject

SECRET_SCHEMA_MARKER = "x-omnitensor-secret"
SECRET_REFERENCE_FIELD = "$secretRef"
MAX_SECRET_FIELDS = 32
MAX_SECRET_VALUE_BYTES = 64 * 1024
MAX_SECRET_PROVIDER_CHARS = 64
MAX_SECRET_KEY_CHARS = 240
MAX_REDACTION_DEPTH = 16
MAX_REDACTION_NODES = 4096
REDACTED = "[REDACTED]"
REDACTION_LIMIT = "[REDACTED:LIMIT]"
_PROVIDER = re.compile(r"^[a-z][a-z0-9-]*$")
_KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")

SecretPath = tuple[str, ...]


class SecretConfigurationError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class SecretResolutionError(RuntimeError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class SecretReference:
    provider: str
    key: str
    version: str | None = None

    def document(self) -> JsonObject:
        body: JsonObject = {"provider": self.provider, "key": self.key}
        if self.version is not None:
            body["version"] = self.version
        return {SECRET_REFERENCE_FIELD: body}


class SecretProvider(Protocol):
    def resolve(self, reference: SecretReference) -> str | bytes: ...


@dataclass(slots=True, repr=False)
class ResolvedConfiguration:
    configuration: JsonObject = field(repr=False)
    redactor: SecretRedactor = field(repr=False)
    _secret_paths: tuple[SecretPath, ...] = field(repr=False)
    _closed: bool = field(default=False, repr=False)

    def __repr__(self) -> str:
        return "ResolvedConfiguration(configuration=[REDACTED])"

    def close(self) -> None:
        if self._closed:
            return
        for path in self._secret_paths:
            _set_path(self.configuration, path, REDACTED)
        self.redactor.clear()
        self._closed = True

    def __enter__(self) -> ResolvedConfiguration:
        if self._closed:
            raise RuntimeError("resolved configuration is closed")
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


class SecretRedactor:
    """Replace resolved values in log text and JSON-like public structures."""

    def __init__(self, values: Sequence[str | bytes]) -> None:
        normalized: list[str] = []
        for value in values:
            if isinstance(value, bytes):
                text = value.decode("utf-8", errors="replace")
            elif isinstance(value, str):
                text = value
            else:
                raise TypeError("redaction values must be text or bytes")
            if text and text not in normalized:
                normalized.append(text)
        self._values = sorted(normalized, key=len, reverse=True)

    def __repr__(self) -> str:
        return "SecretRedactor(values=[REDACTED])"

    def clear(self) -> None:
        self._values.clear()

    def redact_text(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("redacted text must be a string")
        for value in self._values:
            text = text.replace(value, REDACTED)
        return text

    def redact(self, value: object) -> object:
        nodes = [0]
        return self._redact(value, 0, nodes)

    def _redact(self, value: object, depth: int, nodes: list[int]) -> object:
        nodes[0] += 1
        if depth > MAX_REDACTION_DEPTH or nodes[0] > MAX_REDACTION_NODES:
            return REDACTION_LIMIT
        if isinstance(value, Mapping):
            return {str(key): self._redact(item, depth + 1, nodes) for key, item in value.items()}
        if isinstance(value, list):
            return [self._redact(item, depth + 1, nodes) for item in value]
        if isinstance(value, tuple):
            return tuple(self._redact(item, depth + 1, nodes) for item in value)
        return self._redact_scalar(value)

    def _redact_scalar(self, value: object) -> object:
        if isinstance(value, str):
            return self.redact_text(value)
        if isinstance(value, bytes):
            return self.redact_text(value.decode("utf-8", errors="replace"))
        if isinstance(value, BaseException):
            detail = self.redact_text(str(value))
            return f"{type(value).__name__}: {detail}"
        if value is None or isinstance(value, (bool, int, float)):
            return value
        return f"<{type(value).__name__}>"


def secret_paths(schema: Mapping[str, object]) -> tuple[SecretPath, ...]:
    if not isinstance(schema, Mapping):
        raise SecretConfigurationError(
            "invalid-secret-schema", "configuration schema must be an object"
        )
    found: list[SecretPath] = []
    _collect_secret_paths(schema, (), found)
    if len(found) > MAX_SECRET_FIELDS:
        raise SecretConfigurationError(
            "invalid-secret-schema",
            f"at most {MAX_SECRET_FIELDS} secret fields are allowed",
        )
    return tuple(found)


def validate_secret_references(
    schema: Mapping[str, object], configuration: Mapping[str, object]
) -> tuple[SecretPath, ...]:
    paths = secret_paths(schema)
    for path in paths:
        present, value = _get_path(configuration, path)
        if present:
            try:
                parse_secret_reference(value)
            except SecretConfigurationError as error:
                raise SecretConfigurationError(
                    "secret-reference-required",
                    f"{_format_path(path)} must contain a secret reference",
                ) from error
    return paths


def parse_secret_reference(document: object) -> SecretReference:
    if not isinstance(document, Mapping) or set(document) != {SECRET_REFERENCE_FIELD}:
        raise SecretConfigurationError(
            "invalid-secret-reference", "secret reference fields do not match the contract"
        )
    body = document[SECRET_REFERENCE_FIELD]
    if not isinstance(body, Mapping) or not {"provider", "key"} <= set(body):
        raise SecretConfigurationError(
            "invalid-secret-reference", "secret reference body is invalid"
        )
    if set(body) - {"provider", "key", "version"}:
        raise SecretConfigurationError(
            "invalid-secret-reference", "secret reference body has unknown fields"
        )
    provider = body["provider"]
    key = body["key"]
    version = body.get("version")
    if (
        not isinstance(provider, str)
        or not 1 <= len(provider) <= MAX_SECRET_PROVIDER_CHARS
        or _PROVIDER.fullmatch(provider) is None
    ):
        raise SecretConfigurationError("invalid-secret-reference", "secret provider is invalid")
    if (
        not isinstance(key, str)
        or not 1 <= len(key) <= MAX_SECRET_KEY_CHARS
        or _KEY.fullmatch(key) is None
    ):
        raise SecretConfigurationError("invalid-secret-reference", "secret key is invalid")
    if version is not None and (
        not isinstance(version, str)
        or not 1 <= len(version) <= MAX_SECRET_KEY_CHARS
        or _KEY.fullmatch(version) is None
    ):
        raise SecretConfigurationError("invalid-secret-reference", "secret version is invalid")
    return SecretReference(provider, key, version)


def resolve_configuration(
    schema: Mapping[str, object],
    configuration: Mapping[str, object],
    provider: SecretProvider,
) -> ResolvedConfiguration:
    if not isinstance(configuration, Mapping):
        raise SecretConfigurationError("invalid-configuration", "configuration must be an object")
    paths = validate_secret_references(schema, configuration)
    resolved = copy.deepcopy(dict(configuration))
    values: list[str] = []
    for path in paths:
        present, document = _get_path(resolved, path)
        if not present:
            continue
        reference = parse_secret_reference(document)
        try:
            value = provider.resolve(reference)
        except Exception as error:
            raise SecretResolutionError(
                "secret-unavailable",
                f"secret provider failed with {type(error).__name__}",
            ) from error
        text = _secret_text(value)
        values.append(text)
        _set_path(resolved, path, text)
    validator = jsonschema.Draft202012Validator(dict(schema))
    errors = sorted(validator.iter_errors(resolved), key=lambda error: error.json_path)
    if errors:
        location = errors[0].json_path
        raise SecretResolutionError(
            "resolved-configuration-invalid",
            f"resolved configuration does not match schema at {location}",
        )
    return ResolvedConfiguration(resolved, SecretRedactor(values), paths)


def persisted_configuration_errors(
    validator: jsonschema.Draft202012Validator,
    configuration: Mapping[str, object],
    paths: tuple[SecretPath, ...],
) -> list[jsonschema.ValidationError]:
    return sorted(
        (
            error
            for error in validator.iter_errors(configuration)
            if not _path_is_secret(tuple(str(part) for part in error.absolute_path), paths)
        ),
        key=lambda error: error.json_path,
    )


def _collect_secret_paths(
    schema: Mapping[str, object], path: SecretPath, found: list[SecretPath]
) -> None:
    marker = schema.get(SECRET_SCHEMA_MARKER, False)
    if not isinstance(marker, bool):
        raise SecretConfigurationError(
            "invalid-secret-schema",
            f"{_format_path(path)} secret marker must be boolean",
        )
    if marker:
        if schema.get("type") != "string":
            raise SecretConfigurationError(
                "invalid-secret-schema",
                f"{_format_path(path)} secret field must have string type",
            )
        if not path:
            raise SecretConfigurationError(
                "invalid-secret-schema", "configuration root cannot be secret"
            )
        found.append(path)
        return
    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        return
    for name, child in properties.items():
        if isinstance(name, str) and isinstance(child, Mapping):
            _collect_secret_paths(child, (*path, name), found)


def _path_is_secret(path: SecretPath, secrets: tuple[SecretPath, ...]) -> bool:
    return any(path[: len(secret)] == secret for secret in secrets)


def _get_path(configuration: Mapping[str, object], path: SecretPath) -> tuple[bool, object]:
    current: object = configuration
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _set_path(configuration: JsonObject, path: SecretPath, value: object) -> None:
    current: JsonObject = configuration
    for part in path[:-1]:
        child = current[part]
        if not isinstance(child, dict):
            raise SecretConfigurationError(
                "invalid-configuration", f"{_format_path(path)} parent is not an object"
            )
        current = child
    current[path[-1]] = value


def _secret_text(value: object) -> str:
    if isinstance(value, bytes):
        try:
            text = value.decode("utf-8")
        except UnicodeDecodeError as error:
            raise SecretResolutionError(
                "secret-invalid", "secret value is not valid UTF-8"
            ) from error
    elif isinstance(value, str):
        text = value
    else:
        raise SecretResolutionError("secret-invalid", "secret provider returned a non-text value")
    size = len(text.encode("utf-8"))
    if not text or size > MAX_SECRET_VALUE_BYTES:
        raise SecretResolutionError(
            "secret-invalid", "secret value is empty or exceeds the byte limit"
        )
    return text


def _format_path(path: SecretPath) -> str:
    return "$" + "".join(f".{part}" for part in path)
