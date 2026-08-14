"""Explicit, revisioned plugin consent grants with bounded provenance."""

from __future__ import annotations

import contextlib
import json
import re
import threading
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from ..atomicio import JsonTooLargeError, read_json_bounded, write_json_atomic
from ..storelock import store_lock

GRANTS_DOCUMENT_VERSION = 1
DEFAULT_MAX_GRANTS_BYTES = 512 * 1024
DEFAULT_AUDIT_EVENTS = 1024
DEFAULT_RELOAD_LOCK_TIMEOUT_SECONDS = 0.1
MAX_DECLARED_PERMISSIONS = 32
_PLUGIN_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_PERMISSION = re.compile(r"^[a-z][a-z0-9-]*:[a-zA-Z0-9*._/-]+$")
_ACTOR = re.compile(r"^[A-Za-z0-9._:@-]{1,120}$")
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,120}$")


class GrantOrigin(StrEnum):
    USER = "user"
    ADMIN = "admin"
    MIGRATION = "migration"


class GrantAction(StrEnum):
    GRANTED = "granted"
    REVOKED = "revoked"


class GrantError(ValueError):
    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class GrantProvenance:
    actor_id: str
    origin: GrantOrigin
    reason: str
    recorded_at_ms: int
    request_id: str


@dataclass(frozen=True, slots=True)
class ActiveGrant:
    permission: str
    provenance: GrantProvenance


@dataclass(frozen=True, slots=True)
class GrantAuditEvent:
    revision: int
    plugin_id: str
    permission: str
    action: GrantAction
    provenance: GrantProvenance


@dataclass(frozen=True, slots=True)
class GrantSnapshot:
    plugin_id: str
    revision: int
    active: tuple[ActiveGrant, ...]
    audit: tuple[GrantAuditEvent, ...]


class GrantLedger:
    """Persist grants only after declared-permission and revision checks pass."""

    def __init__(
        self,
        path: Path | str,
        *,
        max_bytes: int = DEFAULT_MAX_GRANTS_BYTES,
        audit_limit: int = DEFAULT_AUDIT_EVENTS,
    ):
        _validate_positive_integer(max_bytes, "max_bytes")
        _validate_positive_integer(audit_limit, "audit_limit")
        self._path = Path(path).expanduser()
        self._max_bytes = max_bytes
        self._audit_limit = audit_limit
        self._state_lock = threading.RLock()
        self._revision = 0
        self._grants: dict[str, dict[str, GrantProvenance]] = {}
        self._audit: list[GrantAuditEvent] = []
        self._load()
        self._source_stamp = self._path_stamp()

    @property
    def revision(self) -> int:
        with self._state_lock:
            return self._revision

    def reload(
        self,
        *,
        lock_timeout_seconds: float = DEFAULT_RELOAD_LOCK_TIMEOUT_SECONDS,
    ) -> int:
        """Re-read the ledger from disk and return the current revision.

        In-memory state is a snapshot taken at construction, so a grant another
        process revoked is invisible until it is re-read.  Enforcement paths
        that must observe a withdrawal promptly call this first.
        """
        stamp = self._path_stamp()
        with self._state_lock:
            if stamp == self._source_stamp:
                return self._revision
        with self._locked(timeout_seconds=lock_timeout_seconds):
            return self._revision

    def snapshot(self, plugin_id: str, declared_permissions: set[str]) -> GrantSnapshot:
        declared = _validated_declarations(declared_permissions)
        _validate_plugin_id(plugin_id)
        with self._state_lock:
            active = tuple(
                ActiveGrant(permission, provenance)
                for permission, provenance in sorted(self._grants.get(plugin_id, {}).items())
                if permission in declared
            )
            audit = tuple(event for event in self._audit if event.plugin_id == plugin_id)
            return GrantSnapshot(plugin_id, self._revision, active, audit)

    def grant(
        self,
        plugin_id: str,
        permission: str,
        declared_permissions: set[str],
        provenance: GrantProvenance,
        *,
        expected_revision: int,
    ) -> GrantSnapshot:
        declared = _validated_declarations(declared_permissions)
        _validate_plugin_id(plugin_id)
        _validate_permission(permission)
        _validate_provenance(provenance)
        _validate_non_negative_integer(expected_revision, "expected_revision")
        if permission not in declared:
            raise GrantError(
                "permission-undeclared",
                f"{plugin_id} does not declare {permission}",
            )
        with self._locked():
            self._check_revision(expected_revision)
            if permission in self._grants.get(plugin_id, {}):
                return self.snapshot(plugin_id, declared)
            grants = {key: dict(value) for key, value in self._grants.items()}
            grants.setdefault(plugin_id, {})[permission] = provenance
            self._commit(grants, plugin_id, permission, GrantAction.GRANTED, provenance)
            return self.snapshot(plugin_id, declared)

    def revoke(
        self,
        plugin_id: str,
        permission: str,
        declared_permissions: set[str],
        provenance: GrantProvenance,
        *,
        expected_revision: int,
    ) -> GrantSnapshot:
        declared = _validated_declarations(declared_permissions)
        _validate_plugin_id(plugin_id)
        _validate_permission(permission)
        _validate_provenance(provenance)
        _validate_non_negative_integer(expected_revision, "expected_revision")
        with self._locked():
            self._check_revision(expected_revision)
            if permission not in self._grants.get(plugin_id, {}):
                return self.snapshot(plugin_id, declared)
            grants = {key: dict(value) for key, value in self._grants.items()}
            del grants[plugin_id][permission]
            if not grants[plugin_id]:
                del grants[plugin_id]
            self._commit(grants, plugin_id, permission, GrantAction.REVOKED, provenance)
            return self.snapshot(plugin_id, declared)

    def is_granted(
        self,
        plugin_id: str,
        permission: str,
        declared_permissions: set[str],
    ) -> bool:
        declared = _validated_declarations(declared_permissions)
        _validate_plugin_id(plugin_id)
        _validate_permission(permission)
        with self._state_lock:
            return permission in declared and permission in self._grants.get(plugin_id, {})

    def require(
        self,
        plugin_id: str,
        permission: str,
        declared_permissions: set[str],
    ) -> None:
        if not self.is_granted(plugin_id, permission, declared_permissions):
            raise GrantError("permission-denied", f"{plugin_id} lacks grant for {permission}")

    def active_permissions(
        self,
        plugin_id: str,
        declared_permissions: set[str],
    ) -> frozenset[str]:
        """Return only active declarations for sandbox construction."""
        return frozenset(
            grant.permission
            for grant in self.snapshot(plugin_id, declared_permissions).active
        )

    @contextlib.contextmanager
    def _locked(self, *, timeout_seconds: float | None = None):
        """Serialize a mutation and re-read the ledger the caller will replace.

        The in-memory state is a snapshot taken at construction.  Committing
        from it without re-reading lets this process overwrite a revocation
        another process persisted in the meantime, resurrecting a permission
        the user has already withdrawn.
        """
        with self._state_lock, store_lock(
            self._path.parent,
            f".{self._path.name}.lock",
            timeout_seconds=timeout_seconds,
        ):
            self._revision = 0
            self._grants = {}
            self._audit = []
            self._load()
            self._source_stamp = self._path_stamp()
            yield

    def _check_revision(self, expected_revision: int) -> None:
        _validate_non_negative_integer(expected_revision, "expected_revision")
        if expected_revision != self._revision:
            raise GrantError(
                "revision-mismatch",
                f"expected {expected_revision}; current revision is {self._revision}",
            )

    def _commit(
        self,
        grants: dict[str, dict[str, GrantProvenance]],
        plugin_id: str,
        permission: str,
        action: GrantAction,
        provenance: GrantProvenance,
    ) -> None:
        revision = self._revision + 1
        audit = [
            *self._audit,
            GrantAuditEvent(revision, plugin_id, permission, action, provenance),
        ][-self._audit_limit :]
        document = _encode_document(revision, grants, audit)
        encoded = json.dumps(document, separators=(",", ":"), sort_keys=True).encode()
        if len(encoded) > self._max_bytes:
            raise GrantError(
                "grants-too-large",
                f"grant state is {len(encoded)} bytes; limit is {self._max_bytes}",
            )
        write_json_atomic(self._path, document, prefix=".plugin-grants-")
        self._revision = revision
        self._grants = grants
        self._audit = audit
        self._source_stamp = self._path_stamp()

    def _path_stamp(self) -> tuple[int, int, int, int] | None:
        try:
            status = self._path.stat()
        except FileNotFoundError:
            return None
        return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns)

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            document = read_json_bounded(self._path, self._max_bytes)
        except JsonTooLargeError as error:
            raise GrantError(
                "grants-too-large",
                f"grant state exceeds the {self._max_bytes} byte limit",
            ) from error
        except (OSError, UnicodeError, ValueError) as error:
            raise GrantError("grants-unreadable", "cannot read grant state") from error
        if not isinstance(document, dict) or set(document) != {
            "documentVersion",
            "revision",
            "grants",
            "audit",
        }:
            raise GrantError("invalid-grants", "grant document fields do not match the contract")
        document_version = document["documentVersion"]
        if isinstance(document_version, bool) or document_version != GRANTS_DOCUMENT_VERSION:
            raise GrantError(
                "grants-version-incompatible",
                f"unsupported grant document version: {document_version!r}",
            )
        _validate_non_negative_integer(document["revision"], "stored revision")
        grants = _decode_grants(document["grants"])
        audit = _decode_audit(document["audit"], document["revision"], self._audit_limit)
        self._revision = document["revision"]
        self._grants = grants
        self._audit = audit


def _encode_document(
    revision: int,
    grants: dict[str, dict[str, GrantProvenance]],
    audit: list[GrantAuditEvent],
) -> dict:
    return {
        "documentVersion": GRANTS_DOCUMENT_VERSION,
        "revision": revision,
        "grants": {
            plugin_id: {
                permission: _encode_provenance(provenance)
                for permission, provenance in sorted(plugin_grants.items())
            }
            for plugin_id, plugin_grants in sorted(grants.items())
        },
        "audit": [
            {
                "revision": event.revision,
                "pluginId": event.plugin_id,
                "permission": event.permission,
                "action": str(event.action),
                "provenance": _encode_provenance(event.provenance),
            }
            for event in audit
        ],
    }


def _encode_provenance(provenance: GrantProvenance) -> dict:
    return {
        "actorId": provenance.actor_id,
        "origin": str(provenance.origin),
        "reason": provenance.reason,
        "recordedAtMs": provenance.recorded_at_ms,
        "requestId": provenance.request_id,
    }


def _decode_grants(value: object) -> dict[str, dict[str, GrantProvenance]]:
    if not isinstance(value, dict):
        raise GrantError("invalid-grants", "grants must be an object")
    grants = {}
    for plugin_id, plugin_grants in value.items():
        _validate_plugin_id(plugin_id)
        if not isinstance(plugin_grants, dict):
            raise GrantError("invalid-grants", "plugin grants must be an object")
        decoded = {}
        for permission, provenance in plugin_grants.items():
            _validate_permission(permission)
            decoded[permission] = _decode_provenance(provenance)
        if decoded:
            grants[plugin_id] = decoded
    return grants


def _decode_audit(value: object, revision: int, audit_limit: int) -> list[GrantAuditEvent]:
    if not isinstance(value, list) or len(value) > audit_limit:
        raise GrantError("invalid-grants", f"audit must contain at most {audit_limit} events")
    events = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "revision",
            "pluginId",
            "permission",
            "action",
            "provenance",
        }:
            raise GrantError("invalid-grants", "audit event fields do not match the contract")
        _validate_non_negative_integer(item["revision"], "audit revision")
        if item["revision"] < 1 or item["revision"] > revision:
            raise GrantError("invalid-grants", "audit revision is outside the ledger")
        _validate_plugin_id(item["pluginId"])
        _validate_permission(item["permission"])
        try:
            action = GrantAction(item["action"])
        except (TypeError, ValueError) as error:
            raise GrantError("invalid-grants", "audit action is invalid") from error
        events.append(
            GrantAuditEvent(
                item["revision"],
                item["pluginId"],
                item["permission"],
                action,
                _decode_provenance(item["provenance"]),
            )
        )
    return events


def _decode_provenance(value: object) -> GrantProvenance:
    if not isinstance(value, dict) or set(value) != {
        "actorId",
        "origin",
        "reason",
        "recordedAtMs",
        "requestId",
    }:
        raise GrantError("invalid-grants", "provenance fields do not match the contract")
    try:
        origin = GrantOrigin(value["origin"])
    except (TypeError, ValueError) as error:
        raise GrantError("invalid-grants", "provenance origin is invalid") from error
    provenance = GrantProvenance(
        value["actorId"],
        origin,
        value["reason"],
        value["recordedAtMs"],
        value["requestId"],
    )
    _validate_provenance(provenance)
    return provenance


def _validated_declarations(permissions: set[str]) -> frozenset[str]:
    if not isinstance(permissions, (set, frozenset)):
        raise GrantError("invalid-declarations", "declared permissions must be a set")
    if len(permissions) > MAX_DECLARED_PERMISSIONS:
        raise GrantError(
            "invalid-declarations",
            f"at most {MAX_DECLARED_PERMISSIONS} permissions may be declared",
        )
    for permission in permissions:
        _validate_permission(permission)
    return frozenset(permissions)


def _validate_provenance(provenance: GrantProvenance) -> None:
    if not isinstance(provenance, GrantProvenance):
        raise GrantError("invalid-provenance", "provenance is required")
    if not isinstance(provenance.actor_id, str) or not _ACTOR.fullmatch(provenance.actor_id):
        raise GrantError("invalid-provenance", "actor id is invalid")
    if not isinstance(provenance.origin, GrantOrigin):
        raise GrantError("invalid-provenance", "origin is invalid")
    if not isinstance(provenance.reason, str) or len(provenance.reason) > 240:
        raise GrantError("invalid-provenance", "reason must contain at most 240 characters")
    _validate_non_negative_integer(provenance.recorded_at_ms, "recorded_at_ms")
    if not isinstance(provenance.request_id, str) or not _REQUEST_ID.fullmatch(
        provenance.request_id
    ):
        raise GrantError("invalid-provenance", "request id is invalid")


def _validate_plugin_id(plugin_id: str) -> None:
    if not isinstance(plugin_id, str) or not _PLUGIN_ID.fullmatch(plugin_id):
        raise GrantError("invalid-plugin-id", f"invalid plugin id: {plugin_id!r}")


def _validate_permission(permission: str) -> None:
    if (
        not isinstance(permission, str)
        or len(permission) > 160
        or not _PERMISSION.fullmatch(permission)
    ):
        raise GrantError("invalid-permission", f"invalid permission: {permission!r}")


def _validate_non_negative_integer(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GrantError("invalid-integer", f"{label} must be a non-negative integer")


def _validate_positive_integer(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
