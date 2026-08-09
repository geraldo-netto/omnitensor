"""Offline artifact publisher signature and provenance verification."""

from __future__ import annotations

import base64
import binascii
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from ..atomicio import read_json_bounded
from .artifacts import ArtifactReference

_IDENTITY = re.compile(r"[a-z0-9](?:[a-z0-9._-]{0,62}[a-z0-9])?")
_PROVENANCE_FILE = "provenance.json"
# A provenance record is a fixed handful of identifiers plus one signature;
# anything larger is corruption or tampering, not a document worth parsing.
MAX_PROVENANCE_BYTES = 64 * 1024
_PROVENANCE_VERSION = 1
_MAX_SOURCE_LENGTH = 2048
_MAX_TIMESTAMP_MS = 253402300799999


@dataclass(frozen=True, slots=True)
class ArtifactProvenance:
    """Publisher-signed origin data bound to one immutable artifact."""

    publisher: str
    key_id: str
    source: str
    published_at_ms: int
    signature: str


@dataclass(frozen=True, slots=True)
class ArtifactTrustRoot:
    """Configured publisher key usable without network access."""

    publisher: str
    key_id: str
    public_key: bytes


@dataclass(frozen=True, slots=True)
class ArtifactTrustPolicy:
    """Local signature and revocation policy."""

    require_signature: bool = True
    revoked_publishers: frozenset[str] = field(default_factory=frozenset)
    revoked_keys: frozenset[tuple[str, str]] = field(default_factory=frozenset)


@dataclass(frozen=True, slots=True)
class ArtifactTrustDecision:
    """Stable trust result safe to expose through orchestration."""

    trusted: bool
    reason: str


def artifact_signature_payload(
    reference: ArtifactReference,
    provenance: ArtifactProvenance,
) -> bytes:
    """Return domain-separated canonical bytes signed by publishers."""
    document = {
        "artifact": {
            "format": reference.format,
            "id": reference.id,
            "sha256": reference.sha256,
            "version": reference.version,
        },
        "provenance": {
            "keyId": provenance.key_id,
            "publishedAtMs": provenance.published_at_ms,
            "publisher": provenance.publisher,
            "source": provenance.source,
        },
    }
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return b"omnitensor-artifact-signature-v1\n" + canonical.encode("utf-8")


def artifact_provenance_document(
    reference: ArtifactReference,
    provenance: ArtifactProvenance | None,
) -> dict[str, object]:
    """Build strict persisted provenance bound to its artifact reference."""
    return {
        "version": _PROVENANCE_VERSION,
        "artifact": {
            "id": reference.id,
            "version": reference.version,
            "format": reference.format,
            "sha256": reference.sha256,
        },
        "provenance": (
            {
                "publisher": provenance.publisher,
                "keyId": provenance.key_id,
                "source": provenance.source,
                "publishedAtMs": provenance.published_at_ms,
                "signature": provenance.signature,
            }
            if provenance is not None
            else None
        ),
    }


class ArtifactTrustVerifier:
    """Verify Ed25519 provenance against configured offline trust roots."""

    def __init__(
        self,
        roots: tuple[ArtifactTrustRoot, ...] | list[ArtifactTrustRoot],
        *,
        policy: ArtifactTrustPolicy | None = None,
    ):
        self._policy = policy or ArtifactTrustPolicy()
        self._roots: dict[tuple[str, str], Ed25519PublicKey] = {}
        for root in roots:
            error = _root_error(root)
            if error:
                raise ValueError(error)
            identity = (root.publisher, root.key_id)
            if identity in self._roots:
                raise ValueError(f"duplicate artifact trust root: {root.publisher}/{root.key_id}")
            self._roots[identity] = Ed25519PublicKey.from_public_bytes(root.public_key)

    def verify(
        self,
        reference: ArtifactReference,
        provenance: ArtifactProvenance | None,
    ) -> ArtifactTrustDecision:
        """Verify one signature without external I/O."""
        if provenance is None:
            if self._policy.require_signature:
                return ArtifactTrustDecision(False, "artifact is unsigned")
            return ArtifactTrustDecision(True, "")
        invalid = _provenance_error(provenance)
        if invalid:
            return ArtifactTrustDecision(False, invalid)
        identity = (provenance.publisher, provenance.key_id)
        policy_error = self._policy_error(provenance)
        if policy_error:
            return ArtifactTrustDecision(False, policy_error)
        public_key = self._roots.get(identity)
        if public_key is None:
            return ArtifactTrustDecision(
                False,
                f"publisher key is not trusted: {provenance.publisher}/{provenance.key_id}",
            )
        return _verify_signature(public_key, reference, provenance)

    def verify_installed(
        self,
        reference: ArtifactReference,
        version_root: Path,
    ) -> ArtifactTrustDecision:
        """Re-read persisted provenance and enforce current offline policy."""
        path = Path(version_root) / _PROVENANCE_FILE
        try:
            document = read_json_bounded(path, MAX_PROVENANCE_BYTES)
        except FileNotFoundError:
            return self.verify(reference, None)
        except (OSError, ValueError) as error:
            return ArtifactTrustDecision(False, f"artifact provenance cannot be read: {error}")
        provenance, error = _parse_provenance_document(document, reference)
        if error:
            return ArtifactTrustDecision(False, error)
        return self.verify(reference, provenance)

    def _policy_error(self, provenance: ArtifactProvenance) -> str:
        if provenance.publisher in self._policy.revoked_publishers:
            return f"publisher is revoked: {provenance.publisher}"
        if (provenance.publisher, provenance.key_id) in self._policy.revoked_keys:
            return f"publisher key is revoked: {provenance.publisher}/{provenance.key_id}"
        return ""


def _verify_signature(
    public_key: Ed25519PublicKey,
    reference: ArtifactReference,
    provenance: ArtifactProvenance,
) -> ArtifactTrustDecision:
    try:
        signature = base64.b64decode(provenance.signature, validate=True)
    except (binascii.Error, ValueError):
        return ArtifactTrustDecision(False, "artifact signature is not valid base64")
    if len(signature) != 64:
        return ArtifactTrustDecision(False, "artifact signature has invalid length")
    try:
        public_key.verify(signature, artifact_signature_payload(reference, provenance))
    except InvalidSignature:
        return ArtifactTrustDecision(False, "artifact signature verification failed")
    return ArtifactTrustDecision(True, "")


def _parse_provenance_document(
    document: object,
    reference: ArtifactReference,
) -> tuple[ArtifactProvenance | None, str]:
    expected = {"version", "artifact", "provenance"}
    if not isinstance(document, dict) or set(document) != expected:
        return None, "artifact provenance has invalid fields"
    if document["version"] != _PROVENANCE_VERSION:
        return None, "artifact provenance has unsupported version"
    expected_artifact = artifact_provenance_document(reference, None)["artifact"]
    if document["artifact"] != expected_artifact:
        return None, "artifact provenance identity does not match"
    provenance = document["provenance"]
    if provenance is None:
        return None, ""
    fields = {"publisher", "keyId", "source", "publishedAtMs", "signature"}
    if not isinstance(provenance, dict) or set(provenance) != fields:
        return None, "artifact provenance signature has invalid fields"
    return (
        ArtifactProvenance(
            publisher=provenance["publisher"],
            key_id=provenance["keyId"],
            source=provenance["source"],
            published_at_ms=provenance["publishedAtMs"],
            signature=provenance["signature"],
        ),
        "",
    )


def _root_error(root: ArtifactTrustRoot) -> str:
    if not isinstance(root.publisher, str) or not _IDENTITY.fullmatch(root.publisher):
        return f"invalid artifact publisher: {root.publisher!r}"
    if not isinstance(root.key_id, str) or not _IDENTITY.fullmatch(root.key_id):
        return f"invalid artifact publisher key id: {root.key_id!r}"
    if not isinstance(root.public_key, bytes) or len(root.public_key) != 32:
        return "artifact trust root must contain a 32-byte Ed25519 public key"
    return ""


def _provenance_error(provenance: ArtifactProvenance) -> str:
    if not isinstance(provenance.publisher, str) or not _IDENTITY.fullmatch(provenance.publisher):
        return f"invalid artifact publisher: {provenance.publisher!r}"
    if not isinstance(provenance.key_id, str) or not _IDENTITY.fullmatch(provenance.key_id):
        return f"invalid artifact publisher key id: {provenance.key_id!r}"
    if not isinstance(provenance.source, str) or not (
        1 <= len(provenance.source) <= _MAX_SOURCE_LENGTH
    ):
        return "invalid artifact provenance source"
    if type(provenance.published_at_ms) is not int or not (
        0 <= provenance.published_at_ms <= _MAX_TIMESTAMP_MS
    ):
        return "invalid artifact provenance timestamp"
    if not isinstance(provenance.signature, str):
        return "invalid artifact signature"
    return ""
