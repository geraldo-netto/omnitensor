"""Machine-local Ed25519 identity for signing qualified model artifacts.

The inference service never needs the private key.  Producer commands call
``load_or_create_publisher_identity`` at the offline publication boundary;
the first call creates the named key atomically and later calls verify and
reuse the same bytes.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import contextlib
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from .atomicio import JsonTooLargeError, read_json_bounded, write_json_atomic
from .plugins.artifact_trust import (
    ArtifactProvenance,
    ArtifactTrustRoot,
    artifact_signature_payload,
)
from .plugins.artifacts import ArtifactReference
from .storelock import store_lock

DEFAULT_PUBLISHER_LABEL = "xpuwlm"
PUBLISHER_PURPOSE = "Sign locally qualified XPU Workload Manager model artifacts"
MAX_KEY_BYTES = 16 * 1024
MAX_TRUST_DOCUMENT_BYTES = 64 * 1024
_LABEL = re.compile(r"[a-z0-9](?:[a-z0-9._-]{0,46}[a-z0-9])?")


class PublisherIdentityError(ValueError):
    """Stable failure at the local signing-identity boundary."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class PublisherIdentityPaths:
    private_key: Path
    public_key: Path
    trust_document: Path


@dataclass(frozen=True, slots=True)
class PublisherIdentity:
    label: str
    publisher: str
    key_id: str
    fingerprint_sha256: str
    paths: PublisherIdentityPaths
    created: bool

    def trust_root(self) -> ArtifactTrustRoot:
        """Load the public key from the strict public trust document."""
        return load_publisher_trust_root(self.paths.trust_document)

    def sign(
        self,
        reference: ArtifactReference,
        source: str,
        *,
        published_at_ms: int | None = None,
    ) -> ArtifactProvenance:
        """Sign one immutable artifact reference with the reusable private key."""
        private_key = _load_private_key(self.paths.private_key)
        timestamp = time.time_ns() // 1_000_000 if published_at_ms is None else published_at_ms
        unsigned = ArtifactProvenance(self.publisher, self.key_id, source, timestamp, "")
        signature = private_key.sign(artifact_signature_payload(reference, unsigned))
        return ArtifactProvenance(
            unsigned.publisher,
            unsigned.key_id,
            unsigned.source,
            unsigned.published_at_ms,
            base64.b64encode(signature).decode("ascii"),
        )


def default_publisher_key_root() -> Path:
    """Return the operator-owned data root; never the service state directory."""
    data_home = os.environ.get("XDG_DATA_HOME")
    base = Path(data_home).expanduser() if data_home else Path.home() / ".local/share"
    return base / "omnitensor/publisher-keys"


def default_publisher_trust_root() -> Path:
    """Return the public, non-secret local trust configuration root."""
    config_home = os.environ.get("XDG_CONFIG_HOME")
    base = Path(config_home).expanduser() if config_home else Path.home() / ".config"
    return base / "omnitensor/artifact-publishers"


def load_or_create_publisher_identity(
    *,
    label: str = DEFAULT_PUBLISHER_LABEL,
    key_root: Path | str | None = None,
    trust_root: Path | str | None = None,
) -> PublisherIdentity:
    """Create one labeled Ed25519 keypair when absent, otherwise verify and reuse it."""
    _validate_label(label)
    keys = Path(key_root) if key_root is not None else default_publisher_key_root()
    trusts = Path(trust_root) if trust_root is not None else default_publisher_trust_root()
    key_directory = _secure_directory(keys / label)
    paths = PublisherIdentityPaths(
        key_directory / f"{label}-ed25519-v1.private.pem",
        key_directory / f"{label}-ed25519-v1.public.pem",
        trusts / f"{label}.json",
    )
    with store_lock(key_directory, ".publisher-key.lock"):
        private_exists = _regular_file_exists(paths.private_key)
        public_exists = _regular_file_exists(paths.public_key)
        if private_exists != public_exists:
            raise PublisherIdentityError(
                "keypair-incomplete", "private and public publisher key files must coexist"
            )
        created = not private_exists
        if created:
            _create_keypair(paths)
        private_key = _load_private_key(paths.private_key)
        public_raw = _verified_public_key(private_key, paths.public_key)
        paths.private_key.chmod(0o600)
        paths.public_key.chmod(0o644)
        identity = _identity(label, paths, public_raw, created)
        _write_or_verify_trust_document(identity, public_raw)
        return identity


def load_publisher_trust_root(path: Path | str) -> ArtifactTrustRoot:
    """Load one bounded, exact public trust document."""
    candidate = Path(path)
    try:
        document = read_json_bounded(candidate, MAX_TRUST_DOCUMENT_BYTES)
    except (OSError, ValueError, JsonTooLargeError) as error:
        raise PublisherIdentityError("trust-invalid", "publisher trust cannot be read") from error
    expected = {
        "version",
        "label",
        "publisher",
        "keyId",
        "algorithm",
        "publicKey",
        "fingerprintSha256",
        "purpose",
    }
    if not isinstance(document, dict) or set(document) != expected or document["version"] != 1:
        raise PublisherIdentityError("trust-invalid", "publisher trust fields are invalid")
    if document["algorithm"] != "Ed25519" or document["purpose"] != PUBLISHER_PURPOSE:
        raise PublisherIdentityError("trust-invalid", "publisher trust semantics are invalid")
    _validate_label(document["label"])
    try:
        public_key = base64.b64decode(document["publicKey"], validate=True)
    except (binascii.Error, TypeError, ValueError) as error:
        raise PublisherIdentityError("trust-invalid", "publisher public key is invalid") from error
    fingerprint = hashlib.sha256(public_key).hexdigest()
    if fingerprint != document["fingerprintSha256"]:
        raise PublisherIdentityError("trust-invalid", "publisher key fingerprint does not match")
    root = ArtifactTrustRoot(document["publisher"], document["keyId"], public_key)
    return root


def setup_main(argv: list[str] | None = None) -> int:
    """Create or inspect the default machine-local publisher identity."""
    parser = argparse.ArgumentParser(
        prog="omnitensor-setup-publisher-key",
        description="Create or reuse a labeled Ed25519 key for qualified model artifacts",
    )
    parser.add_argument("--label", default=DEFAULT_PUBLISHER_LABEL)
    parser.add_argument("--key-root")
    parser.add_argument("--trust-root")
    arguments = parser.parse_args(argv)
    try:
        identity = load_or_create_publisher_identity(
            label=arguments.label,
            key_root=arguments.key_root,
            trust_root=arguments.trust_root,
        )
    except (OSError, PublisherIdentityError, ValueError) as error:
        print(f"publisher key setup failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(_identity_document(identity), indent=2))
    return 0


def _validate_label(label: object) -> None:
    if not isinstance(label, str) or _LABEL.fullmatch(label) is None:
        raise PublisherIdentityError("label-invalid", "publisher label is invalid")


def _secure_directory(path: Path) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        raise PublisherIdentityError("path-invalid", "publisher key root must be absolute")
    candidate.mkdir(parents=True, mode=0o700, exist_ok=True)
    if candidate.is_symlink() or not candidate.is_dir():
        raise PublisherIdentityError("path-invalid", "publisher key root must be a directory")
    candidate.chmod(0o700)
    return candidate.resolve()


def _regular_file_exists(path: Path) -> bool:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(mode):
        raise PublisherIdentityError("key-invalid", f"publisher key is not regular: {path.name}")
    return True


def _create_keypair(paths: PublisherIdentityPaths) -> None:
    private_key = Ed25519PrivateKey.generate()
    private_pem = private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    public_pem = private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    _write_bytes_atomic(paths.private_key, private_pem, 0o600)
    try:
        _write_bytes_atomic(paths.public_key, public_pem, 0o644)
    except BaseException:
        with contextlib.suppress(OSError):
            paths.private_key.unlink()
        raise


def _write_bytes_atomic(path: Path, payload: bytes, mode: int) -> None:
    handle, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        os.fchmod(handle, mode)
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(handle)
        with contextlib.suppress(OSError):
            os.unlink(temporary_name)
        raise


def _load_private_key(path: Path) -> Ed25519PrivateKey:
    payload = _read_regular_bounded(path)
    try:
        private_key = serialization.load_pem_private_key(payload, password=None)
    except (TypeError, ValueError) as error:
        raise PublisherIdentityError("key-invalid", "publisher private key is invalid") from error
    if not isinstance(private_key, Ed25519PrivateKey):
        raise PublisherIdentityError("key-invalid", "publisher private key is not Ed25519")
    return private_key


def _verified_public_key(private_key: Ed25519PrivateKey, path: Path) -> bytes:
    payload = _read_regular_bounded(path)
    try:
        public_key = serialization.load_pem_public_key(payload)
        public_raw = public_key.public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise PublisherIdentityError("key-invalid", "publisher public key is invalid") from error
    expected = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    if public_raw != expected:
        raise PublisherIdentityError("key-invalid", "publisher keypair does not match")
    return public_raw


def _read_regular_bounded(path: Path) -> bytes:
    if not _regular_file_exists(path):
        raise PublisherIdentityError("key-invalid", f"publisher key is missing: {path.name}")
    with path.open("rb") as stream:
        payload = stream.read(MAX_KEY_BYTES + 1)
    if len(payload) > MAX_KEY_BYTES:
        raise PublisherIdentityError("key-invalid", "publisher key is too large")
    return payload


def _identity(
    label: str,
    paths: PublisherIdentityPaths,
    public_raw: bytes,
    created: bool,
) -> PublisherIdentity:
    return PublisherIdentity(
        label,
        f"{label}.local",
        f"{label}-ed25519-v1",
        hashlib.sha256(public_raw).hexdigest(),
        paths,
        created,
    )


def _trust_document(identity: PublisherIdentity, public_raw: bytes) -> dict[str, object]:
    return {
        "version": 1,
        "label": identity.label,
        "publisher": identity.publisher,
        "keyId": identity.key_id,
        "algorithm": "Ed25519",
        "publicKey": base64.b64encode(public_raw).decode("ascii"),
        "fingerprintSha256": identity.fingerprint_sha256,
        "purpose": PUBLISHER_PURPOSE,
    }


def _write_or_verify_trust_document(
    identity: PublisherIdentity,
    public_raw: bytes,
) -> None:
    expected = _trust_document(identity, public_raw)
    path = identity.paths.trust_document
    if path.exists():
        try:
            observed = read_json_bounded(path, MAX_TRUST_DOCUMENT_BYTES)
        except (OSError, ValueError, JsonTooLargeError) as error:
            raise PublisherIdentityError(
                "trust-invalid", "publisher trust cannot be read"
            ) from error
        if observed != expected:
            raise PublisherIdentityError(
                "trust-conflict", "existing publisher trust does not match the keypair"
            )
    else:
        write_json_atomic(path, expected, prefix=".publisher-trust-")
    path.chmod(0o644)


def _identity_document(identity: PublisherIdentity) -> dict[str, object]:
    return {
        "label": identity.label,
        "publisher": identity.publisher,
        "keyId": identity.key_id,
        "algorithm": "Ed25519",
        "fingerprintSha256": identity.fingerprint_sha256,
        "privateKeyPath": str(identity.paths.private_key),
        "publicKeyPath": str(identity.paths.public_key),
        "trustDocumentPath": str(identity.paths.trust_document),
        "created": identity.created,
        "purpose": PUBLISHER_PURPOSE,
        "privateKeyPrinted": False,
    }
