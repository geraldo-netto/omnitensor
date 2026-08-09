"""Verified artifact installation with atomic activation and rollback."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from omnitensor.atomicio import write_json_atomic

from .artifact_trust import (
    ArtifactProvenance,
    ArtifactTrustVerifier,
    artifact_provenance_document,
)
from .artifacts import (
    DEFAULT_MAX_ARTIFACT_BYTES,
    ArtifactReference,
    ArtifactResolution,
    ArtifactResolver,
    artifact_filename,
    artifact_reference_error,
)

_ACTIVATION_FILE = "activation.json"
_ARTIFACT_METADATA_FILE = "artifact.json"
_DOCUMENT_VERSION = 1
_READ_CHUNK_BYTES = 1024 * 1024


class ArtifactInstallationError(RuntimeError):
    """Stable installation failure safe to surface through orchestration."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class ArtifactActivation:
    """One active version and its single deterministic rollback target."""

    active: ArtifactReference | None
    rollback: ArtifactReference | None


@dataclass(frozen=True, slots=True)
class ArtifactInstallation:
    """Successful activation result."""

    reference: ArtifactReference
    path: Path
    previous: ArtifactReference | None


def artifact_reference_document(reference: ArtifactReference) -> dict[str, object]:
    """Stable JSON representation used by activation and cache metadata."""
    return {
        "id": reference.id,
        "version": reference.version,
        "format": reference.format,
        "sha256": reference.sha256,
    }


def artifact_reference_from_document(document: object) -> ArtifactReference:
    """Parse an exact stored reference or reject corrupt state."""
    if not isinstance(document, dict) or set(document) != {"id", "version", "format", "sha256"}:
        raise ArtifactInstallationError("metadata-invalid", "artifact reference has invalid fields")
    reference = ArtifactReference(
        id=document["id"],
        version=document["version"],
        format=document["format"],
        sha256=document["sha256"],
    )
    invalid = artifact_reference_error(reference)
    if invalid:
        raise ArtifactInstallationError("metadata-invalid", invalid)
    return reference


class ArtifactInstaller:
    """Import verified files into an immutable version store."""

    def __init__(
        self,
        root: Path,
        *,
        max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
        trust_verifier: ArtifactTrustVerifier | None = None,
    ):
        if max_artifact_bytes < 1:
            raise ValueError("max_artifact_bytes must be positive")
        self._root = Path(root).resolve()
        self._max_artifact_bytes = max_artifact_bytes
        self._trust_verifier = trust_verifier

    def install(
        self,
        reference: ArtifactReference,
        source: Path,
        *,
        provenance: ArtifactProvenance | None = None,
    ) -> ArtifactInstallation:
        """Stage and verify ``source`` before atomically activating its version."""
        invalid = artifact_reference_error(reference)
        if invalid:
            raise ArtifactInstallationError("reference-invalid", invalid)
        artifact_root = self._artifact_root(reference.id)
        current = self.activation(reference.id)
        stage = Path(tempfile.mkdtemp(prefix=".install-", dir=artifact_root))
        try:
            staged_path = stage / artifact_filename(reference.format)
            self._copy_verified(Path(source), staged_path, reference.sha256)
            self._verify_trust(reference, provenance)
            if self._trust_verifier is not None:
                write_json_atomic(
                    stage / "provenance.json",
                    artifact_provenance_document(reference, provenance),
                    prefix=".artifact-provenance-",
                )
            write_json_atomic(
                stage / _ARTIFACT_METADATA_FILE,
                {"version": _DOCUMENT_VERSION, "artifact": artifact_reference_document(reference)},
                prefix=".artifact-metadata-",
            )
            _fsync_directory(stage)
            active_path = self._activate_version(stage, artifact_root, reference)
            self._verify_stored_trust(reference, active_path.parent)
            previous = current.active if current.active != reference else current.rollback
            next_state = ArtifactActivation(reference, previous)
            try:
                self._write_activation(reference.id, next_state)
            except OSError as error:
                raise ArtifactInstallationError(
                    "activation-failed",
                    f"could not persist active version: {error}",
                ) from error
            return ArtifactInstallation(reference, active_path, current.active)
        finally:
            _remove_stage(stage)

    def activation(self, artifact_id: str) -> ArtifactActivation:
        """Read strict activation state; absence means no version is active."""
        artifact_root = self._artifact_root(artifact_id)
        path = artifact_root / _ACTIVATION_FILE
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return ArtifactActivation(None, None)
        except (OSError, ValueError) as error:
            raise ArtifactInstallationError(
                "activation-state-invalid",
                f"cannot read {path}: {error}",
            ) from error
        return _parse_activation(document, artifact_id)

    def resolve_active(self, artifact_id: str) -> ArtifactResolution:
        """Resolve the active pointer through the same immutable digest gate."""
        active = self.activation(artifact_id).active
        if active is None:
            return ArtifactResolution(
                False, None, f"artifact has no active version: {artifact_id}", 0
            )
        trust = self._stored_trust(active)
        if trust:
            return ArtifactResolution(False, None, trust, 0)
        return ArtifactResolver(
            [self._root],
            max_artifact_bytes=self._max_artifact_bytes,
        ).resolve(active)

    def rollback(self, artifact_id: str) -> ArtifactInstallation:
        """Atomically swap active and rollback versions after re-verification."""
        current = self.activation(artifact_id)
        if current.active is None or current.rollback is None:
            raise ArtifactInstallationError(
                "rollback-unavailable",
                f"artifact has no rollback version: {artifact_id}",
            )
        trust = self._stored_trust(current.rollback)
        if trust:
            raise ArtifactInstallationError("rollback-invalid", trust)
        resolution = ArtifactResolver(
            [self._root],
            max_artifact_bytes=self._max_artifact_bytes,
        ).resolve(current.rollback)
        if not resolution.ready or resolution.path is None:
            raise ArtifactInstallationError("rollback-invalid", resolution.reason)
        self._write_activation(
            artifact_id,
            ArtifactActivation(current.rollback, current.active),
        )
        return ArtifactInstallation(current.rollback, resolution.path, current.active)

    def _verify_trust(
        self,
        reference: ArtifactReference,
        provenance: ArtifactProvenance | None,
    ) -> None:
        if self._trust_verifier is None:
            return
        decision = self._trust_verifier.verify(reference, provenance)
        if not decision.trusted:
            raise ArtifactInstallationError("artifact-untrusted", decision.reason)

    def _stored_trust(self, reference: ArtifactReference) -> str:
        if self._trust_verifier is None:
            return ""
        version_root = self._root / reference.id / reference.version
        decision = self._trust_verifier.verify_installed(reference, version_root)
        return "" if decision.trusted else decision.reason

    def _verify_stored_trust(self, reference: ArtifactReference, version_root: Path) -> None:
        if self._trust_verifier is None:
            return
        decision = self._trust_verifier.verify_installed(reference, version_root)
        if not decision.trusted:
            raise ArtifactInstallationError("artifact-untrusted", decision.reason)

    def _artifact_root(self, artifact_id: str) -> Path:
        probe = ArtifactReference(artifact_id, "0.0.0", "onnx", "0" * 64)
        invalid = artifact_reference_error(probe)
        if invalid:
            raise ArtifactInstallationError("reference-invalid", invalid)
        self._root.mkdir(parents=True, exist_ok=True)
        artifact_root = self._root / artifact_id
        artifact_root.mkdir(exist_ok=True)
        if artifact_root.resolve() != artifact_root:
            raise ArtifactInstallationError(
                "store-path-invalid",
                f"artifact directory escapes store root: {artifact_id}",
            )
        return artifact_root

    def _copy_verified(self, source: Path, destination: Path, expected_digest: str) -> int:
        try:
            source_fd = _open_regular_file(source)
        except OSError as error:
            raise ArtifactInstallationError(
                "source-invalid", f"cannot open artifact: {error}"
            ) from error
        digest = hashlib.sha256()
        total = 0
        try:
            with os.fdopen(source_fd, "rb") as input_stream, destination.open("xb") as output:
                while chunk := input_stream.read(_READ_CHUNK_BYTES):
                    total += len(chunk)
                    if total > self._max_artifact_bytes:
                        raise ArtifactInstallationError(
                            "artifact-too-large",
                            f"artifact exceeds {self._max_artifact_bytes} bytes",
                        )
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
        except BaseException:
            with contextlib.suppress(OSError):
                destination.unlink()
            raise
        if digest.hexdigest() != expected_digest:
            destination.unlink()
            raise ArtifactInstallationError("digest-mismatch", "artifact sha256 mismatch")
        return total

    def _activate_version(
        self,
        stage: Path,
        artifact_root: Path,
        reference: ArtifactReference,
    ) -> Path:
        destination = artifact_root / reference.version
        active_path = destination / artifact_filename(reference.format)
        if destination.exists():
            resolution = ArtifactResolver(
                [self._root],
                max_artifact_bytes=self._max_artifact_bytes,
            ).resolve(reference)
            if resolution.ready and resolution.path is not None:
                return resolution.path
            raise ArtifactInstallationError(
                "version-conflict",
                f"installed version is not the requested immutable artifact: {reference.version}",
            )
        try:
            os.replace(stage, destination)
        except OSError as error:
            raise ArtifactInstallationError(
                "activation-failed",
                f"could not activate artifact version: {error}",
            ) from error
        _fsync_directory(artifact_root)
        return active_path.resolve()

    def _write_activation(self, artifact_id: str, state: ArtifactActivation) -> None:
        payload = {
            "version": _DOCUMENT_VERSION,
            "artifactId": artifact_id,
            "active": (
                artifact_reference_document(state.active) if state.active is not None else None
            ),
            "rollback": (
                artifact_reference_document(state.rollback) if state.rollback is not None else None
            ),
        }
        write_json_atomic(
            self._artifact_root(artifact_id) / _ACTIVATION_FILE,
            payload,
            prefix=".activation-",
        )


def _parse_activation(document: object, artifact_id: str) -> ArtifactActivation:
    expected = {"version", "artifactId", "active", "rollback"}
    if not isinstance(document, dict) or set(document) != expected:
        raise ArtifactInstallationError("activation-state-invalid", "activation has invalid fields")
    if document["version"] != _DOCUMENT_VERSION or document["artifactId"] != artifact_id:
        raise ArtifactInstallationError(
            "activation-state-invalid", "activation identity is invalid"
        )
    active = (
        artifact_reference_from_document(document["active"])
        if document["active"] is not None
        else None
    )
    rollback = (
        artifact_reference_from_document(document["rollback"])
        if document["rollback"] is not None
        else None
    )
    if active is not None and active.id != artifact_id:
        raise ArtifactInstallationError(
            "activation-state-invalid", "active artifact id does not match"
        )
    if rollback is not None and rollback.id != artifact_id:
        raise ArtifactInstallationError(
            "activation-state-invalid", "rollback artifact id does not match"
        )
    return ArtifactActivation(active, rollback)


def _open_regular_file(path: Path) -> int:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise OSError("artifact source is not a regular file")
    return descriptor


def _remove_stage(stage: Path) -> None:
    if not stage.exists():
        return
    for entry in stage.iterdir():
        with contextlib.suppress(OSError):
            entry.unlink()
    with contextlib.suppress(OSError):
        stage.rmdir()


def _fsync_directory(directory: Path) -> None:
    with contextlib.suppress(OSError):
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
