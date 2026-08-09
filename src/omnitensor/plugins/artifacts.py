"""Immutable model-artifact lookup inside explicitly allowlisted roots."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MAX_ARTIFACT_BYTES = 512 * 1024 * 1024
_READ_CHUNK_BYTES = 1024 * 1024
_IDENTIFIER = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_VERSION = re.compile(r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_FILENAMES = {
    "tflite-edgetpu": "model.tflite",
    "tflite": "model.tflite",
    "onnx": "model.onnx",
    "openvino": "model.xml",
    "ncnn": "model.param",
}


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """Manifest identity and expected digest for one model artifact."""

    id: str
    version: str
    format: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ArtifactResolution:
    """Readiness result; unavailable artifacts never expose a candidate path."""

    ready: bool
    path: Path | None
    reason: str
    size_bytes: int


def artifact_filename(model_format: str) -> str:
    """Return the conventional primary filename for a supported model format."""
    try:
        return _FILENAMES[model_format]
    except KeyError as error:
        raise ValueError(f"unsupported artifact format: {model_format}") from error


def artifact_reference_error(reference: ArtifactReference) -> str:
    """Return an actionable validation error, or an empty string when valid."""
    if not isinstance(reference.id, str) or not _IDENTIFIER.fullmatch(reference.id):
        return f"invalid artifact id: {reference.id!r}"
    if not isinstance(reference.version, str) or not _VERSION.fullmatch(reference.version):
        return f"invalid artifact version: {reference.version!r}"
    if reference.format not in _FILENAMES:
        return f"unsupported artifact format: {reference.format}"
    if not isinstance(reference.sha256, str) or not _SHA256.fullmatch(reference.sha256):
        return "invalid artifact sha256"
    return ""


class ArtifactResolver:
    """Resolve and verify the first installed artifact in ordered safe roots."""

    def __init__(
        self,
        roots: list[Path] | tuple[Path, ...],
        *,
        max_artifact_bytes: int = DEFAULT_MAX_ARTIFACT_BYTES,
    ):
        if not roots:
            raise ValueError("at least one artifact root is required")
        if max_artifact_bytes < 1:
            raise ValueError("max_artifact_bytes must be positive")
        self._roots = tuple(Path(root).resolve() for root in roots)
        self._max_artifact_bytes = max_artifact_bytes

    def resolve(self, reference: ArtifactReference) -> ArtifactResolution:
        invalid = artifact_reference_error(reference)
        if invalid:
            return ArtifactResolution(False, None, invalid, 0)

        filename = artifact_filename(reference.format)
        for root in self._roots:
            candidate = root / reference.id / reference.version / filename
            try:
                resolved = candidate.resolve(strict=True)
            except FileNotFoundError:
                continue
            try:
                resolved.relative_to(root)
            except ValueError:
                return ArtifactResolution(
                    False,
                    None,
                    f"artifact path escapes allowlisted root: {root}",
                    0,
                )
            return self._verify(resolved, reference)
        return ArtifactResolution(
            False,
            None,
            f"artifact not installed: {reference.id}@{reference.version}",
            0,
        )

    def _verify(
        self,
        path: Path,
        reference: ArtifactReference,
    ) -> ArtifactResolution:
        if not path.is_file():
            return ArtifactResolution(False, None, f"artifact is not a regular file: {path}", 0)
        try:
            size = path.stat().st_size
            if size > self._max_artifact_bytes:
                return ArtifactResolution(
                    False,
                    None,
                    f"artifact exceeds {self._max_artifact_bytes} bytes",
                    size,
                )
            digest, read_size = self._digest(path)
        except OSError as error:
            return ArtifactResolution(False, None, f"artifact cannot be read: {error}", 0)
        if read_size > self._max_artifact_bytes:
            return ArtifactResolution(
                False,
                None,
                f"artifact exceeds {self._max_artifact_bytes} bytes",
                read_size,
            )
        if digest != reference.sha256:
            return ArtifactResolution(False, None, "artifact sha256 mismatch", read_size)
        return ArtifactResolution(True, path, "", read_size)

    def _digest(self, path: Path) -> tuple[str, int]:
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            while chunk := stream.read(_READ_CHUNK_BYTES):
                size += len(chunk)
                if size > self._max_artifact_bytes:
                    break
                digest.update(chunk)
        return digest.hexdigest(), size
