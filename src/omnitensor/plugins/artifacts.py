"""Immutable model-artifact lookup inside explicitly allowlisted roots."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Qwen2.5-VL-7B Q4_K_M is 4,683,072,032 bytes. Keep the service-wide bound
# finite while admitting that pinned vision model and no unbounded input.
DEFAULT_MAX_ARTIFACT_BYTES = 6 * 1024 * 1024 * 1024
_IDENTIFIER = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_VERSION = re.compile(r"^(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_FILENAMES = {
    "tflite-edgetpu": "model.tflite",
    "tflite": "model.tflite",
    "onnx": "model.onnx",
    "openvino": "model.xml",
    "ncnn": "model.param",
    "gguf": "model.gguf",
    "ggml-whisper": "model.bin",
}


# Formats whose primary file is not the whole model.  An ncnn ".param" is only
# the graph; every weight lives in the sibling ".bin", so installing the param
# alone produces an artifact that resolves ready and cannot be executed.
COMPANION_FILENAMES: dict[str, tuple[str, ...]] = {
    "ncnn": ("model.bin",),
    # OpenVINO IR is a graph in .xml and its weights in a sibling .bin, so it
    # has exactly the same defect ncnn had: install the primary alone and the
    # artifact resolves ready while the weights are absent.
    "openvino": ("model.bin",),
}
# How to find each companion beside the primary source file.
COMPANION_SOURCE_SUFFIXES: dict[str, dict[str, str]] = {
    "ncnn": {"model.bin": ".bin"},
    "openvino": {"model.bin": ".bin"},
}


# Files a publisher may install beside any model, for every format, and which
# no format requires.  A labels list is the case: it belongs with the weights
# it names — a list that drifted from them renames every result — but nothing
# about ncnn or OpenVINO says a model must be a classifier.
OPTIONAL_COMPANION_FILENAMES: frozenset[str] = frozenset(
    {"labels.txt", "mmproj.gguf", "tokenizer.json"}
)


def companion_filenames(model_format: str) -> tuple[str, ...]:
    """The files that must accompany the primary file for this format."""
    return COMPANION_FILENAMES.get(model_format, ())


def permitted_companion_filenames(model_format: str) -> frozenset[str]:
    """Every companion this format accepts, required or not."""
    return frozenset(companion_filenames(model_format)) | OPTIONAL_COMPANION_FILENAMES


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """Manifest identity and expected digests for one model artifact.

    ``companions`` is what the publisher vouches for beyond the primary file.
    Empty means it vouched for nothing else, which for a format that keeps its
    weights in a companion means it vouched for the graph and not the numbers.
    """

    id: str
    version: str
    format: str
    sha256: str
    companions: tuple[tuple[str, str], ...] = ()
    # Where this file came from and under what licence. Empty when the manifest
    # states neither, which is the honest answer for a bundled v1 workload whose
    # model was never published anywhere. A worker that is told nothing must
    # report nothing rather than assert a plausible default.
    source_uri: str = ""
    license_spdx: str = ""

    @property
    def declared_companions(self) -> dict[str, str]:
        return dict(self.companions)

    def unpinned_companions(self) -> tuple[str, ...]:
        """Required companions of this format the manifest does not name."""
        declared = self.declared_companions
        return tuple(name for name in companion_filenames(self.format) if name not in declared)


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
        # Local import avoids preparation -> artifact installer -> resolver at
        # module import time while keeping hashing canonical.
        from omnitensor.preparation import FileDigestTooLargeError, file_digest

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
            digest = file_digest(path, max_bytes=self._max_artifact_bytes)
        except FileDigestTooLargeError as error:
            return ArtifactResolution(
                False,
                None,
                f"artifact exceeds {self._max_artifact_bytes} bytes",
                error.observed_bytes,
            )
        except OSError as error:
            return ArtifactResolution(False, None, f"artifact cannot be read: {error}", 0)
        if digest != reference.sha256:
            return ArtifactResolution(False, None, "artifact sha256 mismatch", size)
        return ArtifactResolution(True, path, "", size)
