"""One-shot installation of pinned multimedia transcription artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .plugins.artifact_installation import ArtifactInstallationError, ArtifactInstaller
from .plugins.artifacts import ArtifactReference
from .preparation import file_digest

VISION_MODEL_SIZE_BYTES = 4_683_072_032
VISION_PROJECTOR_SIZE_BYTES = 1_354_162_912
SPEECH_MODEL_SIZE_BYTES = 487_601_967
VISION_REFERENCE = ArtifactReference(
    "qwen2-5-vl-7b-instruct",
    "1.0.0",
    "gguf",
    "9258bf05b12686d097ff3b6b18d968ab393649780aa2b3cd67fec43d50554392",
    (("mmproj.gguf", "c24a7f5fcfc68286f0a217023b6738e73bea4f11787a43e8238d4bb1b8604cde"),),
)
SPEECH_REFERENCE = ArtifactReference(
    "whisper-small-multilingual",
    "1.0.0",
    "ggml-whisper",
    "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b",
)


class MediaInstallationError(RuntimeError):
    """Stable refusal before any media artifact is installed."""


@dataclass(frozen=True, slots=True)
class MediaArtifactSources:
    vision_model: Path
    vision_projector: Path
    speech_model: Path


def install_media_artifacts(
    artifact_root: Path,
    sources: MediaArtifactSources,
    *,
    accepted_qwen_license: str,
    accepted_whisper_license: str,
) -> dict[str, object]:
    """Verify every source first, then atomically install both model families."""
    if accepted_qwen_license != "Apache-2.0" or accepted_whisper_license != "MIT":
        raise MediaInstallationError("license acceptance must explicitly name Apache-2.0 and MIT")
    expected = (
        (sources.vision_model, VISION_REFERENCE.sha256, VISION_MODEL_SIZE_BYTES),
        (
            sources.vision_projector,
            VISION_REFERENCE.declared_companions["mmproj.gguf"],
            VISION_PROJECTOR_SIZE_BYTES,
        ),
        (sources.speech_model, SPEECH_REFERENCE.sha256, SPEECH_MODEL_SIZE_BYTES),
    )
    for path, digest, exact_size in expected:
        _verify_source(path, digest, exact_size)
    installer = ArtifactInstaller(Path(artifact_root))
    vision = installer.install(
        VISION_REFERENCE,
        sources.vision_model,
        companions={"mmproj.gguf": sources.vision_projector},
    )
    speech = installer.install(SPEECH_REFERENCE, sources.speech_model)
    return {
        "version": 1,
        "artifacts": [
            _installed_document(VISION_REFERENCE, vision.path),
            _installed_document(SPEECH_REFERENCE, speech.path),
        ],
        "licensesAccepted": {"qwen": "Apache-2.0", "whisper": "MIT"},
    }


def _verify_source(path: Path, digest: str, exact_size: int) -> None:
    candidate = Path(path)
    try:
        status = candidate.stat()
    except OSError as error:
        raise MediaInstallationError("media artifact is unavailable") from error
    if not candidate.is_absolute() or not candidate.is_file():
        raise MediaInstallationError("media artifacts must be absolute regular files")
    if status.st_size != exact_size:
        raise MediaInstallationError("media artifact size does not match its pinned release")
    if file_digest(candidate) != digest:
        raise MediaInstallationError("media artifact digest does not match its manifest")


def _installed_document(reference: ArtifactReference, path: Path) -> dict[str, object]:
    return {
        "id": reference.id,
        "version": reference.version,
        "format": reference.format,
        "sha256": reference.sha256,
        "path": str(path),
        "companions": reference.declared_companions,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify and install pinned multimedia transcription artifacts"
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--vision-model", type=Path, required=True)
    parser.add_argument("--vision-projector", type=Path, required=True)
    parser.add_argument("--speech-model", type=Path, required=True)
    parser.add_argument("--accept-qwen-license", required=True)
    parser.add_argument("--accept-whisper-license", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    try:
        document = install_media_artifacts(
            arguments.artifact_root,
            MediaArtifactSources(
                arguments.vision_model,
                arguments.vision_projector,
                arguments.speech_model,
            ),
            accepted_qwen_license=arguments.accept_qwen_license,
            accepted_whisper_license=arguments.accept_whisper_license,
        )
    except (MediaInstallationError, ArtifactInstallationError, OSError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(document, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":  # pragma: no cover - console entry point
    main()
