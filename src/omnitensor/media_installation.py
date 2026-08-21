"""One-shot installation of pinned multimedia transcription artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from omnitensor.preparation import file_digest

from .plugins.artifact_installation import (
    ArtifactInstallationError,
    ArtifactInstaller,
    PinnedArtifactInstallationError,
    PinnedSourceErrors,
    installed_artifact_document,
    run_installation_cli,
    verify_pinned_source,
)
from .plugins.artifacts import ArtifactReference

VISION_MODEL_SIZE_BYTES = 4_683_072_032
VISION_PROJECTOR_SIZE_BYTES = 1_354_162_912
SPEECH_MODEL_SIZE_BYTES = 487_601_967
# The default vision model since OMNI-0588; the pins mirror
# generation-models/qwen3-5-9b-vl.json. Without it here, a documented fresh
# install produced a manifest-declared default nothing had mounted, and the
# whole media worker — whisper included — refused to start (OMNI-0590).
DEFAULT_VISION_MODEL_SIZE_BYTES = 5_680_522_464
DEFAULT_VISION_PROJECTOR_SIZE_BYTES = 918_166_080
DEFAULT_VISION_REFERENCE = ArtifactReference(
    "qwen3-5-9b-q4-k-m",
    "1.0.0",
    "gguf",
    "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8",
    (("mmproj.gguf", "f70dc3509053962b0d0d3ee8a7eacebf5d60aa560cad78254ae8698516ae029f"),),
)
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


# Compatibility alias retained for callers that catch the old public name.
MediaInstallationError = PinnedArtifactInstallationError

_SOURCE_ERRORS = PinnedSourceErrors(
    "media artifact is unavailable",
    "media artifacts must be absolute regular files",
    "media artifact size does not match its pinned release",
    "media artifact digest does not match its manifest",
)


@dataclass(frozen=True, slots=True)
class MediaArtifactSources:
    vision_model: Path
    vision_projector: Path
    speech_model: Path
    default_vision_model: Path
    default_vision_projector: Path


def install_media_artifacts(
    artifact_root: Path,
    sources: MediaArtifactSources,
    *,
    accepted_model_license: str,
    accepted_whisper_license: str,
) -> dict[str, object]:
    """Verify every source first, then atomically install both model families."""
    if accepted_model_license != "Apache-2.0" or accepted_whisper_license != "MIT":
        raise MediaInstallationError("license acceptance must explicitly name Apache-2.0 and MIT")
    expected = (
        (sources.vision_model, VISION_REFERENCE.sha256, VISION_MODEL_SIZE_BYTES),
        (
            sources.vision_projector,
            VISION_REFERENCE.declared_companions["mmproj.gguf"],
            VISION_PROJECTOR_SIZE_BYTES,
        ),
        (
            sources.default_vision_model,
            DEFAULT_VISION_REFERENCE.sha256,
            DEFAULT_VISION_MODEL_SIZE_BYTES,
        ),
        (
            sources.default_vision_projector,
            DEFAULT_VISION_REFERENCE.declared_companions["mmproj.gguf"],
            DEFAULT_VISION_PROJECTOR_SIZE_BYTES,
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
    default_vision = installer.install(
        DEFAULT_VISION_REFERENCE,
        sources.default_vision_model,
        companions={"mmproj.gguf": sources.default_vision_projector},
    )
    speech = installer.install(SPEECH_REFERENCE, sources.speech_model)
    return {
        "version": 1,
        "artifacts": [
            _installed_document(VISION_REFERENCE, vision.path),
            _installed_document(DEFAULT_VISION_REFERENCE, default_vision.path),
            _installed_document(SPEECH_REFERENCE, speech.path),
        ],
        "licensesAccepted": {"qwen": "Apache-2.0", "whisper": "MIT"},
    }


def _verify_source(path: Path, digest: str, exact_size: int) -> None:
    verify_pinned_source(
        path,
        digest,
        exact_size,
        errors=_SOURCE_ERRORS,
        error_type=MediaInstallationError,
        digest_file=file_digest,
    )


def _installed_document(reference: ArtifactReference, path: Path) -> dict[str, object]:
    return installed_artifact_document(reference, path, include_companions=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify and install pinned multimedia transcription artifacts"
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--vision-model", type=Path, required=True)
    parser.add_argument("--vision-projector", type=Path, required=True)
    parser.add_argument("--default-vision-model", type=Path, required=True)
    parser.add_argument("--default-vision-projector", type=Path, required=True)
    parser.add_argument("--speech-model", type=Path, required=True)
    parser.add_argument("--accept-model-license", required=True)
    parser.add_argument("--accept-whisper-license", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    def invoke(arguments: argparse.Namespace) -> object:
        return install_media_artifacts(
            arguments.artifact_root,
            MediaArtifactSources(
                arguments.vision_model,
                arguments.vision_projector,
                arguments.speech_model,
                arguments.default_vision_model,
                arguments.default_vision_projector,
            ),
            accepted_model_license=arguments.accept_model_license,
            accepted_whisper_license=arguments.accept_whisper_license,
        )

    run_installation_cli(
        _parser(),
        argv,
        invoke,
        errors=(MediaInstallationError, ArtifactInstallationError, OSError),
    )


if __name__ == "__main__":  # pragma: no cover - console entry point
    main()
