"""Verified installation of the optional selected-text Hebrew model."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

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
from .preparation import file_digest

DICTALM_SIZE_BYTES = 4_374_991_808
DICTALM_REFERENCE = ArtifactReference(
    "dictalm2-hebrew-q4-k-m",
    "1.0.0",
    "gguf",
    "dc53cc29a30444677a7760af31f807a61999477c526cdbe6552803259a94c735",
)


# Compatibility alias retained for callers that catch the old public name.
HebrewInstallationError = PinnedArtifactInstallationError

_SOURCE_ERRORS = PinnedSourceErrors(
    "DictaLM artifact is unavailable",
    "DictaLM artifact must be an absolute regular file",
    "DictaLM GGUF size does not match its pinned release",
    "DictaLM digest does not match its manifest",
)


def install_hebrew_translation_model(
    artifact_root: Path,
    model_path: Path,
    *,
    accepted_license: str,
) -> dict[str, object]:
    if accepted_license != "Apache-2.0":
        raise HebrewInstallationError("license acceptance must explicitly name Apache-2.0")
    source = Path(model_path)
    verify_pinned_source(
        source,
        DICTALM_REFERENCE.sha256,
        DICTALM_SIZE_BYTES,
        errors=_SOURCE_ERRORS,
        error_type=HebrewInstallationError,
        digest_file=file_digest,
    )
    installed = ArtifactInstaller(Path(artifact_root)).install(DICTALM_REFERENCE, source)
    return {
        "version": 1,
        "artifact": installed_artifact_document(
            DICTALM_REFERENCE,
            installed.path,
            include_companions=False,
        ),
        "licenseAccepted": "Apache-2.0",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify and install the pinned Hebrew translation model"
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--accept-license", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    def invoke(arguments: argparse.Namespace) -> object:
        return install_hebrew_translation_model(
            arguments.artifact_root,
            arguments.model,
            accepted_license=arguments.accept_license,
        )

    run_installation_cli(
        _parser(),
        argv,
        invoke,
        errors=(HebrewInstallationError, ArtifactInstallationError, OSError),
    )


if __name__ == "__main__":  # pragma: no cover - console entry point
    main()
