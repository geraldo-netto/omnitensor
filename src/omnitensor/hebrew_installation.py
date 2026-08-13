"""Verified installation of the optional selected-text Hebrew model."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .plugins.artifact_installation import ArtifactInstallationError, ArtifactInstaller
from .plugins.artifacts import ArtifactReference
from .preparation import file_digest

DICTALM_SIZE_BYTES = 4_374_991_808
DICTALM_REFERENCE = ArtifactReference(
    "dictalm2-hebrew-q4-k-m",
    "1.0.0",
    "gguf",
    "dc53cc29a30444677a7760af31f807a61999477c526cdbe6552803259a94c735",
)


class HebrewInstallationError(RuntimeError):
    """Stable refusal before the optional artifact is installed."""


def install_hebrew_translation_model(
    artifact_root: Path,
    model_path: Path,
    *,
    accepted_license: str,
) -> dict[str, object]:
    if accepted_license != "Apache-2.0":
        raise HebrewInstallationError("license acceptance must explicitly name Apache-2.0")
    source = Path(model_path)
    try:
        status = source.stat()
    except OSError as error:
        raise HebrewInstallationError("DictaLM artifact is unavailable") from error
    if not source.is_absolute() or not source.is_file():
        raise HebrewInstallationError("DictaLM artifact must be an absolute regular file")
    if status.st_size != DICTALM_SIZE_BYTES:
        raise HebrewInstallationError("DictaLM GGUF size does not match its pinned release")
    if file_digest(source) != DICTALM_REFERENCE.sha256:
        raise HebrewInstallationError("DictaLM digest does not match its manifest")
    installed = ArtifactInstaller(Path(artifact_root)).install(DICTALM_REFERENCE, source)
    return {
        "version": 1,
        "artifact": {
            "id": DICTALM_REFERENCE.id,
            "version": DICTALM_REFERENCE.version,
            "format": DICTALM_REFERENCE.format,
            "sha256": DICTALM_REFERENCE.sha256,
            "path": str(installed.path),
        },
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
    arguments = _parser().parse_args(argv)
    try:
        document = install_hebrew_translation_model(
            arguments.artifact_root,
            arguments.model,
            accepted_license=arguments.accept_license,
        )
    except (HebrewInstallationError, ArtifactInstallationError, OSError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(document, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":  # pragma: no cover - console entry point
    main()
