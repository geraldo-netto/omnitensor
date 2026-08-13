"""One-shot installation of pinned Qwen and document-retrieval artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .plugins.artifact_installation import ArtifactInstallationError, ArtifactInstaller
from .plugins.artifacts import ArtifactReference
from .preparation import file_digest

QWEN_SIZE_BYTES = 5_027_783_488
QWEN_REFERENCE = ArtifactReference(
    "qwen3-8b-q4-k-m",
    "1.0.0",
    "gguf",
    "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785",
)
BGE_REFERENCE = ArtifactReference(
    "bge-small-en-v1-5-ask-gpu",
    "1.0.0",
    "ncnn",
    "8a27ec7beb314ba2824a8c03ced33f922f079a8d9dc8aefb712d4b1c165d33bb",
    (
        (
            "model.bin",
            "44034070d5e86e6be4829dc4a69c1e959ee895b07877d89b34c98d2adcf9e994",
        ),
        (
            "tokenizer.json",
            "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
        ),
    ),
)


class QwenInstallationError(RuntimeError):
    """Stable refusal before any artifact is installed."""


@dataclass(frozen=True, slots=True)
class QwenArtifactSources:
    qwen_model: Path
    bge_param: Path
    bge_bin: Path
    bge_tokenizer: Path


def install_qwen_artifacts(
    artifact_root: Path,
    sources: QwenArtifactSources,
    *,
    accepted_qwen_license: str,
    accepted_bge_license: str,
) -> dict[str, object]:
    """Verify every input first, then atomically install each immutable version."""
    if accepted_qwen_license != "Apache-2.0" or accepted_bge_license != "MIT":
        raise QwenInstallationError("license acceptance must explicitly name Apache-2.0 and MIT")
    expected = (
        (sources.qwen_model, QWEN_REFERENCE.sha256, QWEN_SIZE_BYTES),
        (sources.bge_param, BGE_REFERENCE.sha256, None),
        (
            sources.bge_bin,
            BGE_REFERENCE.declared_companions["model.bin"],
            None,
        ),
        (
            sources.bge_tokenizer,
            BGE_REFERENCE.declared_companions["tokenizer.json"],
            None,
        ),
    )
    for path, digest, exact_size in expected:
        _verify_source(path, digest, exact_size)
    installer = ArtifactInstaller(Path(artifact_root))
    qwen = installer.install(QWEN_REFERENCE, sources.qwen_model)
    bge = installer.install(
        BGE_REFERENCE,
        sources.bge_param,
        companions={
            "model.bin": sources.bge_bin,
            "tokenizer.json": sources.bge_tokenizer,
        },
    )
    return {
        "version": 1,
        "artifacts": [
            _installed_document(QWEN_REFERENCE, qwen.path),
            _installed_document(BGE_REFERENCE, bge.path),
        ],
        "licensesAccepted": {"bge": "MIT", "qwen": "Apache-2.0"},
    }


def _verify_source(path: Path, digest: str, exact_size: int | None) -> None:
    candidate = Path(path)
    try:
        status = candidate.stat()
    except OSError as error:
        raise QwenInstallationError("provider artifact is unavailable") from error
    if not candidate.is_absolute() or not candidate.is_file():
        raise QwenInstallationError("provider artifacts must be absolute regular files")
    if exact_size is not None and status.st_size != exact_size:
        raise QwenInstallationError("Qwen GGUF size does not match its pinned release")
    if file_digest(candidate) != digest:
        raise QwenInstallationError("provider artifact digest does not match its manifest")


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
        description="Verify and install pinned artifacts for Qwen workload providers"
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--qwen-model", type=Path, required=True)
    parser.add_argument("--bge-param", type=Path, required=True)
    parser.add_argument("--bge-bin", type=Path, required=True)
    parser.add_argument("--bge-tokenizer", type=Path, required=True)
    parser.add_argument("--accept-qwen-license", required=True)
    parser.add_argument("--accept-bge-license", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    try:
        document = install_qwen_artifacts(
            arguments.artifact_root,
            QwenArtifactSources(
                arguments.qwen_model,
                arguments.bge_param,
                arguments.bge_bin,
                arguments.bge_tokenizer,
            ),
            accepted_qwen_license=arguments.accept_qwen_license,
            accepted_bge_license=arguments.accept_bge_license,
        )
    except (QwenInstallationError, ArtifactInstallationError, OSError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(document, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":  # pragma: no cover - console entry point
    main()
