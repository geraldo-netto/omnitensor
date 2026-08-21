"""One-shot installation of the pinned generation and retrieval artifacts.

Named for what it installs rather than for whose models it happens to pin: the
generation models are immutable digests in this file and the workloads that use
them name no vendor at all.
"""

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

DEFAULT_GENERATION_MODEL_SIZE_BYTES = 5_168_653_536
DEFAULT_GENERATION_MODEL_REFERENCE = ArtifactReference(
    "qwen3-5-9b-iq4-xs",
    "1.0.0",
    "gguf",
    "7e918aeca06c52bcb528ea6b04b4ec957e75ee8c0a73138854c0dfcf371ea429",
    (),
    "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/"
    "3885219b6810b007914f3a7950a8d1b469d598a5/Qwen3.5-9B-IQ4_XS.gguf",
    "Apache-2.0",
)
GENERATION_MODEL_SIZE_BYTES = 5_027_783_488
GENERATION_MODEL_REFERENCE = ArtifactReference(
    "qwen3-8b-q4-k-m",
    "1.0.0",
    "gguf",
    "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785",
    (),
    "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/"
    "7c41481f57cb95916b40956ab2f0b139b296d974/Qwen3-8B-Q4_K_M.gguf",
    "Apache-2.0",
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
    "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/"
    "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/onnx/model.onnx",
    "MIT",
)


# Compatibility alias retained for callers that catch the old public name.
GenerationInstallationError = PinnedArtifactInstallationError

_SOURCE_ERRORS = PinnedSourceErrors(
    "provider artifact is unavailable",
    "provider artifacts must be absolute regular files",
    "Qwen GGUF size does not match its pinned release",
    "provider artifact digest does not match its manifest",
)


@dataclass(frozen=True, slots=True)
class GenerationArtifactSources:
    qwen_model: Path
    default_qwen_model: Path
    bge_param: Path
    bge_bin: Path
    bge_tokenizer: Path


def install_generation_artifacts(
    artifact_root: Path,
    sources: GenerationArtifactSources,
    *,
    accepted_model_license: str,
    accepted_bge_license: str,
) -> dict[str, object]:
    """Verify every input first, then atomically install each immutable version."""
    if accepted_model_license != "Apache-2.0" or accepted_bge_license != "MIT":
        raise GenerationInstallationError(
            "license acceptance must explicitly name Apache-2.0 and MIT"
        )
    expected = (
        (
            sources.default_qwen_model,
            DEFAULT_GENERATION_MODEL_REFERENCE.sha256,
            DEFAULT_GENERATION_MODEL_SIZE_BYTES,
        ),
        (sources.qwen_model, GENERATION_MODEL_REFERENCE.sha256, GENERATION_MODEL_SIZE_BYTES),
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
    default_qwen = installer.install(
        DEFAULT_GENERATION_MODEL_REFERENCE,
        sources.default_qwen_model,
    )
    qwen = installer.install(GENERATION_MODEL_REFERENCE, sources.qwen_model)
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
            _installed_document(DEFAULT_GENERATION_MODEL_REFERENCE, default_qwen.path),
            _installed_document(GENERATION_MODEL_REFERENCE, qwen.path),
            _installed_document(BGE_REFERENCE, bge.path),
        ],
        "licensesAccepted": {"bge": "MIT", "qwen": "Apache-2.0"},
    }


def _verify_source(path: Path, digest: str, exact_size: int | None) -> None:
    verify_pinned_source(
        path,
        digest,
        exact_size,
        errors=_SOURCE_ERRORS,
        error_type=GenerationInstallationError,
        digest_file=file_digest,
    )


def _installed_document(reference: ArtifactReference, path: Path) -> dict[str, object]:
    return installed_artifact_document(reference, path, include_companions=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify and install pinned artifacts for Qwen workload providers"
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--default-qwen-model", type=Path, required=True)
    parser.add_argument("--qwen-model", type=Path, required=True)
    parser.add_argument("--bge-param", type=Path, required=True)
    parser.add_argument("--bge-bin", type=Path, required=True)
    parser.add_argument("--bge-tokenizer", type=Path, required=True)
    parser.add_argument("--accept-model-license", required=True)
    parser.add_argument("--accept-bge-license", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    def invoke(arguments: argparse.Namespace) -> object:
        return install_generation_artifacts(
            arguments.artifact_root,
            GenerationArtifactSources(
                arguments.qwen_model,
                arguments.default_qwen_model,
                arguments.bge_param,
                arguments.bge_bin,
                arguments.bge_tokenizer,
            ),
            accepted_model_license=arguments.accept_model_license,
            accepted_bge_license=arguments.accept_bge_license,
        )

    run_installation_cli(
        _parser(),
        argv,
        invoke,
        errors=(GenerationInstallationError, ArtifactInstallationError, OSError),
    )


if __name__ == "__main__":  # pragma: no cover - console entry point
    main()
