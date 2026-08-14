"""Command-line entry point for document-model production."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .document_model_contracts import bundled_corpus_path
from .document_model_installation import install_document_model
from .document_model_types import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_BINDINGS_ROOT,
    DEFAULT_BUILD_ROOT,
    DEFAULT_SOURCE_ROOT,
    RECIPE_ID,
    DocumentModelError,
)
from .recipe_fetch import fetch_model_sources
from .recipe_model import ModelRecipeError
from .recipe_registry import resolve_model_recipe_path


def main(
    argv: list[str] | None = None,
    *,
    module_file: str = __file__,
    resolver=resolve_model_recipe_path,
    fetcher=fetch_model_sources,
    installer=install_document_model,
    corpus_provider=None,
) -> int:
    parser = argparse.ArgumentParser(
        prog="omnitensor-install-document-model",
        description=(
            "Fetch, build, GPU-gate, and install pinned BGE-small for Document Intelligence"
        ),
    )
    parser.add_argument("--accept-license", required=True)
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--build-root", default=DEFAULT_BUILD_ROOT)
    parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--bindings-root", default=DEFAULT_BINDINGS_ROOT)
    parser.add_argument("--gpu-device", type=int, default=None)
    arguments = parser.parse_args(argv)
    try:
        recipe_path = resolver(RECIPE_ID)
        source = fetcher(
            recipe_path,
            Path(arguments.source_root).expanduser(),
            accepted_license=arguments.accept_license,
        )
        corpus_path = corpus_provider() if corpus_provider else bundled_corpus_path(module_file)
        installed = installer(
            source,
            corpus_path=corpus_path,
            build_root=Path(arguments.build_root).expanduser(),
            artifact_root=Path(arguments.artifact_root).expanduser(),
            bindings_root=Path(arguments.bindings_root).expanduser(),
            device_index=arguments.gpu_device,
        )
    except (DocumentModelError, ModelRecipeError, OSError, ValueError) as error:
        print(f"document model installation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(installed.document(), indent=2))
    return 0
