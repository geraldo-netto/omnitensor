"""Corpus and source contracts for document-model production."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

from ..atomicio import JsonTooLargeError, read_json_bounded
from ..document_model_types import (
    MAX_CORPUS_BYTES,
    QUERY_PREFIX,
    RECIPE_ID,
    DocumentModelError,
)
from .embedding_production import EmbeddingHoldout
from .recipe_model import FetchedModelSource


def source_path(source: FetchedModelSource, role: str) -> Path:
    item = next((item for item in source.recipe.sources if item.role == role), None)
    if item is None:
        raise DocumentModelError("recipe-incompatible", f"recipe has no {role} source")
    return source.root / item.filename


def load_bge_holdout(
    path: Path | str,
    *,
    reader: Callable[[Path, int], object] = read_json_bounded,
) -> tuple[EmbeddingHoldout, tuple[int, ...]]:
    try:
        document = reader(Path(path), MAX_CORPUS_BYTES)
    except (OSError, ValueError, JsonTooLargeError) as error:
        raise DocumentModelError("corpus-invalid", f"cannot read BGE corpus: {error}") from error
    fields = {
        "corpusVersion",
        "id",
        "license",
        "provenance",
        "queries",
        "documents",
        "expectedTopDocument",
    }
    if not isinstance(document, dict) or set(document) != fields or document["corpusVersion"] != 1:
        raise DocumentModelError("corpus-invalid", "BGE corpus fields or version are invalid")
    queries = document["queries"]
    documents = document["documents"]
    expected = document["expectedTopDocument"]
    if (
        not isinstance(queries, list)
        or not isinstance(documents, list)
        or not isinstance(expected, list)
        or len(expected) != len(queries)
        or any(type(index) is not int or not 0 <= index < len(documents) for index in expected)
    ):
        raise DocumentModelError("corpus-invalid", "BGE retrieval expectations are invalid")
    if any(not isinstance(text, str) for text in [*queries, *documents]):
        raise DocumentModelError("corpus-invalid", "BGE corpus texts must be strings")
    corpus_hash = hashlib.sha256(
        json.dumps(
            {"queries": queries, "documents": documents},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    try:
        holdout = EmbeddingHoldout(
            tuple(f"{QUERY_PREFIX}{query}" for query in queries),
            tuple(documents),
            document["license"],
            corpus_hash,
        )
    except (TypeError, ValueError) as error:
        raise DocumentModelError("corpus-invalid", str(error)) from error
    return holdout, tuple(expected)


def bundled_corpus_path(module_file: str) -> Path:
    checkout = Path(module_file).resolve().parents[3] / "evaluation-corpora" / f"{RECIPE_ID}.json"
    if checkout.is_file():
        return checkout
    return Path(module_file).resolve().parents[1] / "evaluation-corpora" / f"{RECIPE_ID}.json"
