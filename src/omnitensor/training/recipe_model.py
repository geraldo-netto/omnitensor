"""Stable model-recipe values shared by validation, registries, and fetchers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..stable_error import StableError

MODEL_RECIPE_VERSION = 1
RECEIPT_FILENAME = "source-receipt.json"
_LEGACY_MODULE = "omnitensor.training.recipes"


class ModelRecipeError(StableError, ValueError):
    """Stable producer failure safe to show to a local operator."""


@dataclass(frozen=True, slots=True)
class ModelSource:
    """One immutable file belonging to a portable model release."""

    role: str
    uri: str
    revision: str
    filename: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ModelLicense:
    """Weight-specific terms, deliberately separate from repository code."""

    spdx: str
    terms_uri: str
    attribution: str


@dataclass(frozen=True, slots=True)
class TargetClaim:
    """What is actually known about one native compilation lane."""

    status: str
    compiler: str | None
    fully_quantized: bool
    reason: str
    evidence_sha256: str | None


@dataclass(frozen=True, slots=True)
class ModelRecipe:
    """Validated portable source identity and its target-independent semantics."""

    id: str
    version: str
    family: str
    profile_ids: tuple[str, ...]
    source_format: str
    sources: tuple[ModelSource, ...]
    license: ModelLicense
    tensor_contract: dict
    output_contract: dict
    preprocessing: dict | None
    producer: dict | None
    evaluation: tuple[dict, ...]
    targets: dict[str, TargetClaim]
    document: dict
    document_sha256: str

    @property
    def model_source(self) -> ModelSource:
        return next(source for source in self.sources if source.role == "model")


@dataclass(frozen=True, slots=True)
class FetchedModelSource:
    """Atomic local source installation and its persisted receipt."""

    recipe: ModelRecipe
    root: Path
    receipt_path: Path

    def document(self) -> dict:
        return {
            "recipeId": self.recipe.id,
            "recipeVersion": self.recipe.version,
            "recipeSha256": self.recipe.document_sha256,
            "root": str(self.root),
            "receipt": str(self.receipt_path),
            "files": [source.filename for source in self.recipe.sources],
        }


class SourceTransport(Protocol):
    """Yield bytes from one source URI without owning recipe policy."""

    def chunks(self, uri: str, maximum_bytes: int) -> Iterable[bytes]: ...


for _legacy_type in (
    ModelRecipeError,
    ModelSource,
    ModelLicense,
    TargetClaim,
    ModelRecipe,
    FetchedModelSource,
    SourceTransport,
):
    _legacy_type.__module__ = _LEGACY_MODULE
