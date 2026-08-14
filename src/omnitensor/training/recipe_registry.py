"""Bounded model-recipe loading, catalog discovery, and reference resolution."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path

from ..atomicio import JsonTooLargeError, read_json_bounded
from ..registry import validate_document
from .recipe_model import (
    ModelLicense,
    ModelRecipe,
    ModelRecipeError,
    ModelSource,
    TargetClaim,
)
from .recipe_validation import validate_recipe_semantics

MAX_RECIPE_BYTES = 256 * 1024
MAX_BUNDLED_RECIPES = 128
_RECIPE_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
REFUSED_BUNDLED_RECIPES = {
    "zero-dce": (
        "the official Zero-DCE code and weights are CC-BY-NC-4.0 for academic "
        "research only; OmniTensor accepts only redistributable commercial-use sources"
    )
}


def load_model_recipe(path: Path | str) -> ModelRecipe:
    """Load one bounded recipe and enforce cross-field source/model semantics."""
    try:
        document = read_json_bounded(Path(path), MAX_RECIPE_BYTES)
    except (OSError, ValueError, JsonTooLargeError) as error:
        raise ModelRecipeError("recipe-invalid", f"cannot read model recipe: {error}") from error
    violations = validate_document("model-recipe.schema.json", document)
    if violations:
        raise ModelRecipeError("recipe-invalid", "; ".join(violations))
    assert isinstance(document, dict)  # schema proves this
    validate_recipe_semantics(document)
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    sources = tuple(
        ModelSource(
            item["role"],
            item["uri"],
            item["revision"],
            item["filename"],
            item["sha256"],
            item["sizeBytes"],
        )
        for item in document["sources"]
    )
    license_document = document["license"]
    targets = {
        accelerator: TargetClaim(
            claim["status"],
            claim["compiler"],
            claim["fullyQuantized"],
            claim["reason"],
            claim.get("evidenceSha256"),
        )
        for accelerator, claim in document["targets"].items()
    }
    return ModelRecipe(
        document["id"],
        document["version"],
        document["family"],
        tuple(document["profileIds"]),
        document["sourceFormat"],
        sources,
        ModelLicense(
            license_document["spdx"],
            license_document["termsUri"],
            license_document["attribution"],
        ),
        document["tensorContract"],
        document["outputContract"],
        document.get("preprocessing"),
        document.get("producer"),
        tuple(document["evaluation"]),
        targets,
        document,
        hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


def bundled_model_recipe_root(module_file: Path | str | None = None) -> Path:
    """Return the source-checkout or installed-wheel recipe catalog."""
    package_root = Path(module_file or __file__).resolve().parents[1]
    for candidate in (package_root / "model-recipes", package_root.parents[1] / "model-recipes"):
        if candidate.is_dir():
            return candidate
    raise ModelRecipeError("recipe-catalog-missing", "bundled model recipe catalog is absent")


def load_bundled_model_recipes(
    *,
    root: Path | None = None,
    maximum_recipes: int = MAX_BUNDLED_RECIPES,
    loader: Callable[[Path | str], ModelRecipe] = load_model_recipe,
) -> tuple[ModelRecipe, ...]:
    """Load the complete bounded catalog and reject ambiguous package contents."""
    catalog_root = root if root is not None else bundled_model_recipe_root()
    paths = sorted(catalog_root.glob("*.json"))
    if not paths:
        raise ModelRecipeError("recipe-catalog-invalid", "bundled model recipe catalog is empty")
    if len(paths) > maximum_recipes:
        raise ModelRecipeError(
            "recipe-catalog-invalid", "bundled model recipe catalog is too large"
        )
    recipes: list[ModelRecipe] = []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ModelRecipeError("recipe-catalog-invalid", f"unsafe recipe entry: {path.name}")
        recipe = loader(path)
        if path.name != f"{recipe.id}.json":
            raise ModelRecipeError(
                "recipe-catalog-invalid", f"recipe filename does not match id: {path.name}"
            )
        recipes.append(recipe)
    return tuple(recipes)


def resolve_model_recipe_path(
    reference: str | Path,
    *,
    root: Path | None = None,
    root_resolver: Callable[[], Path] = bundled_model_recipe_root,
    refused: Mapping[str, str] = REFUSED_BUNDLED_RECIPES,
) -> Path:
    """Resolve an explicit path or one traversal-safe bundled recipe id."""
    path = Path(reference).expanduser()
    if path.is_file():
        return path
    text = str(reference)
    if not _RECIPE_ID.fullmatch(text):
        return path
    if text in refused:
        raise ModelRecipeError("recipe-refused", refused[text])
    catalog_root = root if root is not None else root_resolver()
    bundled = catalog_root / f"{text}.json"
    if not bundled.is_file() or bundled.is_symlink():
        raise ModelRecipeError("recipe-not-found", f"no bundled model recipe is named {text}")
    return bundled


__all__ = [
    "REFUSED_BUNDLED_RECIPES",
    "bundled_model_recipe_root",
    "load_bundled_model_recipes",
    "load_model_recipe",
    "resolve_model_recipe_path",
]
