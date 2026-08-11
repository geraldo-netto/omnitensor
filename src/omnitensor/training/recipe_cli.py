"""Explicit command-line intake for pinned portable model sources."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .recipes import (
    ModelRecipeError,
    fetch_model_sources,
    load_bundled_model_recipes,
    resolve_model_recipe_path,
)

DEFAULT_SOURCE_ROOT = "~/.local/share/omnitensor/model-sources"


def fetch_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="omnitensor-fetch-model-source",
        description="Fetch and verify a pinned portable model recipe outside the service",
    )
    parser.add_argument("recipe", help="bundled recipe id or path to a model-recipe JSON document")
    parser.add_argument(
        "--accept-license",
        required=True,
        help="exact SPDX identifier shown in the reviewed recipe",
    )
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    arguments = parser.parse_args(argv)
    try:
        fetched = fetch_model_sources(
            resolve_model_recipe_path(arguments.recipe),
            Path(arguments.source_root).expanduser(),
            accepted_license=arguments.accept_license,
        )
    except (ModelRecipeError, OSError, ValueError) as error:
        print(f"model source fetch failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(fetched.document(), indent=2))
    return 0


def list_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="omnitensor-list-model-recipes",
        description="List reviewed portable source recipes and honest target status",
    )
    parser.parse_args(argv)
    try:
        recipes = load_bundled_model_recipes()
    except (ModelRecipeError, OSError, ValueError) as error:
        print(f"model recipe catalog failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            [
                {
                    "id": recipe.id,
                    "version": recipe.version,
                    "family": recipe.family,
                    "profileIds": list(recipe.profile_ids),
                    "sourceFormat": recipe.source_format,
                    "license": recipe.license.spdx,
                    "targets": {
                        target: claim.status for target, claim in recipe.targets.items()
                    },
                }
                for recipe in recipes
            ],
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(fetch_main())
