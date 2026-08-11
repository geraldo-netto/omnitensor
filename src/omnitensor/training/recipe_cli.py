"""Explicit command-line intake for pinned portable model sources."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .recipes import ModelRecipeError, fetch_model_sources

DEFAULT_SOURCE_ROOT = "~/.local/share/omnitensor/model-sources"


def fetch_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="omnitensor-fetch-model-source",
        description="Fetch and verify a pinned portable model recipe outside the service",
    )
    parser.add_argument("recipe", help="path to a model-recipe JSON document")
    parser.add_argument(
        "--accept-license",
        required=True,
        help="exact SPDX identifier shown in the reviewed recipe",
    )
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    arguments = parser.parse_args(argv)
    try:
        fetched = fetch_model_sources(
            Path(arguments.recipe).expanduser(),
            Path(arguments.source_root).expanduser(),
            accepted_license=arguments.accept_license,
        )
    except (ModelRecipeError, OSError, ValueError) as error:
        print(f"model source fetch failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(fetched.document(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(fetch_main())
