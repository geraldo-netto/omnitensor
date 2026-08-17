"""The serving runtime must not reach into the producer half.

This is the one thing the split can break silently: `omnitensor.training` is no
longer in the service wheel, so a single import of it from a serving module
would raise `ModuleNotFoundError` on a desktop that installed inference only —
at run time, on somebody's machine. Everything else about the split is visible
in the built wheels.
"""

from __future__ import annotations

import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "src" / "omnitensor"


def imported_names(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            yield f"{'.' * node.level}{node.module or ''}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name


def test_the_serving_runtime_never_imports_the_producer_half():
    reaching = {
        str(path.relative_to(SOURCE)): sorted(
            name
            for name in imported_names(path)
            if name.startswith(("omnitensor.training", ".training", "..training"))
        )
        for path in SOURCE.rglob("*.py")
        if "training" not in path.relative_to(SOURCE).parts
    }

    assert {path: names for path, names in reaching.items() if names} == {}
