"""One reading of what a module imports, for every layering rule to share.

Six layering gates each hand-wrote this, and each hand-wrote a different
subset of the import forms: `test_extracted_service_owners_have_no_static_
service_backedge` matched only `ast.ImportFrom`, so `import omnitensor.service`
in a leaf passed it; `test_console_targets_stay_on_facade` matched
`node.module == "cli"`, so `from omnitensor.training.cli import main` passed
too. A rule that is green because it looked in the wrong place is worse than
no rule: it is a claim nobody re-checks.

`imported_modules` resolves every form to the dotted names a reader means by
"this module imports that one" — plain `import a.b`, `from a.b import c`,
`from . import c` and `from ..a import c` — and resolves the relative ones
against the importing module's own package, so a rule can be written once
against absolute names and hold whichever way the import is spelled.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Iterator
from pathlib import Path

__all__ = ["forbidden_imports", "imported_modules", "module_name_for"]


def module_name_for(path: Path, source_root: Path) -> str:
    """The dotted name `path` is importable as, relative to `source_root`."""
    relative = path.resolve().relative_to(source_root.resolve())
    parts = list(relative.parts)
    if parts[-1] == "__init__.py":
        parts.pop()
    else:
        parts[-1] = parts[-1].removesuffix(".py")
    return ".".join(parts)


def _absolute(module: str | None, level: int, package: str) -> str:
    if not level:
        return module or ""
    base = package.split(".")
    # `from . import x` inside `a.b.c` is relative to `a.b`; each further dot
    # climbs one more package.
    ascent = base[: len(base) - (level - 1)] if level - 1 else base
    return ".".join([*ascent, module]) if module else ".".join(ascent)


def imported_modules(path: Path, source_root: Path) -> set[str]:
    """Every module `path` imports, as an absolute dotted name.

    Both the module and the names bound from it are yielded for a `from`
    import, because `from omnitensor import service` and
    `from omnitensor.service import x` name the same dependency and a rule
    should not have to know which was written.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = module_name_for(path, source_root).rsplit(".", 1)[0]
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = _absolute(node.module, node.level, package)
            if module:
                found.add(module)
                found.update(f"{module}.{alias.name}" for alias in node.names)
    return found


def forbidden_imports(
    paths: Iterable[Path], forbidden: Iterable[str], *, source_root: Path
) -> list[str]:
    """`path:module` for every import of a forbidden module or its contents.

    A forbidden name matches the module itself and anything under it, so
    naming `omnitensor.service` also refuses `omnitensor.service.main`.
    """
    refused = tuple(forbidden)
    violations = []
    for path in sorted(paths):
        for imported in sorted(imported_modules(path, source_root)):
            if any(imported == name or imported.startswith(f"{name}.") for name in refused):
                violations.append(f"{path}:{imported}")
    return violations


def python_sources(root: Path) -> Iterator[Path]:
    """Every Python source under `root`, skipping caches."""
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" not in path.parts:
            yield path
