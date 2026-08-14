"""Strict, source-validated mutation selector manifests."""

from __future__ import annotations

import ast
import importlib.metadata
import json
import re
from dataclasses import dataclass
from pathlib import Path

MANIFEST_VERSION = 1
MUTMUT_VERSION = "3.7.0"
_SHARD_NAME = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
_WILDCARDS = frozenset("*?[")


@dataclass(frozen=True, slots=True)
class MutationShard:
    name: str
    selectors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MutationManifest:
    version: int
    shards: tuple[MutationShard, ...]

    def shard(self, name: str) -> MutationShard:
        for shard in self.shards:
            if shard.name == name:
                return shard
        raise ValueError(f"mutation manifest has no shard named {name}")

    def shard_names(self) -> tuple[str, ...]:
        return tuple(shard.name for shard in self.shards)


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"mutation manifest repeats key: {key}")
        document[key] = value
    return document


def _source_files(root: Path) -> tuple[Path, ...]:
    if not root.is_dir():
        raise ValueError(f"mutation source root is not a directory: {root}")
    files = tuple(sorted(root.rglob("*.py")))
    if not files:
        raise ValueError(f"mutation source root contains no Python files: {root}")
    return files


def _instrumentable_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    if not node.decorator_list:
        return True
    return len(node.decorator_list) == 1 and isinstance(node.decorator_list[0], ast.Name) and (
        node.decorator_list[0].id in {"classmethod", "staticmethod"}
    )


def _module_selectors(path: Path, root: Path) -> set[str]:
    parts = list(path.relative_to(root).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    if not parts:
        return set()
    module = ".".join(parts)
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as error:
        raise ValueError(f"cannot inspect mutation source {path}: {error}") from error
    selectors = {
        f"{module}.x_{node.name}"
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and _instrumentable_function(node)
    }
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and not node.decorator_list:
            selectors.update(
                f"{module}.xǁ{node.name}ǁ{child.name}"
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and _instrumentable_function(child)
            )
    return selectors


def source_selector_inventory(source_root: Path | str) -> frozenset[str]:
    """Return exact mutmut callable bases present below ``source_root``."""
    root = Path(source_root)
    selectors: set[str] = set()
    for path in _source_files(root):
        selectors.update(_module_selectors(path, root))
    return frozenset(selectors)


def _module_path(root: Path, module: str) -> Path:
    relative = Path(*module.split("."))
    source = root / relative.with_suffix(".py")
    if source.is_file():
        return source
    package = root / relative / "__init__.py"
    if package.is_file():
        return package
    raise ValueError(f"mutation module has no source file: {module}")


def mutable_selector_inventory(
    source_root: Path | str,
    modules: frozenset[str],
) -> frozenset[str]:
    """Return mutation-bearing callables using the repository's pinned mutmut."""
    try:
        version = importlib.metadata.version("mutmut")
        from mutmut.mutation.file_mutation import mutate_file_contents  # noqa: PLC0415

        from .mutation_engine import without_string_literal_mutations  # noqa: PLC0415
    except (ImportError, importlib.metadata.PackageNotFoundError) as error:
        raise ValueError(f"mutmut {MUTMUT_VERSION} is required to validate selectors") from error
    if version != MUTMUT_VERSION:
        raise ValueError(
            f"mutmut {MUTMUT_VERSION} is required to validate selectors; found {version}"
        )
    root = Path(source_root)
    selectors: set[str] = set()
    for module in sorted(modules):
        path = _module_path(root, module)
        try:
            with without_string_literal_mutations():
                mutated = mutate_file_contents(str(path), path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError) as error:
            raise ValueError(f"cannot generate mutation selectors for {module}: {error}") from error
        selectors.update(f"{module}.{name}" for name in mutated.hash_by_function_name)
    return frozenset(selectors)


def _manifest_document(path: Path) -> dict:
    try:
        document = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"mutation manifest is not JSON: {error}") from error
    if not isinstance(document, dict) or set(document) != {"version", "shards"}:
        raise ValueError("mutation manifest must contain only version and shards")
    return document


def _selector_tuple(name: str, value: object, inventory: frozenset[str]) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"mutation shard {name} selectors must be a nonempty array")
    if any(not isinstance(selector, str) for selector in value):
        raise ValueError(f"mutation shard {name} selectors must be strings")
    selectors = tuple(value)
    if selectors != tuple(sorted(selectors)):
        raise ValueError(f"mutation shard {name} selectors must be sorted")
    if len(set(selectors)) != len(selectors):
        raise ValueError(f"mutation shard {name} selectors must be unique")
    if any(_WILDCARDS.intersection(selector) for selector in selectors):
        raise ValueError("mutation selectors must name exact callables without wildcards")
    stale = next((selector for selector in selectors if selector not in inventory), None)
    if stale is not None:
        raise ValueError(f"mutation selector is stale or unknown: {stale}")
    return selectors


def _mutation_shard(raw: object, inventory: frozenset[str]) -> MutationShard:
    if not isinstance(raw, dict) or set(raw) != {"name", "selectors"}:
        raise ValueError("mutation shard must contain only name and selectors")
    name = raw["name"]
    if not isinstance(name, str) or _SHARD_NAME.fullmatch(name) is None:
        raise ValueError("mutation shard name is invalid")
    return MutationShard(name, _selector_tuple(name, raw["selectors"], inventory))


def load_mutation_manifest(
    path: Path | str,
    *,
    source_root: Path | str = Path("src"),
) -> MutationManifest:
    """Load one closed, ordered manifest and refuse selectors stale in source."""
    manifest_path = Path(path)
    document = _manifest_document(manifest_path)
    version = document["version"]
    if type(version) is not int or version != MANIFEST_VERSION:
        raise ValueError(f"mutation manifest version must be {MANIFEST_VERSION}")
    raw_shards = document["shards"]
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ValueError("mutation manifest shards must be a nonempty array")

    inventory = source_selector_inventory(source_root)
    shards = tuple(_mutation_shard(raw, inventory) for raw in raw_shards)
    names = tuple(shard.name for shard in shards)
    if names != tuple(sorted(names)):
        raise ValueError("mutation shards must be sorted by name")
    if len(set(names)) != len(names):
        raise ValueError("mutation shard names must be unique")
    selectors = tuple(selector for shard in shards for selector in shard.selectors)
    if len(set(selectors)) != len(selectors):
        raise ValueError("mutation selector appears in multiple shards")
    modules = frozenset(selector.split(".x", 1)[0] for selector in selectors)
    mutable = mutable_selector_inventory(source_root, modules)
    unexecutable = sorted(set(selectors) - mutable)
    if unexecutable:
        raise ValueError(f"mutation selector has no mutation points: {unexecutable[0]}")
    missing = sorted(mutable - set(selectors))
    if missing:
        raise ValueError(f"mutation manifest omits mutable callable: {missing[0]}")
    return MutationManifest(version, shards)
