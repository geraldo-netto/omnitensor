"""Workload manifest registry.

Loads and validates ``manifest.json`` files against the canonical
``workload-manifest.schema.json``.  The manifest's optional ordered
``acceleratorPreference`` is the routing intent the scheduler consumes;
when omitted the global ``tpu > npu > gpu`` default applies.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import jsonschema

from .discovery import BACKENDS
from .state import ProfilePolicy

_PACKAGE_DIR = Path(__file__).resolve().parent
_PACKAGED_SCHEMAS = _PACKAGE_DIR / "schemas"
# Editable/source checkout fallback.  Installed packages never inspect an
# arbitrary site-packages grandparent for a directory named ``schemas``.
_SOURCE_SCHEMAS = _PACKAGE_DIR.parents[1] / "schemas" if _PACKAGE_DIR.parent.name == "src" else None
MAX_WORKLOADS = 128
MAX_MANIFEST_BYTES = 64 * 1024

DEFAULT_PREFERENCE = tuple(BACKENDS)


def _schema_path(name: str) -> Path:
    packaged = _PACKAGED_SCHEMAS / name
    if packaged.is_file():
        return packaged
    if _SOURCE_SCHEMAS is not None:
        source = _SOURCE_SCHEMAS / name
        if source.is_file():
            return source
    raise FileNotFoundError(f"canonical schema is not installed: {name}")


def load_schema(name: str) -> dict:
    # Contracts are deliberately read for each validation.  Operators can
    # atomically update a schema without restarting the long-running service.
    return json.loads(_schema_path(name).read_bytes())


def _validator(name: str) -> jsonschema.Validator:
    return jsonschema.Draft202012Validator(load_schema(name))


def validate_document(name: str, document: object) -> list[str]:
    """Return human-readable schema violations; empty means valid."""
    return [
        f"{'/'.join(str(part) for part in error.absolute_path) or '/'}: {error.message}"
        for error in _validator(name).iter_errors(document)
    ]


@dataclass(frozen=True)
class Workload:
    id: str
    manifest: dict

    @property
    def accelerator(self) -> str:
        return self.manifest["requirements"]["accelerator"]

    @property
    def preference(self) -> tuple[str, ...]:
        declared = self.manifest["requirements"].get("acceleratorPreference")
        return tuple(declared) if declared else DEFAULT_PREFERENCE

    @property
    def model(self) -> dict | None:
        return self.manifest["requirements"]["model"]

    def default_policy(self) -> ProfilePolicy:
        defaults = self.manifest["defaults"]
        return ProfilePolicy(enabled=defaults["enabled"], weight=defaults["weight"])


class ManifestError(ValueError):
    """A manifest failed validation or the registry bounds."""


def _load_manifest(manifest_path: Path, directory_name: str) -> dict:
    """One strictly validated manifest document, or :class:`ManifestError`."""
    if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ManifestError(f"{manifest_path}: manifest exceeds {MAX_MANIFEST_BYTES} bytes")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise ManifestError(f"{manifest_path}: invalid JSON: {error}") from error
    violations = validate_document("workload-manifest.schema.json", manifest)
    if violations:
        raise ManifestError(f"{manifest_path}: {'; '.join(violations)}")
    if manifest["id"] != directory_name:
        raise ManifestError(f"{manifest_path}: id must match directory name")
    return manifest


def load_workloads(root: Path) -> dict[str, Workload]:
    """Load every ``<id>/manifest.json`` under ``root``, strictly validated."""
    if not root.is_dir():
        return {}
    workloads: dict[str, Workload] = {}
    for directory in sorted(entry for entry in root.iterdir() if entry.is_dir()):
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = _load_manifest(manifest_path, directory.name)
        workloads[manifest["id"]] = Workload(id=manifest["id"], manifest=manifest)
        if len(workloads) > MAX_WORKLOADS:
            raise ManifestError(f"{root}: more than {MAX_WORKLOADS} workloads")
    return workloads
