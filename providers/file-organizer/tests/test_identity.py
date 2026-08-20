"""The shim's identity, checked where the wheel that ships it is built.

Five lines of code, and every one of them is an identity that has to agree with
three other places: the entry-point name the service discovers, the module it
imports, the manifest packaged beside it, and the plugin id inside that
manifest. When they disagree the worker is discovered and then refuses to
start, which a person sees only as "worker-unavailable".

`tests/test_generation_installation.py` in the service repository checks the same
agreement from the other side. That is a different boundary: it reads the
manifests in the checkout, not the wheel this distribution builds.
"""

from __future__ import annotations

import ast
import json
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKLOAD_ID = "file-organizer"
PACKAGE = "omnitensor_file_organizer"


def project() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def manifest() -> dict:
    packaged = project()["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    [(source, destination)] = packaged.items()
    assert destination == f"{PACKAGE}/omnitensor-plugin.json"
    return json.loads((ROOT / source).read_text(encoding="utf-8"))


def test_the_entry_point_names_this_workload_and_this_package():
    entry_points = project()["project"]["entry-points"]["omnitensor.workloads"]
    assert entry_points == {WORKLOAD_ID: f"{PACKAGE}:create"}


def test_the_packaged_manifest_declares_the_same_entry_point():
    document = manifest()
    assert document["id"] == WORKLOAD_ID
    # The name the service resolves in the `omnitensor.workloads` group; a
    # manifest naming another one is discovered and then cannot be started.
    assert document["plugin"]["entryPoint"] == WORKLOAD_ID


def _shim_reexport() -> tuple[str, str]:
    """The module and factory name the shim's single import names."""
    tree = ast.parse((ROOT / "src" / PACKAGE / "__init__.py").read_text(encoding="utf-8"))
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert len(imports) == 1, "the shim is one re-export"
    [alias] = imports[0].names
    assert alias.asname == "create"
    exported = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(getattr(target, "id", None) == "__all__" for target in node.targets)
    ]
    assert [ast.literal_eval(value) for value in exported] == [["create"]]
    return imports[0].module, alias.name


def test_the_shim_re_exports_a_factory_the_runtime_actually_defines():
    """A text check passed a shim that named a factory nobody exports.

    Gate audit, 2026-08-20: this asserted `"import create_" in source`, which
    is true of a shim naming `create_typo` — the wheel then installs, the
    service discovers the entry point, and the worker fails at start with
    "worker-unavailable". Read the sibling's own exports instead.
    """

    module, factory = _shim_reexport()
    sibling = ROOT.parent / "vulkan-runtime" / "src" / module.replace(".", "/") / "__init__.py"
    tree = ast.parse(sibling.read_text(encoding="utf-8"))
    exported = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(getattr(target, "id", None) == "__all__" for target in node.targets)
    ]
    assert factory in exported[0], f"{module} does not export {factory}"


def test_the_installed_shim_binds_the_runtime_factory():
    """The proof the structural check cannot give: the objects are the same.

    Skipped, loudly, when the sibling wheel is not installed — CI installs
    every provider through `scripts/run-provider-suites.sh --install`, and so
    should a local run that wants this covered.
    """

    module, factory = _shim_reexport()
    reason = "run scripts/run-provider-suites.sh --install to install both distributions"
    runtime = pytest.importorskip(module, reason=reason)
    shim = pytest.importorskip(PACKAGE, reason=reason)
    assert shim.create is getattr(runtime, factory)
    assert shim.__all__ == ["create"]


def test_the_wheel_ships_this_package_and_nothing_else():
    wheel = project()["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert wheel["packages"] == [f"src/{PACKAGE}"]
