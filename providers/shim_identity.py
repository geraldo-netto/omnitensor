"""The one identity check a provider wheel can make and the checkout cannot.

Five distributions each carried a byte-for-byte copy of the same file,
differing only in a workload id and a package name, so the shim contract was
stated five times and had to be edited in five places. Four of the five checks
in those copies are structural — the entry point, the packaged manifest, the
manifest's plugin id, the wheel's package list — and the service repository's
`tests/test_generation_installation.py` now derives every one of them from the
tree and checks both directions of the correspondence at once.

What it cannot check is what only an installed wheel knows: that the object
the shim exports *is* the runtime's factory. A structural check reads two
files and concludes they agree; this imports both and compares the objects, so
a shim that names a factory the runtime stopped exporting fails here instead
of at a worker start a person sees only as "worker-unavailable".
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

INSTALL_HINT = "run scripts/run-provider-suites.sh --install to install both distributions"


def shim_reexport(root: Path, package: str) -> tuple[str, str]:
    """The module and factory name the shim's single import names.

    Read rather than imported, because the point of the caller is to compare
    what the source says with what the installed wheel does; taking both from
    the same import would compare a thing with itself.
    """
    tree = ast.parse((root / "src" / package / "__init__.py").read_text(encoding="utf-8"))
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


def assert_installed_shim_binds_the_runtime_factory(root: Path, package: str) -> None:
    """Skipped, loudly, when the sibling wheel is not installed.

    CI installs every provider through `scripts/run-provider-suites.sh
    --install`, and so should a local run that wants this covered.
    """
    module, factory = shim_reexport(root, package)
    runtime = pytest.importorskip(module, reason=INSTALL_HINT)
    shim = pytest.importorskip(package, reason=INSTALL_HINT)
    assert shim.create is getattr(runtime, factory)
    assert shim.__all__ == ["create"]
