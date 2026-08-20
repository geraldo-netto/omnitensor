"""The layering rules' shared reading, proved against every import form.

Written because the rules built on it were green for the wrong reason: the
service back-edge gate matched `ast.ImportFrom` alone, so `import
omnitensor.service` inside an extracted owner passed it. A rule nobody has
watched fail is a rule nobody knows works, so each form gets a case here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from importrules import forbidden_imports, imported_modules, module_name_for, python_sources


@pytest.fixture
def package(tmp_path: Path) -> Path:
    root = tmp_path / "src"
    (root / "pkg" / "inner").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pkg" / "inner" / "__init__.py").write_text("", encoding="utf-8")
    return root


def write(package: Path, relative: str, source: str) -> Path:
    path = package / relative
    path.write_text(source, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import pkg.forbidden\n", "pkg.forbidden"),
        ("import pkg.forbidden as alias\n", "pkg.forbidden"),
        ("from pkg.forbidden import thing\n", "pkg.forbidden"),
        ("from pkg import forbidden\n", "pkg.forbidden"),
        ("from . import forbidden\n", "pkg.forbidden"),
        ("from .forbidden import thing\n", "pkg.forbidden"),
        ("def late():\n    import pkg.forbidden\n", "pkg.forbidden"),
        ("if True:\n    from pkg import forbidden\n", "pkg.forbidden"),
    ],
)
def test_every_spelling_of_one_dependency_is_seen(package, source, expected):
    path = write(package, "pkg/leaf.py", source)
    assert expected in imported_modules(path, package)
    assert f"{path}:{expected}" in forbidden_imports([path], ["pkg.forbidden"], source_root=package)


def test_a_relative_import_resolves_against_the_importing_package(package):
    path = write(
        package, "pkg/inner/leaf.py", "from ..forbidden import thing\nfrom . import near\n"
    )
    assert imported_modules(path, package) >= {
        "pkg.forbidden",
        "pkg.forbidden.thing",
        "pkg.inner",
        "pkg.inner.near",
    }


def test_a_permitted_neighbour_is_not_a_violation(package):
    path = write(package, "pkg/leaf.py", "import pkg.allowed\nfrom pkg import allowed\n")
    assert forbidden_imports([path], ["pkg.forbidden"], source_root=package) == []


def test_a_forbidden_name_covers_what_is_under_it_but_not_what_merely_starts_with_it(package):
    path = write(
        package,
        "pkg/leaf.py",
        "import pkg.service.inner\nimport pkg.service_helper\n",
    )
    assert forbidden_imports([path], ["pkg.service"], source_root=package) == [
        f"{path}:pkg.service.inner"
    ]


def test_module_names_come_from_the_path_including_packages(package):
    assert module_name_for(package / "pkg/inner/__init__.py", package) == "pkg.inner"
    assert module_name_for(package / "pkg/leaf.py", package) == "pkg.leaf"


def test_python_sources_skips_caches(package):
    (package / "pkg/__pycache__").mkdir()
    (package / "pkg/__pycache__/stale.py").write_text("", encoding="utf-8")
    write(package, "pkg/leaf.py", "")
    assert package / "pkg/leaf.py" in set(python_sources(package))
    assert package / "pkg/__pycache__/stale.py" not in set(python_sources(package))
