"""The service wheel serves; the training wheel produces. They stay apart.

The split only holds while the dependency runs one way. A single `from
.training import ...` inside the serving runtime would make the service wheel
un-importable on a desktop that never installed the producer half — and would
do it at run time, on somebody's machine, rather than here.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVICE = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
TRAINING = tomllib.loads(
    (ROOT / "packaging" / "omnitensor-training" / "pyproject.toml").read_text(encoding="utf-8")
)
SOURCE = ROOT / "src" / "omnitensor"


def service_modules():
    for path in SOURCE.rglob("*.py"):
        if "training" not in path.relative_to(SOURCE).parts:
            yield path


def imported_modules(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            prefix = "." * node.level
            yield f"{prefix}{node.module or ''}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name


def test_the_serving_runtime_never_imports_the_producer_half():
    offenders = {
        str(path.relative_to(ROOT)): sorted(
            name
            for name in imported_modules(path)
            if name.startswith(("omnitensor.training", ".training", "..training"))
        )
        for path in service_modules()
    }
    assert {path: names for path, names in offenders.items() if names} == {}


def test_the_service_wheel_does_not_carry_the_training_package():
    wheel = SERVICE["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert wheel["packages"] == ["src/omnitensor"]
    assert "src/omnitensor/training" in wheel["exclude"]
    included = SERVICE["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert "model-recipes" not in included


def test_the_training_wheel_carries_the_training_package_and_its_recipes():
    included = TRAINING["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert included == {
        "../../src/omnitensor/training": "omnitensor/training",
        "../../model-recipes": "omnitensor/model-recipes",
    }


def test_no_console_script_is_declared_by_both_distributions():
    service = set(SERVICE["project"]["scripts"])
    training = set(TRAINING["project"]["scripts"])

    assert service & training == set()
    # Every trainer entry point names the package that ships it.
    assert all(
        target.startswith("omnitensor.training")
        for target in TRAINING["project"]["scripts"].values()
    )
    assert not any(
        target.startswith("omnitensor.training")
        for target in SERVICE["project"]["scripts"].values()
    )


def test_the_two_halves_are_pinned_to_one_version():
    # Recipes ship in the training wheel and the schemas validating them ship
    # in the service wheel. Installed at different versions, a recipe would be
    # checked against a contract it was never written for.
    version = SERVICE["project"]["version"]
    assert TRAINING["project"]["version"] == version
    assert TRAINING["project"]["dependencies"] == [f"omnitensor=={version}"]


def test_producer_only_dependencies_left_with_their_importers():
    service_extras = set(SERVICE["project"]["optional-dependencies"])
    training_extras = set(TRAINING["project"]["optional-dependencies"])

    assert training_extras == {
        "train",
        "model-producers",
        "document-producers",
        "foundation-producers",
        "retinexformer-producers",
    }
    assert service_extras & training_extras == set()
