"""The shim's identity, checked where the wheel that ships it is built.

Five lines of code, and every one of them is an identity that has to agree with
three other places: the entry-point name the service discovers, the module it
imports, the manifest packaged beside it, and the plugin id inside that
manifest. When they disagree the worker is discovered and then refuses to
start, which a person sees only as "worker-unavailable".

`tests/test_qwen_installation.py` in the service repository checks the same
agreement from the other side. That is a different boundary: it reads the
manifests in the checkout, not the wheel this distribution builds.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKLOAD_ID = "event-extraction"
PACKAGE = "omnitensor_event_extraction"


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


def test_the_module_exports_exactly_the_entry_point_the_wheel_declares():
    source = (ROOT / "src" / PACKAGE / "__init__.py").read_text(encoding="utf-8")
    assert "import create_" in source, "the shim re-exports the runtime's factory"
    assert '__all__ = ["create"]' in source


def test_the_wheel_ships_this_package_and_nothing_else():
    wheel = project()["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert wheel["packages"] == [f"src/{PACKAGE}"]
