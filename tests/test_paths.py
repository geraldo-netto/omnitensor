"""One place says where this service keeps things."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from omnitensor import paths

ROOT = Path(__file__).parents[1]
LOCATIONS = tuple(getattr(paths, name) for name in paths.__all__)


def sources() -> list[Path]:
    return [
        path
        for path in (ROOT / "src").rglob("*.py")
        if "__pycache__" not in path.parts and path.name != "paths.py"
    ]


def test_no_module_writes_a_canonical_location_out_for_itself():
    """OMNI-0526: the artifact root was written out in four modules."""
    literals = {location for location in LOCATIONS if location.startswith("~")}
    offenders = []
    for path in sources():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and node.value in literals:
                offenders.append(f"{path}:{node.lineno} {node.value}")
    assert offenders == []


def test_the_state_file_keeps_the_name_the_applet_reads():
    # A contract with another repository, not a tidy-up: moving it makes the
    # panel report a runtime that is not running.
    assert paths.SNAPSHOT_PATH == "~/.local/state/xpu-workload-manager/state.json"


def test_every_location_is_under_a_single_state_data_or_cache_directory():
    """Three roots and no fourth: state, data, and the disposable cache.

    The lease is the only cache entry, and it is one because a lease is a lock
    rather than data — a host that clears its cache should lose it, and losing
    it costs nothing.
    """
    for location in LOCATIONS:
        assert location.startswith(("~/.local/", "~/.cache")), location
    assert not re.search(r"//", "".join(LOCATIONS))
    assert paths.BENCHMARK_LEASE_ROOT == "~/.cache/omnitensor-bench"


def test_the_modules_that_had_copies_now_agree():
    from omnitensor import composition, document_model_types, probe_cli
    from omnitensor.training import cli, training_record_cli

    assert composition.DEFAULT_ARTIFACT_ROOT == paths.ARTIFACT_ROOT
    assert document_model_types.DEFAULT_ARTIFACT_ROOT == paths.ARTIFACT_ROOT
    assert cli.DEFAULT_ARTIFACT_ROOT == paths.ARTIFACT_ROOT
    assert Path(paths.ARTIFACT_ROOT).expanduser() == probe_cli.DEFAULT_ARTIFACT_ROOT
    assert composition.DEFAULT_STATE_PATH == paths.SNAPSHOT_PATH
    assert cli.DEFAULT_SNAPSHOT_PATH == paths.SNAPSHOT_PATH
    assert training_record_cli.DEFAULT_SNAPSHOT_PATH == paths.SNAPSHOT_PATH


def test_only_the_xdg_fallback_resolves_the_user_home_for_itself():
    """A default nobody passed is how a test writes to the real `$HOME`.

    `benchmark/harness.py` resolved `Path.home() / ".cache/omnitensor-bench"`
    when a caller named no lease root, so three tests in
    `test_benchmark_runtime_release.py` took a lease in the real user's cache
    on every run. The default belongs at the CLI boundary, where a person
    chose it; everything below is told. `publisher_identity` keeps its two
    because they are the documented XDG fallback and each is reached only when
    the corresponding environment variable is unset.
    """
    offenders = []
    for path in sources():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "home"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "Path"
            ):
                offenders.append(str(path.relative_to(ROOT)))

    assert sorted(set(offenders)) == ["src/omnitensor/publisher_identity.py"]
