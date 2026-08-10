from __future__ import annotations

import inspect

import pytest

from omnitensor.plugins import build_ingestion
from omnitensor.plugins.build_ingestion import (
    MAX_DURATION_MS,
    BuildMetadataIngestor,
    BuildOutcome,
    BuildRecord,
    build_record_error,
)
from omnitensor.plugins.ingestion import IngestionError


def write(root, relative, content=b"source"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def record(build_id="build-1", **changes):
    values = {
        "build_id": build_id,
        "outcome": BuildOutcome.SUCCEEDED,
        "duration_ms": 60_000,
        "started_at_ms": 1_700_000_000_000,
        "changed_paths": ("src/a.py",),
        "failed_checks": (),
    }
    values.update(changes)
    return BuildRecord(**values)


def test_a_repository_is_profiled_from_metadata_alone(tmp_path):
    repo = tmp_path / "repo"
    write(repo, "a.py", b"x" * 10)
    write(repo, "b.py", b"y" * 20)
    write(repo, "README.md", b"z" * 5)

    [profile] = BuildMetadataIngestor([repo]).profile()

    assert profile.file_count == 3
    assert profile.total_bytes == 35
    assert profile.suffix_counts == {".md": 1, ".py": 2}


def test_no_file_content_appears_in_a_profile(tmp_path):
    """A build advisor that reads content is a source-code exfiltrator."""
    repo = tmp_path / "repo"
    write(repo, "secret.py", b"API_KEY = 'private-value'")

    [profile] = BuildMetadataIngestor([repo]).profile()

    assert "private-value" not in repr(profile)
    assert set(vars(profile) if hasattr(profile, "__dict__") else {}) == set()


def test_nothing_here_can_execute_project_code():
    """Asking a build system what it would do means running its config.

    Checked against the imports rather than the raw text, so the module may
    still explain in prose why it does not run anything.
    """
    import ast

    tree = ast.parse(inspect.getsource(build_ingestion))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {"subprocess", "os", "importlib", "runpy", "shutil", "pty"}

    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not called & {"eval", "exec", "compile", "__import__"}


def test_only_configured_repositories_are_profiled(tmp_path):
    configured = tmp_path / "configured"
    other = tmp_path / "other"
    write(configured, "a.py")
    write(other, "b.py")

    profiles = BuildMetadataIngestor([configured]).profile()

    assert len(profiles) == 1
    assert profiles[0].file_count == 1


def test_a_symlink_out_of_the_repository_is_not_followed(tmp_path):
    repo = tmp_path / "repo"
    secret = write(tmp_path / "elsewhere", "secret.py", b"private")
    repo.mkdir()
    (repo / "link.py").symlink_to(secret)

    [profile] = BuildMetadataIngestor([repo]).profile()

    assert profile.file_count == 0


def test_a_valid_history_is_accepted():
    accepted, rejected = BuildMetadataIngestor(["/tmp"]).accept_history(
        [record("build-1"), record("build-2", outcome=BuildOutcome.FAILED)]
    )
    assert [item.build_id for item in accepted] == ["build-1", "build-2"]
    assert rejected == ()


@pytest.mark.parametrize(
    "changes",
    [
        {"outcome": "succeeded"},
        {"build_id": ""},
        {"duration_ms": -1},
        {"duration_ms": MAX_DURATION_MS + 1},
        {"started_at_ms": True},
        {"changed_paths": ["src/a.py"]},
        {"failed_checks": (7,)},
    ],
)
def test_a_malformed_record_is_rejected_with_a_reason(changes):
    accepted, rejected = BuildMetadataIngestor(["/tmp"]).accept_history([record(**changes)])
    assert accepted == ()
    assert len(rejected) == 1


def test_a_non_record_is_rejected():
    assert build_record_error({"buildId": "x"}) != ""


def test_history_is_bounded_and_truncation_is_reported():
    ingestor = BuildMetadataIngestor(["/tmp"], max_records=2)
    accepted, rejected = ingestor.accept_history([record(f"build-{i}") for i in range(5)])
    assert len(accepted) == 2
    assert any("truncated at 2" in reason for reason in rejected)


def test_the_record_bound_is_validated():
    with pytest.raises(IngestionError, match="max_records"):
        BuildMetadataIngestor(["/tmp"], max_records=0)


def test_a_failed_build_carries_its_failed_checks():
    accepted, _rejected = BuildMetadataIngestor(["/tmp"]).accept_history(
        [record(outcome=BuildOutcome.FAILED, failed_checks=("lint", "types"))]
    )
    assert accepted[0].failed_checks == ("lint", "types")


def test_a_missing_repository_profiles_as_empty(tmp_path):
    [profile] = BuildMetadataIngestor([tmp_path / "absent"]).profile()
    assert profile.file_count == 0
    assert profile.total_bytes == 0
