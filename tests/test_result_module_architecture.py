from __future__ import annotations

import ast
import pickle
from pathlib import Path

import pytest

from omnitensor import plugins
from omnitensor.plugins import job_results, results, visual_index, visual_results


def _tree(module) -> ast.Module:
    return ast.parse(Path(module.__file__).read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ["JobResultError", "JobRecord", "JobResultStore"])
def test_job_result_old_new_and_barrel_imports_are_identical(name):
    canonical = getattr(job_results, name)
    assert getattr(results, name) is canonical
    assert getattr(plugins, name) is canonical
    assert canonical.__module__ == "omnitensor.plugins.job_results"


@pytest.mark.parametrize(
    "name",
    ["DEFAULT_MAX_RETAINED_JOBS", "DEFAULT_RESULT_TTL_SECONDS", "MAX_RETAINED_JOBS_LIMIT"],
)
def test_job_result_constants_remain_available_from_the_old_module(name):
    assert getattr(results, name) is getattr(job_results, name)


def test_visual_result_old_and_new_imports_are_identical():
    assert visual_results.VisualLibraryResults is visual_index.VisualLibraryResults
    assert visual_results.VISUAL_TAG_PERMISSION is visual_index.VISUAL_TAG_PERMISSION
    assert visual_index.VisualLibraryResults.__module__ == "omnitensor.plugins.visual_index"


@pytest.mark.parametrize(
    ("legacy_global", "canonical"),
    [
        (
            b"comnitensor.plugins.results\nJobResultError\n.",
            job_results.JobResultError,
        ),
        (b"comnitensor.plugins.results\nJobRecord\n.", job_results.JobRecord),
        (b"comnitensor.plugins.results\nJobResultStore\n.", job_results.JobResultStore),
        (
            b"comnitensor.plugins.visual_results\nVisualLibraryResults\n.",
            visual_index.VisualLibraryResults,
        ),
    ],
)
def test_legacy_pickle_globals_resolve_through_compatibility_shims(legacy_global, canonical):
    assert pickle.loads(legacy_global) is canonical


def test_new_job_records_pickle_with_the_canonical_module():
    record = job_results.JobRecord("job-1", None, None, 1.0)
    payload = pickle.dumps(record)
    assert b"omnitensor.plugins.job_results" in payload
    assert pickle.loads(payload) == record


def test_a_legacy_job_record_pickle_still_loads_as_the_canonical_class():
    record = job_results.JobRecord("job-1", None, None, 1.0)
    legacy_payload = pickle.dumps(record, protocol=0).replace(
        b"omnitensor.plugins.job_results\nJobRecord",
        b"omnitensor.plugins.results\nJobRecord",
    )
    restored = pickle.loads(legacy_payload)
    assert restored == record
    assert type(restored) is job_results.JobRecord


def test_compatibility_modules_are_explicit_reexport_only_shims():
    for module, expected_import in (
        (results, "job_results"),
        (visual_results, "visual_index"),
    ):
        tree = _tree(module)
        assert not any(isinstance(node, (ast.ClassDef, ast.FunctionDef)) for node in tree.body)
        imports = {
            node.module
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.level == 1
        }
        assert imports == {expected_import}


def test_visual_result_fold_has_no_reverse_search_cycle():
    imports = {
        node.module
        for node in _tree(visual_index).body
        if isinstance(node, ast.ImportFrom) and node.level == 1
    }
    assert "search" in imports

    search_source = Path(visual_index.__file__).with_name("search.py").read_text(encoding="utf-8")
    search_imports = {
        node.module
        for node in ast.parse(search_source).body
        if isinstance(node, ast.ImportFrom) and node.level == 1
    }
    assert "visual_index" not in search_imports


def test_production_uses_the_canonical_job_result_module():
    source_root = Path(job_results.__file__).parents[1]
    offenders = []
    for path in source_root.rglob("*.py"):
        if path == Path(results.__file__):
            continue
        source = path.read_text(encoding="utf-8")
        if "plugins.results" in source or "from .results" in source:
            offenders.append(path.relative_to(source_root))
    assert offenders == []
