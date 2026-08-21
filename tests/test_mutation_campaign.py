from __future__ import annotations

import copy
import importlib.metadata
import json
import os
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from omnitensor import mutation_campaign, mutation_engine
from omnitensor.mutation_campaign import (
    DEFAULT_MUTMUT_EXECUTABLE,
    execute_mutation_shard,
    mutation_commands,
    mutation_patterns,
)
from omnitensor.mutation_engine import (
    _detach_project_imports,
    disable_string_literal_mutations,
    without_string_literal_mutations,
)
from omnitensor.mutation_manifest import (
    MutationShard,
    _module_path,
    load_mutation_manifest,
    mutable_selector_inventory,
    source_selector_inventory,
)
from omnitensor.mutation_quality import main as mutation_quality_main

ROOT = Path(__file__).parents[1]
MANIFEST = ROOT / "mutation-selectors.json"
_MUTMUT_ACTIVE = "MUTANT_UNDER_TEST" in os.environ


def _source_root(tmp_path: Path) -> Path:
    root = tmp_path / "src"
    package = root / "omnitensor"
    package.mkdir(parents=True)
    (package / "subject.py").write_text(
        "def alpha(value):\n    return value + 1\n\n"
        "def inert():\n    return None\n\n"
        "@decorator\ndef skipped():\n    return 0\n\n"
        "class Worker:\n"
        "    @property\n    def skipped(self):\n        return 0\n\n"
        "    @staticmethod\n    def run(value):\n        return value * 2\n\n"
        "@decorator\nclass Skipped:\n    def method(self):\n        return 0\n",
        encoding="utf-8",
    )
    return root


def _document() -> dict:
    return {
        "version": 2,
        "scope": {
            "status": "partial",
            "blockedBy": "OMNI-0297",
            "onlyMutate": ["src/omnitensor/subject.py"],
            "expansionPriority": [
                "src/omnitensor/training/",
                "src/omnitensor/plugins/document_acceptance.py",
            ],
        },
        "shards": [
            {
                "name": "subject",
                "selectors": [
                    "omnitensor.subject.x_alpha",
                    "omnitensor.subject.xǁWorkerǁrun",
                ],
            }
        ],
    }


def _write_manifest(tmp_path: Path, document: object) -> Path:
    path = tmp_path / "selectors.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


@pytest.mark.skipif(_MUTMUT_ACTIVE, reason="mutmut transforms the inspected source tree")
def test_tracked_manifest_is_exact_complete_and_source_current():
    manifest = load_mutation_manifest(MANIFEST, source_root=ROOT / "src")

    assert manifest.shard_names() == (
        "acceptance",
        "document-binding",
        "event-workload",
        "executors",
        "grounded-answer",
        "mutation-launcher",
        "runtime-contract",
        "scheduler",
        "snapshot-forecast",
        "tensor-output",
    )
    # event-workload grew by version 2's serializers, the pass planner, and the
    # confirmation gate and calendar renderer split out of `events`. The private
    # fragment store moved to `fragments`, which no workload owns, and its
    # selectors moved with it — the shard is a batch, not an owner. The pass
    # merger's dedup key went the other way: it was a copy of the parser's and
    # was removed rather than kept in sync. `grounded-answer` gained the pass
    # planner that replaced the top-eight span cut, which is arithmetic over
    # the context window and worth mutating.
    #
    # The counts moved again when the shape those callables had was collapsed:
    # four `…Errorǁ__init__` selectors were the same three lines, and mutating
    # them four times measured one behaviour four times, so `runtime-contract`
    # tracks `StableError.__init__` once instead. `grounded-answer` follows
    # `document_qa`'s split into the span index and the answer contract, and
    # `scheduler` follows `_BackendQueue`'s arithmetic into `stride`, where the
    # weighting and the held-out clamp actually live now.
    #
    # `acceptance` moved 43 -> 49 when the ten per-module bound-check wrappers
    # became one `BoundChecks` in the kit: four `document_acceptance` wrappers
    # that were pure delegation are gone, and the ten methods that now hold the
    # behaviour are tracked instead — in one place rather than in each workload
    # that qualifies.
    #
    # `snapshot-forecast` moved 13 -> 14 with `selected_files_document`: where
    # this service can read a file somebody selected is a different answer from
    # where a caller may stage one, and publishing the wrong one refuses valid
    # sources (`../xpuwlm` F2).
    #
    # `event-workload` moved 56 -> 57 when the recovery journal stopped
    # answering "nothing was interrupted" for a document it could not read
    # (OMNI-0548): the discard is a statement now, and a statement is
    # behaviour. It moved 57 -> 63 when the event preflight and the one
    # reconsidered refusal left the GPU adapter for the workload they are
    # about (OMNI-0519) — the same six callables, now in a module this
    # manifest covers.
    assert [len(shard.selectors) for shard in manifest.shards] == [
        49,
        3,
        63,
        34,
        18,
        2,
        6,
        31,
        # snapshot-forecast grew _roots_document when OMNI-0606 merged the
        # two root-document bodies into it.
        15,
        12,
    ]
    assert sum(len(shard.selectors) for shard in manifest.shards) == 233
    modules = {
        selector.split(".x", 1)[0] for shard in manifest.shards for selector in shard.selectors
    }
    assert modules == {
        "omnitensor.contract",
        "omnitensor.executors.base",
        "omnitensor.executors.gpu",
        "omnitensor.forecastresult",
        "omnitensor.mutation_engine",
        "omnitensor.outputcontract",
        "omnitensor.plugins.acceptance_kit",
        "omnitensor.plugins.document_acceptance",
        "omnitensor.plugins.document_answer",
        "omnitensor.plugins.document_qa",
        "omnitensor.plugins.document_spans",
        "omnitensor.plugins.event_calendar",
        "omnitensor.plugins.event_confirmation",
        "omnitensor.plugins.event_passes",
        "omnitensor.plugins.event_workload",
        "omnitensor.plugins.fragments",
        "omnitensor.scheduler",
        "omnitensor.snapshot",
        "omnitensor.stable_error",
        "omnitensor.stride",
        "omnitensor.tensorcontract",
        "omnitensor.training.binding",
    }
    assert manifest.scope.status == "partial"
    assert manifest.scope.blocked_by == "OMNI-0297"
    assert manifest.scope.expansion_priority == (
        "src/omnitensor/training/",
        "src/omnitensor/plugins/document_acceptance.py",
    )
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert tuple(project["tool"]["mutmut"]["only_mutate"]) == manifest.scope.only_mutate
    assert manifest.scope.only_mutate == tuple(
        sorted(f"src/{module.replace('.', '/')}.py" for module in modules)
    )
    expected_hotspots = {
        "acceptance": {
            "omnitensor.plugins.acceptance_kit",
            "omnitensor.plugins.document_acceptance",
        },
        "document-binding": {"omnitensor.training.binding"},
        "event-workload": {
            "omnitensor.plugins.event_calendar",
            "omnitensor.plugins.event_confirmation",
            "omnitensor.plugins.event_passes",
            "omnitensor.plugins.event_workload",
            "omnitensor.plugins.fragments",
        },
        "grounded-answer": {
            "omnitensor.plugins.document_answer",
            "omnitensor.plugins.document_qa",
            "omnitensor.plugins.document_spans",
        },
        "mutation-launcher": {"omnitensor.mutation_engine"},
    }
    for name, modules in expected_hotspots.items():
        shard = manifest.shard(name)
        assert {selector.split(".x", 1)[0] for selector in shard.selectors} == modules
        run_command, results_command = mutation_commands(shard, executable="mutmut")
        assert run_command == (
            "mutmut",
            "run",
            *(f"{selector}__mutmut_*" for selector in shard.selectors),
        )
        assert results_command == ("mutmut", "results", "--all", "true")
    with pytest.raises(ValueError, match="no shard named unknown"):
        manifest.shard("unknown")


def test_source_inventory_uses_exact_function_and_method_encoding(tmp_path):
    root = _source_root(tmp_path)
    assert source_selector_inventory(root) == {
        "omnitensor.subject.x_alpha",
        "omnitensor.subject.x_inert",
        "omnitensor.subject.xǁWorkerǁrun",
    }
    assert load_mutation_manifest(_write_manifest(tmp_path, _document()), source_root=root).shard(
        "subject"
    ).selectors == (
        "omnitensor.subject.x_alpha",
        "omnitensor.subject.xǁWorkerǁrun",
    )
    assert mutable_selector_inventory(root, frozenset({"omnitensor.subject"})) == {
        "omnitensor.subject.x_alpha",
        "omnitensor.subject.xǁWorkerǁrun",
    }


def test_mutable_inventory_resolves_modules_and_requires_the_pinned_engine(monkeypatch, tmp_path):
    root = _source_root(tmp_path)
    package = root / "omnitensor/package"
    package.mkdir()
    (package / "__init__.py").write_text("def value():\n    return 1\n", encoding="utf-8")

    assert _module_path(root, "omnitensor.subject") == root / "omnitensor/subject.py"
    assert _module_path(root, "omnitensor.package") == package / "__init__.py"
    with pytest.raises(ValueError, match="has no source file"):
        _module_path(root, "omnitensor.missing")

    monkeypatch.setattr("omnitensor.mutation_manifest.importlib.metadata.version", lambda _: "3.6")
    with pytest.raises(ValueError, match="3.7.0.*found 3.6"):
        mutable_selector_inventory(root, frozenset({"omnitensor.subject"}))

    def missing_engine(_: str) -> str:
        raise importlib.metadata.PackageNotFoundError("mutmut")

    monkeypatch.setattr("omnitensor.mutation_manifest.importlib.metadata.version", missing_engine)
    with pytest.raises(ValueError, match="3.7.0 is required"):
        mutable_selector_inventory(root, frozenset({"omnitensor.subject"}))


def test_project_mutation_engine_disables_only_string_literals():
    from mutmut.mutation import mutators

    original = list(mutators.mutation_operators)
    expected = [entry for entry in original if entry[1] is not mutators.operator_string]
    with without_string_literal_mutations():
        assert mutators.mutation_operators == expected
        disable_string_literal_mutations()
        assert mutators.mutation_operators == expected
    assert mutators.mutation_operators == original


def test_project_mutation_launcher_detaches_original_package_before_cli(monkeypatch):
    modules = {
        "omnitensor": object(),
        "omnitensor.mutation_engine": object(),
        "omnitensorx": object(),
        "mutmut": object(),
    }
    _detach_project_imports(modules)
    assert set(modules) == {"omnitensorx", "mutmut"}

    calls = []
    monkeypatch.setattr(
        mutation_engine, "disable_string_literal_mutations", lambda: calls.append("policy")
    )
    monkeypatch.setattr(mutation_engine, "_detach_project_imports", lambda: calls.append("detach"))
    monkeypatch.setattr("mutmut.__main__.cli", lambda: calls.append("cli"))
    mutation_engine.main()
    assert calls == ["policy", "detach", "cli"]


def test_manifest_requires_every_and_only_mutation_bearing_callable(tmp_path):
    root = _source_root(tmp_path)
    missing = _document()
    missing["shards"][0]["selectors"] = ["omnitensor.subject.x_alpha"]
    with pytest.raises(ValueError, match="omits mutable callable"):
        load_mutation_manifest(_write_manifest(tmp_path, missing), source_root=root)

    inert = _document()
    inert["shards"][0]["selectors"].insert(1, "omnitensor.subject.x_inert")
    with pytest.raises(ValueError, match="has no mutation points"):
        load_mutation_manifest(_write_manifest(tmp_path, inert), source_root=root)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda document: document.update(extra=True), "only version, scope, and shards"),
        (lambda document: document.update(version=True), "version must be 2"),
        (lambda document: document.update(shards=[]), "nonempty array"),
        (
            lambda document: document["shards"][0].update(name="Bad Name"),
            "name is invalid",
        ),
        (
            lambda document: document["shards"][0]["selectors"].reverse(),
            "must be sorted",
        ),
        (
            lambda document: document["shards"][0]["selectors"].append(
                "omnitensor.subject.xǁWorkerǁrun"
            ),
            "must be unique",
        ),
        (
            lambda document: document["shards"][0].update(selectors=["omnitensor.subject.x_*"]),
            "without wildcards",
        ),
        (
            lambda document: document["shards"][0].update(
                selectors=["omnitensor.subject.x_missing"]
            ),
            "stale or unknown",
        ),
    ],
)
def test_manifest_refuses_open_ambiguous_or_stale_scope(tmp_path, change, message):
    root = _source_root(tmp_path)
    document = _document()
    change(document)
    with pytest.raises(ValueError, match=message):
        load_mutation_manifest(_write_manifest(tmp_path, document), source_root=root)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda document: document["scope"].update(extra=True),
            "must contain only status",
        ),
        (
            lambda document: document["scope"].update(status="complete"),
            "status must be partial",
        ),
        (
            lambda document: document["scope"].update(blockedBy="OMNI-9999"),
            "blocked by OMNI-0297",
        ),
        (
            lambda document: document["scope"].update(onlyMutate=["src/omnitensor/*.py"]),
            "exact Python paths",
        ),
        (
            lambda document: document["scope"].update(onlyMutate=[]),
            "nonempty array",
        ),
        (
            lambda document: document["scope"].update(expansionPriority=[]),
            "nonempty array",
        ),
        (
            lambda document: document["scope"].update(onlyMutate=["src/omnitensor/missing.py"]),
            "exactly match selector modules",
        ),
    ],
)
def test_manifest_requires_a_closed_partial_corpus_scope(tmp_path, change, message):
    root = _source_root(tmp_path)
    document = _document()
    change(document)
    with pytest.raises(ValueError, match=message):
        load_mutation_manifest(_write_manifest(tmp_path, document), source_root=root)


def test_manifest_rejects_duplicate_json_keys_shards_and_cross_shard_selectors(tmp_path):
    root = _source_root(tmp_path)
    duplicate_key = tmp_path / "duplicate.json"
    duplicate_key.write_text('{"version":1,"version":1,"shards":[]}', encoding="utf-8")
    with pytest.raises(ValueError, match="repeats key: version"):
        load_mutation_manifest(duplicate_key, source_root=root)

    document = _document()
    duplicate = copy.deepcopy(document["shards"][0])
    duplicate["name"] = "another"
    document["shards"].insert(0, duplicate)
    with pytest.raises(ValueError, match="appears in multiple shards"):
        load_mutation_manifest(_write_manifest(tmp_path, document), source_root=root)

    document = _document()
    document["shards"].append(copy.deepcopy(document["shards"][0]))
    with pytest.raises(ValueError, match="sorted by name|names must be unique"):
        load_mutation_manifest(_write_manifest(tmp_path, document), source_root=root)


def test_manifest_rejects_missing_empty_and_invalid_source_roots(tmp_path):
    with pytest.raises(ValueError, match="not a directory"):
        source_selector_inventory(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="contains no Python files"):
        source_selector_inventory(empty)
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "bad.py").write_text("def nope(:\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot inspect mutation source"):
        source_selector_inventory(broken)


def test_campaign_builds_deterministic_shell_free_exact_commands():
    shard = MutationShard(
        "subject",
        ("omnitensor.subject.x_alpha", "omnitensor.subject.xǁWorkerǁrun"),
    )
    assert mutation_patterns(shard.selectors) == (
        "omnitensor.subject.x_alpha__mutmut_*",
        "omnitensor.subject.xǁWorkerǁrun__mutmut_*",
    )
    assert mutation_commands(shard, executable="/venv/bin/mutmut") == (
        (
            "/venv/bin/mutmut",
            "run",
            "omnitensor.subject.x_alpha__mutmut_*",
            "omnitensor.subject.xǁWorkerǁrun__mutmut_*",
        ),
        ("/venv/bin/mutmut", "results", "--all", "true"),
    )
    assert mutation_commands(shard)[0][0] == DEFAULT_MUTMUT_EXECUTABLE
    assert str(Path(sys.executable).with_name("omnitensor-mutmut")) == DEFAULT_MUTMUT_EXECUTABLE
    assert Path(DEFAULT_MUTMUT_EXECUTABLE).is_file()
    with pytest.raises(ValueError, match="no selectors"):
        mutation_patterns(())


def test_campaign_runs_then_reports_and_gates_the_same_exact_selectors(tmp_path):
    shard = MutationShard("subject", ("omnitensor.subject.x_alpha",))
    calls = []

    def runner(command, **kwargs):
        calls.append((command, kwargs))
        if command[1] == "run":
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(
            returncode=0,
            stdout="omnitensor.subject.x_alpha__mutmut_1: killed\n",
        )

    report = tmp_path / "mutmut.txt"
    report.write_text("stale", encoding="utf-8")
    assert execute_mutation_shard(shard, report, runner=runner) == ()
    assert calls == [
        (
            (DEFAULT_MUTMUT_EXECUTABLE, "run", "omnitensor.subject.x_alpha__mutmut_*"),
            {"check": False},
        ),
        (
            (DEFAULT_MUTMUT_EXECUTABLE, "results", "--all", "true"),
            {"check": False, "capture_output": True, "text": True},
        ),
    ]
    assert report.read_text(encoding="utf-8") == ("omnitensor.subject.x_alpha__mutmut_1: killed\n")


@given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789 ", max_size=512))
@settings(max_examples=20, deadline=None)
def test_campaign_report_replaces_symlink_without_touching_target(tmp_path_factory, suffix):
    shard = MutationShard("subject", ("omnitensor.subject.x_alpha",))
    root = tmp_path_factory.mktemp("mutation-report")
    outside = root / "outside.txt"
    outside.write_text("protected", encoding="utf-8")
    report = root / "report.txt"
    report.symlink_to(outside)
    output = f"omnitensor.subject.x_alpha__mutmut_1: killed\n# {suffix}\n"

    def runner(command, **_kwargs):
        return SimpleNamespace(returncode=0, stdout="" if command[1] == "run" else output)

    execute_mutation_shard(shard, report, runner=runner)

    assert not report.is_symlink()
    assert report.read_text(encoding="utf-8") == output
    assert report.stat().st_mode & 0o777 == 0o600
    assert outside.read_text(encoding="utf-8") == "protected"


@pytest.mark.parametrize(
    ("stage", "message"),
    [("run", "run exited 9"), ("results", "results exited 9")],
)
def test_campaign_refuses_tool_failures_and_removes_stale_reports(tmp_path, stage, message):
    shard = MutationShard("subject", ("omnitensor.subject.x_alpha",))

    def runner(command, **_kwargs):
        failed = command[1] == stage
        return SimpleNamespace(returncode=9 if failed else 0, stdout="")

    report = tmp_path / "mutmut.txt"
    report.write_text("stale", encoding="utf-8")
    with pytest.raises(RuntimeError, match=message):
        execute_mutation_shard(shard, report, runner=runner)
    assert not report.exists()


def test_campaign_cli_control_flow_without_expanding_selector_inventory(
    monkeypatch, tmp_path, capsys
):
    shard = MutationShard("subject", ("omnitensor.subject.x_alpha",))

    def select(name):
        if name != shard.name:
            raise ValueError(f"mutation manifest has no shard named {name}")
        return shard

    manifest = SimpleNamespace(shard_names=lambda: (shard.name,), shard=select)
    monkeypatch.setattr(
        mutation_campaign,
        "load_mutation_manifest",
        lambda *_args, **_kwargs: manifest,
    )

    assert mutation_campaign.main(["manifest.json", "--list-shards"]) == 0
    assert capsys.readouterr().out == '["subject"]\n'
    assert (
        mutation_campaign.main(
            ["manifest.json", "--list-shards", "--report", str(tmp_path / "report")]
        )
        == 2
    )
    assert "--report is invalid" in capsys.readouterr().out
    assert mutation_campaign.main(["manifest.json", "--shard", shard.name]) == 2
    assert "--report is required" in capsys.readouterr().out

    arguments = [
        "manifest.json",
        "--shard",
        shard.name,
        "--report",
        str(tmp_path / "report"),
        "--threshold",
        "90",
    ]
    monkeypatch.setattr(
        mutation_campaign,
        "execute_mutation_shard",
        lambda *_args, **_kwargs: (),
    )
    assert mutation_campaign.main(arguments) == 0
    assert "1 callables at or above 90%" in capsys.readouterr().out
    monkeypatch.setattr(
        mutation_campaign,
        "execute_mutation_shard",
        lambda *_args, **_kwargs: ("selector 50.0% (1/2 detected)",),
    )
    assert mutation_campaign.main(arguments) == 1
    assert "Per-callable mutation score below 90%" in capsys.readouterr().out

    def deny_manifest(*_args, **_kwargs):
        raise OSError("denied")

    monkeypatch.setattr(mutation_campaign, "load_mutation_manifest", deny_manifest)
    assert mutation_campaign.main(["manifest.json", "--list-shards"]) == 2
    assert "mutation campaign failed: denied" in capsys.readouterr().out


@pytest.mark.skipif(_MUTMUT_ACTIVE, reason="mutmut transforms the inspected source tree")
def test_campaign_cli_lists_manifest_shards_without_running_mutmut(capsys):
    assert mutation_campaign.main([str(MANIFEST), "--list-shards"]) == 0
    assert json.loads(capsys.readouterr().out) == [
        "acceptance",
        "document-binding",
        "event-workload",
        "executors",
        "grounded-answer",
        "mutation-launcher",
        "runtime-contract",
        "scheduler",
        "snapshot-forecast",
        "tensor-output",
    ]
    assert (
        mutation_campaign.main([str(MANIFEST), "--shard", "unknown", "--report", "/tmp/report"])
        == 2
    )
    assert "no shard named unknown" in capsys.readouterr().out


@pytest.mark.skipif(_MUTMUT_ACTIVE, reason="mutmut transforms the inspected source tree")
def test_campaign_cli_gates_selected_shard_and_validates_mode_options(
    monkeypatch, tmp_path, capsys
):
    observed = []

    def execute(shard, report, **options):
        observed.append((shard.name, report, options))
        return ()

    monkeypatch.setattr(mutation_campaign, "execute_mutation_shard", execute)
    arguments = [
        str(MANIFEST),
        "--shard",
        "runtime-contract",
        "--report",
        str(tmp_path / "report.txt"),
        "--threshold",
        "90",
        "--mutmut-executable",
        "/venv/bin/mutmut",
    ]
    assert mutation_campaign.main(arguments) == 0
    assert observed == [
        (
            "runtime-contract",
            tmp_path / "report.txt",
            {"threshold": 90.0, "executable": "/venv/bin/mutmut"},
        )
    ]
    selectors = (
        load_mutation_manifest(MANIFEST, source_root=ROOT / "src")
        .shard("runtime-contract")
        .selectors
    )
    assert capsys.readouterr().out == (
        f"mutation shard runtime-contract: {len(selectors)} callables at or above 90%\n"
    )

    monkeypatch.setattr(
        mutation_campaign,
        "execute_mutation_shard",
        lambda *_args, **_kwargs: ("selector 50.0% (1/2 detected)",),
    )
    assert mutation_campaign.main(arguments) == 1
    assert capsys.readouterr().out == (
        "Per-callable mutation score below 90%:\nselector 50.0% (1/2 detected)\n"
    )
    assert mutation_campaign.main([str(MANIFEST), "--shard", "runtime-contract"]) == 2
    assert "--report is required" in capsys.readouterr().out
    assert (
        mutation_campaign.main(
            [str(MANIFEST), "--list-shards", "--report", str(tmp_path / "report")]
        )
        == 2
    )
    assert "--report is invalid" in capsys.readouterr().out


@pytest.mark.skipif(_MUTMUT_ACTIVE, reason="mutmut transforms the inspected source tree")
def test_mutation_quality_accepts_the_same_manifest_shard(tmp_path, capsys):
    manifest = load_mutation_manifest(MANIFEST, source_root=ROOT / "src")
    shard = manifest.shard("runtime-contract")
    report = tmp_path / "mutmut.txt"
    report.write_text(
        "\n".join(f"{selector}__mutmut_1: killed" for selector in shard.selectors),
        encoding="utf-8",
    )
    assert (
        mutation_quality_main(
            [
                str(report),
                "--selector-file",
                str(MANIFEST),
                "--shard",
                shard.name,
                "--source-root",
                str(ROOT / "src"),
            ]
        )
        == 0
    )
    # Derived, not pinned: a shard gaining a callable is not a defect in the
    # quality gate, and pinning the count made this test fail for refactors.
    assert capsys.readouterr().out == (
        f"per-callable mutation score: {len(shard.selectors)} callables at or above 80%\n"
    )
    assert (
        mutation_quality_main(
            [str(report), "--selector-file", str(MANIFEST), "--source-root", str(ROOT / "src")]
        )
        == 2
    )
    assert "--shard is required" in capsys.readouterr().out


def test_ci_matrix_is_derived_from_manifest_and_runs_the_same_campaign():
    workflow = (ROOT / ".github/workflows/mutation.yml").read_text(encoding="utf-8")
    assert "mutation_campaign" in workflow
    assert "mutation-selectors.json --list-shards" in workflow
    assert "fromJSON(needs.selector-plan.outputs.shards)" in workflow
    assert "mutation_campaign mutation-selectors.json" in workflow
    assert '--shard "${{ matrix.shard }}"' in workflow
    assert "if-no-files-found: error" in workflow

    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"mutation-selectors.json",' in project
    assert '"mutmut==3.7.0",' in project
    assert 'omnitensor-mutmut = "omnitensor.mutation_engine:main"' in project
    project_document = tomllib.loads(project)
    assert "scripts/" in project_document["tool"]["mutmut"]["also_copy"]
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "--selector-file mutation-selectors.json --shard scheduler" in readme
    for shard in ("acceptance", "document-binding", "event-workload", "grounded-answer"):
        assert f"--shard {shard} --report /tmp/omnitensor-mutmut-{shard}.txt" in readme


def test_a_selector_module_is_split_on_the_separator_not_on_a_bare_x():
    """OMNI-0440: `omnitensor.executors.xpu` reported module `omnitensor.executors`."""
    from omnitensor.mutation_manifest import selector_module

    assert selector_module("omnitensor.discovery.x_detect_gpu") == "omnitensor.discovery"
    assert selector_module("omnitensor.executors.gpu.xǁGpuExecutorǁavailability") == (
        "omnitensor.executors.gpu"
    )
    # The segment that used to break it: a module whose name starts with `x`.
    assert selector_module("omnitensor.executors.xpu.x_run") == "omnitensor.executors.xpu"
    assert selector_module("omnitensor.x_ray.xǁScannerǁscan") == "omnitensor.x_ray"
    assert selector_module("omnitensor.discovery") == "omnitensor.discovery"
