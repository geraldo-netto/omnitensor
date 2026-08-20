import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = tuple(sorted((ROOT / ".github" / "workflows").glob("*.yml")))
ACTION_REVISION = re.compile(r"uses:\s+actions/[\w-]+@(?P<revision>[0-9a-f]{40})(?:\s+#\s+v\d+)?$")


def test_root_lock_covers_ci_tooling() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    development = project["project"]["optional-dependencies"]["dev"]
    assert any(requirement.startswith("build>=") for requirement in development)
    assert any(requirement.startswith("hatchling>=") for requirement in development)

    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    assert 'name = "omnitensor"' in lock
    assert 'name = "build"' in lock
    assert 'name = "hatchling"' in lock
    assert "sha256:" in lock


def test_workflows_consume_the_lock_and_pin_external_code() -> None:
    assert {workflow.name for workflow in WORKFLOWS} == {
        "external-plugin.yml",
        "mutation.yml",
        "quality.yml",
    }
    for workflow in WORKFLOWS:
        source = workflow.read_text(encoding="utf-8")
        assert "python -m pip install uv==0.11.28" in source
        assert "uv sync --locked --extra dev" in source
        assert "actions/checkout@v" not in source
        assert "actions/setup-python@v" not in source
        for line in source.splitlines():
            if "uses: actions/" in line:
                message = f"unpinned action in {workflow.name}: {line}"
                assert ACTION_REVISION.search(line.strip()), message


def test_provider_requirements_refuses_a_missing_argument() -> None:
    # CI reads this script through a process substitution, where a traceback
    # becomes an empty requirements file and a silently under-installed
    # provider environment -- the failure the script exists to prevent.
    script = ROOT / "scripts" / "provider-requirements.py"
    completed = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, check=False
    )
    assert completed.returncode != 0
    assert completed.stdout == ""
    assert "usage" in completed.stderr.lower()

    missing = subprocess.run(
        [sys.executable, str(script), str(ROOT / "providers" / "does-not-exist")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert missing.returncode != 0
    assert missing.stdout == ""
    assert "does-not-exist" in missing.stderr


def test_provider_suite_script_owns_the_install_loop() -> None:
    script = (ROOT / "scripts" / "run-provider-suites.sh").read_text(encoding="utf-8")
    assert "--install" in script
    assert "provider-requirements.py" in script
    assert "install --no-deps -e" in script

    workflow = (ROOT / ".github" / "workflows" / "quality.yml").read_text(encoding="utf-8")
    assert "./scripts/run-provider-suites.sh --install --cov" in workflow
    # The loop lives in one place now, so CI cannot drift away from what a
    # developer reproduces locally.
    assert "provider-requirements.py" not in workflow
    assert "uv pip install --no-deps -e" not in workflow


def test_formatting_is_gated_somewhere():
    """`ruff check` and `ruff format` refuse different mistakes.

    Gate audit, 2026-08-20: `ruff format --check` appeared in no workflow, no
    script and no document, so formatting was enforced by nothing and a
    reformatted file reached `develop` unremarked.
    """

    quality = (ROOT / ".github" / "workflows" / "quality.yml").read_text(encoding="utf-8")
    assert "ruff format --check" in quality
    assert "ruff check" in quality


def test_external_plugin_smoke_is_a_script_ci_calls_rather_than_steps_only_ci_has() -> None:
    """The gate that proves an installed plugin needs no checkout must be runnable."""

    script_path = ROOT / "scripts" / "run-external-plugin-smoke.sh"
    assert script_path.stat().st_mode & 0o111, "the script must be executable"
    script = script_path.read_text(encoding="utf-8")
    assert "-m build --wheel" in script
    assert "omnitensor-plugin-smoke" in script
    assert "-m venv" in script

    workflow = (ROOT / ".github" / "workflows" / "external-plugin.yml").read_text(encoding="utf-8")
    assert "./scripts/run-external-plugin-smoke.sh" in workflow
    # One place, so CI cannot drift away from what a developer reproduces.
    assert "python -m build" not in workflow
    assert "omnitensor-plugin-smoke" not in workflow


def test_every_workflow_step_runs_something_a_developer_can_run() -> None:
    """Each CI stage names a command that exists in this checkout.

    The audit's third question: a stage whose name claims more than it runs,
    or whose steps exist nowhere else, is a gate nobody can reproduce.
    """

    local = {
        "./scripts/run-provider-suites.sh": ROOT / "scripts" / "run-provider-suites.sh",
        "./scripts/run-external-plugin-smoke.sh": ROOT / "scripts" / "run-external-plugin-smoke.sh",
        "omnitensor.mutation_campaign": ROOT / "src" / "omnitensor" / "mutation_campaign.py",
    }
    referenced = set()
    for workflow in WORKFLOWS:
        source = workflow.read_text(encoding="utf-8")
        referenced.update(name for name in local if name in source)
    assert referenced == set(local)
    for name, path in local.items():
        assert path.exists(), name


def test_every_provider_distribution_declares_a_coverage_floor_ci_enforces() -> None:
    """`providers/` is a sixth of the runtime source and had no floor at all.

    `[tool.coverage.run] source = ["src/omnitensor"]`, so the root
    `--cov-fail-under=80` says nothing about a provider: a module no test loads
    was indistinguishable from one at 100%. The suites cannot be merged into
    the root run — each distribution has its own dependencies — so each one
    declares its own floor and the script reads it. Checked in both
    directions: a distribution with no floor fails here, and a floor CI never
    runs is caught by the workflow assertions below.
    """
    providers = sorted((ROOT / "providers").glob("*/pyproject.toml"))
    assert len(providers) >= 6

    floors = {}
    for path in providers:
        project = tomllib.loads(path.read_text(encoding="utf-8"))
        floor = project.get("tool", {}).get("coverage", {}).get("report", {}).get("fail_under")
        assert isinstance(floor, int), f"{path.parent.name} declares no coverage floor"
        assert 0 < floor <= 100
        floors[path.parent.name] = floor

    script = (ROOT / "scripts" / "run-provider-suites.sh").read_text(encoding="utf-8")
    assert "--cov)" in script
    assert "fail_under" in script
    assert "--cov-fail-under=" in script

    workflow = (ROOT / ".github" / "workflows" / "quality.yml").read_text(encoding="utf-8")
    assert "./scripts/run-provider-suites.sh --install --cov" in workflow


def _console_scripts() -> dict[str, tuple[str, str]]:
    """Every command the two distributions install, by name."""
    found: dict[str, tuple[str, str]] = {}
    for path in (ROOT / "pyproject.toml", ROOT / "packaging/omnitensor-training/pyproject.toml"):
        project = tomllib.loads(path.read_text(encoding="utf-8"))["project"]
        for name, target in project.get("scripts", {}).items():
            module, _colon, attribute = target.partition(":")
            assert name not in found, f"{name} is installed by two distributions"
            found[name] = (module, attribute)
    return found


def test_every_installed_command_names_a_callable_that_exists() -> None:
    """An entry point that resolves to nothing fails on the machine, at run time.

    Nothing checked this: two console scripts were asserted individually in
    `test_documentation.py` and the other twenty-eight by no test at all, so a
    renamed `main` would have been found by the person who typed the command.
    Import-free on purpose — the training commands live in a distribution this
    environment need not have installed — so the module's source is read
    instead.
    """
    import ast

    source_root = ROOT / "src"
    missing = []
    for name, (module, attribute) in sorted(_console_scripts().items()):
        path = source_root / (module.replace(".", "/") + ".py")
        if not path.is_file():
            path = source_root / module.replace(".", "/") / "__init__.py"
        if not path.is_file():
            missing.append(f"{name}: no module {module}")
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        defined = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                defined.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
            elif isinstance(node, ast.Assign):
                defined.update(target.id for target in node.targets if isinstance(target, ast.Name))
        if attribute not in defined:
            missing.append(f"{name}: {module} defines no {attribute}")

    assert missing == []
    # Both distributions are read, not only the service one.
    assert "omnitensor" in _console_scripts()
    assert "omnitensor-train-model" in _console_scripts()
