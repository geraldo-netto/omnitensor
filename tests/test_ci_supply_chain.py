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
    assert "./scripts/run-provider-suites.sh --install" in workflow
    # The loop lives in one place now, so CI cannot drift away from what a
    # developer reproduces locally.
    assert "provider-requirements.py" not in workflow
    assert "uv pip install --no-deps -e" not in workflow
