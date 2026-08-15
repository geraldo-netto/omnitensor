from __future__ import annotations

import ast
import asyncio
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import omnitensor.acceptance as facade
import omnitensor.acceptance_checks as checks
import omnitensor.acceptance_cli as cli
import omnitensor.acceptance_contracts as contracts
import omnitensor.acceptance_probes as probes


def test_facade_exports_the_exact_owner_objects():
    contract_names = ("Check", "InstallationReport", "ServiceProbe", "ControlProbe")
    check_names = (
        "REQUIRED_SCHEMAS",
        "DEFAULT_SNAPSHOT_MAX_AGE_MS",
        "FUTURE_TOLERANCE_MS",
        "BACKEND_RUNTIMES",
        "check_executable",
        "check_service",
        "check_schemas",
        "check_workload_catalog",
        "check_plugin_discovery",
        "check_isolation",
        "check_bus",
        "check_snapshot",
        "check_applet",
        "check_confinement",
        "check_applet_contract",
        "check_backends",
        "verify_installation",
    )
    probe_names = (
        "SystemdUserServiceProbe",
        "SocketApplyCommandProbe",
        "_executor_for",
        "_runtime_verdict",
        "_run_command",
    )
    cli_names = (
        "DEFAULT_APPLET_ROOT",
        "load_applet_checksums",
        "build_default_report",
        "main",
    )

    for name in contract_names:
        assert getattr(facade, name) is getattr(contracts, name)
    for name in check_names:
        assert getattr(facade, name) is getattr(checks, name)
    for name in probe_names:
        assert getattr(facade, name) is getattr(probes, name)
    for name in cli_names:
        assert getattr(facade, name) is getattr(cli, name)


def test_moved_values_keep_legacy_repr_and_pickle_globals():
    outcome = facade.Check("service", True, "active")
    report = facade.InstallationReport((outcome,))

    assert type(outcome).__module__ == "omnitensor.acceptance"
    assert type(report).__module__ == "omnitensor.acceptance"
    assert facade.SystemdUserServiceProbe.__module__ == "omnitensor.acceptance"
    assert facade.SocketApplyCommandProbe.__module__ == "omnitensor.acceptance"
    assert pickle.loads(pickle.dumps(outcome)) == outcome
    assert pickle.loads(pickle.dumps(report)) == report

    for name in (
        "check_executable",
        "check_schemas",
        "check_backends",
        "verify_installation",
        "_executor_for",
        "_runtime_verdict",
        "_run_command",
        "load_applet_checksums",
        "build_default_report",
        "main",
    ):
        value = getattr(facade, name)
        assert value.__module__ == "omnitensor.acceptance"
        assert pickle.loads(pickle.dumps(value)) is value


def test_owner_dag_has_no_facade_backedge():
    root = Path(checks.__file__).parent
    for name in (
        "acceptance_contracts.py",
        "acceptance_probes.py",
        "acceptance_checks.py",
        "acceptance_cli.py",
    ):
        source = (root / name).read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module not in {"acceptance", "omnitensor.acceptance"}
                assert not (
                    node.module in {None, "omnitensor"}
                    and any(alias.name == "acceptance" for alias in node.names)
                )
            if isinstance(node, ast.Import):
                assert all(alias.name != "omnitensor.acceptance" for alias in node.names)


def test_a_leaf_import_does_not_load_the_facade():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import omnitensor.acceptance_checks; "
            "assert 'omnitensor.acceptance' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_backend_and_executor_private_facade_seams_are_live(monkeypatch):
    calls = []
    monkeypatch.setattr(
        facade,
        "_runtime_verdict",
        lambda module: calls.append(module) or None,
    )
    assert checks.check_backends((('gpu', 'json', 'unused'),)).ok is True
    assert calls == ["json"]

    available = SimpleNamespace(available=False, code="runtime-unusable", reason="broken")
    executor = SimpleNamespace(availability=lambda: available)
    monkeypatch.setattr(facade, "_executor_for", lambda module: executor)
    assert probes._runtime_verdict("sample-runtime") == "broken"

    def explode():
        raise OSError

    exploding = SimpleNamespace(availability=explode)
    monkeypatch.setattr(facade, "_executor_for", lambda module: exploding)
    assert probes._runtime_verdict("sample-runtime") == "OSError"


def test_workload_root_failure_stays_a_failed_check(monkeypatch):
    def missing():
        raise FileNotFoundError("bundled workload catalog is absent")

    monkeypatch.setattr(facade, "bundled_workloads_path", missing)
    assert checks.check_workload_catalog() == facade.Check(
        "workloads", False, "bundled workload catalog is absent"
    )


def test_systemd_default_runner_resolves_the_facade_at_construction(monkeypatch):
    calls = []

    def run(argv):
        calls.append(argv)
        return 0, "active\n"

    monkeypatch.setattr(facade, "_run_command", run)
    assert probes.SystemdUserServiceProbe().unit_state() == (
        True,
        "omnitensor.service is active",
    )
    assert calls == [["systemctl", "--user", "is-active", "omnitensor.service"]]


def test_socket_probe_exhaustion_preserves_last_error_and_cancellation_is_not_retried():
    failure = ConnectionRefusedError("still starting")
    attempts = []

    async def failing(text):
        attempts.append(text)
        raise failure

    with pytest.raises(ConnectionRefusedError) as raised:
        probes.SocketApplyCommandProbe(
            timeout_s=0.01,
            retry_interval_s=0,
            caller=failing,
        ).apply_command("not-json")
    assert raised.value is failure
    assert len(attempts) > 1

    cancelled_attempts = []

    async def cancelled(text):
        cancelled_attempts.append(text)
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        probes.SocketApplyCommandProbe(caller=cancelled).apply_command("not-json")
    assert cancelled_attempts == ["not-json"]


def test_cli_facade_builder_seam_preserves_arguments_and_exact_output(
    monkeypatch, capsys, tmp_path,
):
    captured = []
    report = facade.InstallationReport((facade.Check("service", True, "active"),))

    def build(**kwargs):
        captured.append(kwargs)
        return report

    monkeypatch.setattr(facade, "build_default_report", build)
    snapshot = tmp_path / "state.json"
    assert cli.main(["--snapshot", str(snapshot)]) == 0
    assert captured[0]["snapshot_path"] == snapshot
    assert captured[0]["applet_root"] is None
    assert captured[0]["applet_checksums"] is None
    assert isinstance(captured[0]["now_ms"], int)
    assert capsys.readouterr().out == "PASS  service: active\n"


def test_cli_preserves_environment_default_help_and_parse_errors(
    monkeypatch, capsys, tmp_path,
):
    captured = []
    report = facade.InstallationReport(())
    monkeypatch.setenv("OMNITENSOR_STATE_PATH", str(tmp_path / "from-env.json"))
    monkeypatch.setattr(
        facade,
        "build_default_report",
        lambda **kwargs: captured.append(kwargs) or report,
    )

    assert cli.main([]) == 0
    assert captured[0]["snapshot_path"] == tmp_path / "from-env.json"
    assert capsys.readouterr().out == "\n"

    with pytest.raises(SystemExit) as helped:
        cli.main(["--help"])
    help_output = capsys.readouterr()
    assert helped.value.code == 0
    assert help_output.err == ""
    assert help_output.out.startswith("usage: ")
    assert "--snapshot" in help_output.out
    assert "--applet-root" in help_output.out
    assert "--applet-checksums" in help_output.out

    with pytest.raises(SystemExit) as invalid:
        cli.main(["--unknown"])
    parse_output = capsys.readouterr()
    assert invalid.value.code == 2
    assert parse_output.out == ""
    assert parse_output.err.startswith("usage: ")
    assert "unrecognized arguments: --unknown" in parse_output.err


def test_console_entrypoint_stays_on_the_stable_facade():
    project = Path(facade.__file__).parents[2] / "pyproject.toml"
    assert 'omnitensor-verify-install = "omnitensor.acceptance:main"' in project.read_text(
        encoding="utf-8"
    )
