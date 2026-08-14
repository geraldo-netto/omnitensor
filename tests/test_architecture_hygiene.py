from __future__ import annotations

import ast
from contextlib import contextmanager
from pathlib import Path

from omnitensor import snapshot, telemetry_types
from omnitensor.plugins import (
    collection,
    hardware_collection,
    network_collection,
    peripheral_collection,
    resource_collection,
    storage_collection,
    telemetry,
)
from omnitensor.training import cli, compilers, desktop, desktop_history, hardware, installation


def test_neutral_boundaries_use_canonical_absolute_imports():
    root = Path(__file__).parents[1] / "src" / "omnitensor"
    neutral = {"forecasting", "preparation", "telemetry_recorder", "telemetry_types"}
    violations = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.level
                and (
                    node.module in neutral
                    or (node.module is None and any(alias.name in neutral for alias in node.names))
                )
            ):
                violations.append(f"{path.relative_to(root)}:{node.lineno}")
    assert violations == []


def test_stable_identity_patterns_have_one_neutral_owner_per_contract():
    assert collection.STABLE_ID is telemetry_types.STABLE_ID
    assert hardware_collection._telemetry_types.STABLE_ID is telemetry_types.STABLE_ID
    assert resource_collection._STABLE_ID is telemetry_types.STABLE_ID
    assert storage_collection._STABLE_ID is telemetry_types.STABLE_ID
    assert hardware.STABLE_ID is telemetry_types.STABLE_ID
    assert network_collection._STABLE_ID is telemetry_types.NETWORK_STABLE_ID
    assert peripheral_collection._STABLE_ID is telemetry_types.PERIPHERAL_STABLE_ID
    assert telemetry_types.PERIPHERAL_STABLE_ID is telemetry_types.NETWORK_STABLE_ID


def test_snapshot_emits_the_plugin_telemetry_contract_owner_version(monkeypatch):
    assert snapshot.PLUGIN_TELEMETRY_VERSION == telemetry.PLUGIN_TELEMETRY_CONTRACT_VERSION
    monkeypatch.setattr(snapshot, "validate_document", lambda *_args: [])
    document = snapshot.build_snapshot([], {}, {}, plugin_telemetry=[])
    assert document["pluginTelemetry"] == {
        "version": telemetry.PLUGIN_TELEMETRY_CONTRACT_VERSION,
        "plugins": [],
    }


def test_compiler_tool_resolution_has_one_owner_and_a_legacy_seam(tmp_path):
    interpreter = tmp_path / "python"
    interpreter.write_text("", encoding="utf-8")
    sibling = tmp_path / "compiler"
    sibling.write_text("", encoding="utf-8")
    sibling.chmod(0o755)

    assert installation.resolve_compiler_tool is compilers.resolve_compiler_tool
    assert compilers.resolve_compiler_tool(
        "compiler",
        python_executable=interpreter,
        executable_check=lambda path, _mode: path == sibling,
        path_search=lambda _name: "/wrong/path",
    ) == sibling
    assert compilers.resolve_compiler_tool(
        "missing",
        python_executable=interpreter,
        executable_check=lambda _path, _mode: False,
        path_search=lambda _name: "/tools/missing",
    ) == Path("/tools/missing")


def test_desktop_history_revocation_has_one_owner_and_legacy_identities():
    assert desktop.revoke_desktop_history is desktop_history.revoke_desktop_history
    assert cli.revoke_desktop_history is desktop_history.revoke_desktop_history
    assert desktop.DESKTOP_REVOCATION_CONFIRMATION == (
        desktop_history.DESKTOP_REVOCATION_CONFIRMATION
    )
    assert desktop.revoke_desktop_history.__module__ == "omnitensor.training.desktop"


def test_desktop_history_owner_lock_seam_stays_live_after_legacy_import(monkeypatch, tmp_path):
    observed = []

    @contextmanager
    def lock(root, name):
        observed.append((root, name))
        yield

    monkeypatch.setattr(desktop_history, "store_lock", lock)
    assert desktop_history.revoke_desktop_history(
        tmp_path / "missing.jsonl",
        desktop_history.DESKTOP_REVOCATION_CONFIRMATION,
    ) is False
    assert observed == [(tmp_path, ".desktop-training.lock")]
