"""The shipped systemd user unit must encode the service's real runtime
dependencies (regression for OMNI-0018)."""

from __future__ import annotations

import configparser
from pathlib import Path

UNIT_PATH = Path(__file__).resolve().parents[1] / "systemd" / "omnitensor.service"


def _load_unit() -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    # systemd units are INI-like; keep keys case-sensitive.
    parser.optionxform = str
    parser.read_string(UNIT_PATH.read_text(encoding="utf-8"))
    return parser


def test_unit_requires_the_session_bus_socket():
    """WantedBy=default.target starts the unit in non-graphical sessions too;
    without Requires=dbus.socket the session bus may be absent there and the
    service would fail and restart in a loop."""
    unit = _load_unit()["Unit"]
    assert "dbus.socket" in unit["Requires"].split()
    assert "dbus.socket" in unit["After"].split()


def test_unit_is_ordered_after_the_graphical_session():
    unit = _load_unit()["Unit"]
    assert "graphical-session.target" in unit["After"].split()


def test_unit_install_and_hardening_are_intact():
    parsed = _load_unit()
    assert parsed["Install"]["WantedBy"] == "default.target"
    service = parsed["Service"]
    assert service["Restart"] == "on-failure"
    assert service["NoNewPrivileges"] == "true"
    assert service["ProtectSystem"] == "strict"
    assert service["ProtectHome"] == "read-only"


def test_unit_creates_its_state_directories_instead_of_assuming_them():
    """ReadWritePaths= needs the directories to exist, and ProtectHome=read-only
    stops the service from creating them, so a fresh install could never
    persist policy or snapshots.  StateDirectory= creates them (OMNI-0129)."""
    service = _load_unit()["Service"]
    assert "ReadWritePaths" not in service
    assert set(service.get("StateDirectory", raw=True).split()) == {
        "omnitensor",
        "xpu-workload-manager",
    }


def test_unit_places_no_cgroup_or_resource_bound_on_the_service():
    """The delegated subtree only ever broke worker launch, so it is gone.

    The service's own PID stays in the delegated root, which makes the kernel
    refuse to populate `cgroup.subtree_control`; every per-worker cgroup then
    had no `pids.max` to write and every external worker failed to start.
    """
    service = _load_unit()["Service"]
    for bound in ("Delegate", "MemoryMax", "MemoryHigh", "CPUQuota", "TasksMax"):
        assert bound not in service, bound
