from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from omnitensor.plugins import (
    BWRAP_PATH,
    MAX_SANDBOX_PATHS,
    FilesystemSandbox,
    SandboxPolicyError,
)
from omnitensor.plugins import sandbox as sandbox_module


def test_filesystem_permission_split_is_exact_and_deterministic(tmp_path):
    readable = tmp_path / "readable"
    writable = tmp_path / "writable"
    colon_path = tmp_path / "name:with-colon"

    assert sandbox_module._filesystem_paths(
        frozenset(
            {
                "unrelated:value",
                f"write:{writable}",
                "device:/dev/dri/renderD128",
                f"read:{readable}/*",
                f"read:{colon_path}",
            }
        )
    ) == [
        ("device", "/dev/dri/renderD128"),
        ("read", str(colon_path)),
        ("read", str(readable)),
        ("write", str(writable)),
    ]


def test_policy_uses_only_declared_active_filesystem_and_device_grants(tmp_path):
    readable = tmp_path / "input"
    writable = tmp_path / "output"
    readable.mkdir()
    writable.mkdir()
    permissions = {
        f"read:{readable}/*",
        f"write:{writable}",
        "device:/dev/apex_0",
        "network:localhost",
    }

    sandbox = FilesystemSandbox.from_permissions(
        permissions,
        permissions,
        runtime_paths=(tmp_path,),
    )

    assert sandbox.read_paths == (str(readable),)
    assert sandbox.write_paths == (str(writable),)
    assert sandbox.device_paths == ("/dev/apex_0",)
    assert sandbox.runtime_paths == (str(tmp_path.resolve()),)


def test_undeclared_or_ungranted_paths_never_enter_the_policy(tmp_path):
    allowed = f"read:{tmp_path}/allowed"
    other = f"read:{tmp_path}/other"

    sandbox = FilesystemSandbox.from_permissions({allowed, other}, {allowed})

    assert sandbox.read_paths == (f"{tmp_path}/allowed",)
    assert f"{tmp_path}/other" not in sandbox.wrap(("/usr/bin/python3",))
    with pytest.raises(SandboxPolicyError, match="undeclared"):
        FilesystemSandbox.from_permissions({allowed}, {other})


def test_selected_files_permission_mounts_only_the_private_broker_root(tmp_path):
    broker = tmp_path / "broker"
    original = tmp_path / "original.txt"
    broker.mkdir()
    original.write_text("private", encoding="utf-8")
    permission = "files:read-selected"

    sandbox = FilesystemSandbox.from_permissions(
        {permission},
        {permission},
        selected_files_root=broker,
    )

    assert sandbox.read_paths == (str(broker.resolve()),)
    assert str(original) not in sandbox.wrap(("/usr/bin/python3",))
    with pytest.raises(SandboxPolicyError, match="brokered input root"):
        FilesystemSandbox.from_permissions({permission}, {permission})
    denied = FilesystemSandbox.from_permissions({permission}, set(), selected_files_root=broker)
    assert denied.read_paths == ()


def test_timezone_database_is_usable_and_read_only_inside_real_sandbox():
    zoneinfo = Path("/usr/share/zoneinfo")
    if not Path(BWRAP_PATH).is_file() or not zoneinfo.is_dir():
        pytest.skip("bubblewrap or system timezone database is unavailable")
    sandbox = FilesystemSandbox.from_permissions(
        set(),
        set(),
        runtime_paths=(zoneinfo,),
    )
    script = (
        "from datetime import datetime; from zoneinfo import ZoneInfo; "
        "zone=ZoneInfo('Europe/Rome'); "
        "assert datetime(2026,8,12,10,tzinfo=zone).isoformat().endswith('+02:00'); "
        "open('/usr/share/zoneinfo/UTC','wb')"
    )

    completed = subprocess.run(
        sandbox.wrap(("/usr/bin/python3", "-c", script)),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "Read-only file system" in completed.stderr


def test_host_owned_state_and_named_accelerator_are_narrowly_mounted(tmp_path):
    state = tmp_path / "plugin-state"
    lease = tmp_path / "gpu.lock"
    state.mkdir()
    lease.touch()
    device = Path("/dev/dri/renderD128")
    if not device.exists():
        pytest.skip("host has no renderD128 device node")

    sandbox = FilesystemSandbox.from_permissions(
        {"accelerator:gpu"},
        {"accelerator:gpu"},
        trusted_write_paths=(state, lease),
        accelerator_devices=(device,),
    )

    assert sandbox.write_paths == tuple(sorted((str(lease), str(state))))
    assert sandbox.device_paths == (str(device),)
    wrapped = sandbox.wrap(("/usr/bin/python3", "worker.py"))
    assert any(
        wrapped[index : index + 3] == ("--bind-try", str(state), str(state))
        for index in range(len(wrapped) - 2)
    )
    index = wrapped.index("--dev-bind-try")
    assert wrapped[index : index + 3] == ("--dev-bind-try", str(device), str(device))


def test_host_owned_resources_still_reject_missing_or_unallowlisted_paths(tmp_path):
    with pytest.raises(SandboxPolicyError, match="runtime paths must exist"):
        FilesystemSandbox.from_permissions(
            set(), set(), trusted_write_paths=(tmp_path / "missing",)
        )
    with pytest.raises(SandboxPolicyError, match="accelerator device is not allowlisted"):
        FilesystemSandbox.from_permissions(set(), set(), accelerator_devices=(tmp_path,))


def test_trusted_sysfs_link_targets_only_an_exact_mounted_device_identity():
    source = "../../devices/pci0000:00/0000:03:00.0/drm/renderD128"
    destination = "/sys/dev/char/226:128"

    sandbox = FilesystemSandbox.from_permissions(
        set(),
        set(),
        runtime_paths=(Path("/sys/devices"),),
        trusted_symlinks=((source, destination),),
    )

    assert sandbox.trusted_symlinks == ((source, destination),)
    wrapped = sandbox.wrap(("/usr/bin/python3",))
    index = wrapped.index("--symlink")
    assert wrapped[index : index + 3] == ("--symlink", source, destination)
    assert ("--ro-bind", "/sys/dev/char", "/sys/dev/char") not in tuple(
        zip(wrapped, wrapped[1:], wrapped[2:], strict=False)
    )


def test_trusted_symlink_parent_directories_are_exact_and_depth_ordered():
    assert sandbox_module._symlink_directories(
        (("source-a", "/z/a/b/link"), ("source-b", "/a/z/link"))
    ) == ("/a", "/z", "/a/z", "/z/a", "/z/a/b")


@pytest.mark.parametrize(
    ("source", "destination", "message"),
    [
        ("/sys/devices/gpu", "/sys/dev/char/226:128", "invalid"),
        ("../../devices/gpu", "/sys/dev/char/not-a-node", "invalid"),
        ("../../../outside/gpu", "/sys/dev/char/226:128", "invalid"),
    ],
)
def test_trusted_sysfs_links_fail_closed(source, destination, message):
    with pytest.raises(SandboxPolicyError, match=message):
        FilesystemSandbox.from_permissions(
            set(),
            set(),
            runtime_paths=(Path("/sys/devices"),),
            trusted_symlinks=((source, destination),),
        )


def test_trusted_sysfs_link_refuses_an_unmounted_device_identity():
    with pytest.raises(SandboxPolicyError, match="not mounted"):
        FilesystemSandbox.from_permissions(
            set(),
            set(),
            runtime_paths=(Path("/usr/lib"),),
            trusted_symlinks=(("../../devices/other/gpu", "/sys/dev/char/226:128"),),
        )


def test_trusted_sysfs_link_destination_cannot_be_duplicated():
    link = ("../../devices/gpu", "/sys/dev/char/226:128")
    with pytest.raises(SandboxPolicyError, match="duplicated"):
        FilesystemSandbox.from_permissions(
            set(),
            set(),
            runtime_paths=(Path("/sys/devices"),),
            trusted_symlinks=(link, link),
        )


@pytest.mark.parametrize(
    "permission, message",
    [
        ("read", "malformed"),
        ("read:relative", "absolute"),
        ("read:/", "canonical"),
        ("read:/tmp/../private", "canonical"),
        ("read:/tmp/*/secret", "trailing wildcard"),
        ("read:/proc/1/root", "host namespace"),
        ("write:/dev/shm", "host namespace"),
        ("read:/sys/kernel/debug/tracing", "host namespace"),
        ("device:/dev/null", "accelerator node"),
        ("device:/dev/apex_*", "accelerator node"),
    ],
)
def test_unsafe_filesystem_permissions_fail_closed(permission, message):
    with pytest.raises(SandboxPolicyError, match=message):
        FilesystemSandbox.from_permissions({permission}, {permission})


@pytest.mark.parametrize(
    "device",
    [
        "/dev/apex_0",
        "/dev/apex_19",
        "/dev/accel/accel0",
        "/dev/dri/renderD128",
    ],
)
def test_only_named_accelerator_device_families_are_allowed(device):
    permission = f"device:{device}"
    sandbox = FilesystemSandbox.from_permissions({permission}, {permission})
    assert sandbox.device_paths == (device,)


def test_path_count_and_runtime_roots_are_bounded_and_canonical(tmp_path):
    permissions = {f"read:{tmp_path}/source-{index}" for index in range(MAX_SANDBOX_PATHS + 1)}
    with pytest.raises(SandboxPolicyError, match="at most"):
        FilesystemSandbox.from_permissions(permissions, permissions)
    for runtime_path in (Path("relative"), tmp_path / "missing", Path("/")):
        with pytest.raises(SandboxPolicyError, match="runtime path"):
            FilesystemSandbox.from_permissions(set(), set(), runtime_paths=(runtime_path,))
    with pytest.raises(SandboxPolicyError, match="declared runtime path"):
        FilesystemSandbox.from_permissions(
            set(),
            set(),
            runtime_paths=(tmp_path,),
            python_path=tmp_path.parent,
        )


def test_worker_bootstrap_uses_only_its_trusted_package_root(tmp_path):
    package_root = tmp_path / "source"
    plugin_site = tmp_path / "plugin"
    package_root.mkdir()
    plugin_site.mkdir()
    sandbox = FilesystemSandbox.from_permissions(
        set(),
        set(),
        runtime_paths=(package_root, plugin_site),
        python_path=package_root,
    )

    command = sandbox.wrap(("/usr/bin/python3", "-m", "omnitensor.plugins.worker"))

    index = command.index("PYTHONPATH")
    assert command[index - 1] == "--setenv"
    assert command[index + 1] == str(package_root)
    assert str(plugin_site) not in command[index + 1]


def test_write_grant_dominates_same_path_read_grant(tmp_path):
    read = f"read:{tmp_path}"
    write = f"write:{tmp_path}"
    sandbox = FilesystemSandbox.from_permissions({read, write}, {read, write})
    assert sandbox.read_paths == ()
    assert sandbox.write_paths == (str(tmp_path),)


def test_wrapped_command_uses_private_process_mount_and_device_namespaces(tmp_path):
    readable = tmp_path / "readable"
    writable = tmp_path / "writable"
    readable.mkdir()
    writable.mkdir()
    declared = {
        f"read:{readable}",
        f"write:{writable}",
        "device:/dev/dri/renderD128",
    }
    sandbox = FilesystemSandbox.from_permissions(declared, declared)

    command = sandbox.wrap(("/usr/bin/python3", "-V"))

    assert command[0] == BWRAP_PATH
    for option in (
        "--die-with-parent",
        "--new-session",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-uts",
        "--unshare-ipc",
        "--unshare-cgroup",
        "--proc",
        "--dev",
        "--tmpfs",
        "--clearenv",
    ):
        assert option in command
    assert ("--ro-bind-try", str(readable), str(readable)) == _binding(
        command, "--ro-bind-try", str(readable)
    )
    assert ("--bind-try", str(writable), str(writable)) == _binding(
        command, "--bind-try", str(writable)
    )
    assert _binding(command, "--dev-bind-try", "/dev/dri/renderD128") == (
        "--dev-bind-try",
        "/dev/dri/renderD128",
        "/dev/dri/renderD128",
    )
    assert ("--ro-bind", "/", "/") not in zip(command, command[1:], command[2:], strict=False)
    assert command[-2:] == ("/usr/bin/python3", "-V")


def test_bubblewrap_denies_undeclared_reads_and_allows_declared_read(tmp_path):
    allowed = tmp_path / "allowed.txt"
    denied = tmp_path / "denied.txt"
    allowed.write_text("allowed", encoding="utf-8")
    denied.write_text("denied", encoding="utf-8")
    permission = f"read:{allowed}"
    sandbox = FilesystemSandbox.from_permissions({permission}, {permission})
    script = (
        "from pathlib import Path; import sys; "
        "print(Path(sys.argv[1]).read_text()); "
        "print(Path(sys.argv[2]).exists())"
    )

    result = subprocess.run(
        sandbox.wrap(("/usr/bin/python3", "-c", script, str(allowed), str(denied))),
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == ["allowed", "False"]


def _binding(command: tuple[str, ...], option: str, source: str) -> tuple[str, str, str]:
    index = command.index(option)
    while command[index + 1] != source:
        index = command.index(option, index + 1)
    return command[index : index + 3]


def test_a_worker_gets_no_network_by_default():
    """Nothing unshared the network namespace, so workers had full host reach."""
    policy = FilesystemSandbox.from_permissions(set(), set())

    command = policy.wrap(("/usr/bin/python3", "-m", "worker"))

    assert policy.network is False
    assert "--unshare-net" in command
    assert "--share-net" not in command


def test_localhost_is_served_by_the_private_namespace():
    """bwrap gives a new netns its own loopback, so this needs no host sharing."""
    policy = FilesystemSandbox.from_permissions({"network:localhost"}, {"network:localhost"})

    command = policy.wrap(("/usr/bin/python3",))

    assert policy.network is False
    assert "--unshare-net" in command


def test_only_an_outbound_grant_shares_the_host_network():
    policy = FilesystemSandbox.from_permissions({"network:outbound"}, {"network:outbound"})

    command = policy.wrap(("/usr/bin/python3",))

    assert policy.network is True
    assert "--share-net" in command
    assert "--unshare-net" not in command


def test_a_declared_but_ungranted_network_permission_grants_nothing():
    policy = FilesystemSandbox.from_permissions({"network:outbound"}, set())
    assert policy.network is False
    assert "--unshare-net" in policy.wrap(("/usr/bin/python3",))


@pytest.mark.parametrize(
    "permission", ["network:", "network:internet", "network:0.0.0.0", "network:*"]
)
def test_an_unrecognised_network_scope_is_refused_rather_than_ignored(permission):
    with pytest.raises(SandboxPolicyError, match="network permission scope"):
        FilesystemSandbox.from_permissions({permission}, {permission})


def test_every_capability_is_dropped():
    command = FilesystemSandbox.from_permissions(set(), set()).wrap(("/usr/bin/python3",))
    assert "--cap-drop" in command
    assert command[command.index("--cap-drop") + 1] == "ALL"


_NETWORK_PROBE = """
import socket
sock = socket.socket()
sock.settimeout(3)
try:
    sock.connect(("1.1.1.1", 53))
    print("reachable")
except OSError:
    print("unreachable")
"""


@pytest.mark.skipif(not Path(BWRAP_PATH).exists(), reason="bwrap is not installed")
@pytest.mark.parametrize(
    ("granted", "expected"),
    [(set(), "unreachable"), ({"network:outbound"}, "reachable")],
)
def test_network_isolation_holds_against_real_bwrap(granted, expected):
    """The argv is only a claim; this runs it and checks what actually happens."""
    policy = FilesystemSandbox.from_permissions(granted, granted)
    argv = policy.wrap((sys.executable, "-c", _NETWORK_PROBE))

    completed = subprocess.run(argv, capture_output=True, text=True, timeout=60, check=False)

    if completed.returncode != 0:
        pytest.skip(f"bwrap could not run here: {completed.stderr.strip()[:120]}")
    assert completed.stdout.strip() == expected
