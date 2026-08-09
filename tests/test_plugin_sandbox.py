from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from omnitensor.plugins import (
    BWRAP_PATH,
    MAX_SANDBOX_PATHS,
    FilesystemSandbox,
    SandboxPolicyError,
)


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
    permissions = {
        f"read:{tmp_path}/source-{index}"
        for index in range(MAX_SANDBOX_PATHS + 1)
    }
    with pytest.raises(SandboxPolicyError, match="at most"):
        FilesystemSandbox.from_permissions(permissions, permissions)
    for runtime_path in (Path("relative"), tmp_path / "missing", Path("/")):
        with pytest.raises(SandboxPolicyError, match="runtime path"):
            FilesystemSandbox.from_permissions(
                set(), set(), runtime_paths=(runtime_path,)
            )


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
    assert ("--ro-bind", "/", "/") not in zip(
        command, command[1:], command[2:], strict=False
    )
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
