"""Fail-closed per-plugin filesystem and device sandbox construction."""

from __future__ import annotations

import os
import posixpath
import re
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

BWRAP_PATH = "/usr/bin/bwrap"
MAX_SANDBOX_PATHS = 64
_FILESYSTEM_ACTIONS = frozenset({"read", "write", "device"})
# The only permission that re-opens the network namespace.  Anything else a
# plugin declares leaves the worker with no route to anywhere.
NETWORK_ACTION = "network"
# bwrap gives a new network namespace its own loopback, so "localhost" is a
# scope the sandbox can honour exactly: the worker can talk to itself and to
# nothing else.  "outbound" is the only scope that shares the host namespace,
# because bwrap has no partial sharing to offer — anything narrower would be a
# claim the sandbox cannot enforce.
NETWORK_LOCALHOST = "localhost"
NETWORK_OUTBOUND = "outbound"
_NETWORK_SCOPES = frozenset({NETWORK_LOCALHOST, NETWORK_OUTBOUND})
_DEVICE_PATH = re.compile(
    r"^/dev/(?:apex_[0-9]+|accel/accel[0-9]+|dri/renderD[0-9]+)$"
)
_FORBIDDEN_FILESYSTEM_ROOTS = (
    "/proc",
    "/dev",
    "/sys/kernel/debug",
    "/sys/kernel/tracing",
    "/sys/fs/bpf",
)
_SYSTEM_RUNTIME_ROOTS = (
    "/usr/lib",
    "/usr/lib64",
    "/usr/local/lib",
    "/lib",
    "/lib64",
)


class SandboxPolicyError(ValueError):
    """A permission cannot be represented by the fail-closed sandbox."""


@dataclass(frozen=True, slots=True)
class FilesystemSandbox:
    """Immutable bwrap policy derived only from declared active grants."""

    read_paths: tuple[str, ...]
    write_paths: tuple[str, ...]
    device_paths: tuple[str, ...]
    runtime_paths: tuple[str, ...]
    network: bool = False
    python_path: str | None = None

    @classmethod
    def from_permissions(
        cls,
        declared: Collection[str],
        granted: Collection[str],
        *,
        runtime_paths: Sequence[str | Path] = (),
        python_path: str | Path | None = None,
    ) -> FilesystemSandbox:
        declared_set = frozenset(declared)
        granted_set = frozenset(granted)
        undeclared = granted_set - declared_set
        if undeclared:
            raise SandboxPolicyError(
                f"sandbox grant is undeclared: {min(undeclared)}"
            )
        filesystem = []
        for permission in sorted(granted_set):
            action, separator, value = permission.partition(":")
            if action not in _FILESYSTEM_ACTIONS:
                continue
            if not separator:
                raise SandboxPolicyError("filesystem permission is malformed")
            filesystem.append((action, _permission_path(action, value)))
        if len(filesystem) > MAX_SANDBOX_PATHS:
            raise SandboxPolicyError(
                f"sandbox allows at most {MAX_SANDBOX_PATHS} filesystem paths"
            )
        writes = {value for action, value in filesystem if action == "write"}
        reads = {
            value
            for action, value in filesystem
            if action == "read" and value not in writes
        }
        devices = {value for action, value in filesystem if action == "device"}
        trusted = tuple(sorted({_runtime_path(path) for path in runtime_paths}))
        trusted_python = _trusted_python_path(python_path, trusted)
        return cls(
            tuple(sorted(reads)),
            tuple(sorted(writes)),
            tuple(sorted(devices)),
            trusted,
            _network_granted(granted_set),
            trusted_python,
        )

    def wrap(self, argv: Sequence[str]) -> tuple[str, ...]:
        """Wrap a validated worker argv in isolated mount/process namespaces."""
        executable_roots: list[str] = []
        if argv and Path(argv[0]).is_absolute() and Path(argv[0]).exists():
            executable_roots = list(_executable_runtime_paths(argv[0]))
        command = [
            BWRAP_PATH,
            "--die-with-parent",
            "--new-session",
            "--unshare-user",
            "--unshare-pid",
            "--unshare-uts",
            "--unshare-ipc",
            "--unshare-cgroup",
            # Every capability is dropped rather than relied upon to be absent:
            # the worker is unprivileged in the parent namespace already, and
            # stating it here keeps that true if the launcher ever changes.
            "--cap-drop",
            "ALL",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--clearenv",
            "--setenv",
            "PATH",
            "/usr/bin",
            "--setenv",
            "PYTHONDONTWRITEBYTECODE",
            "1",
            "--setenv",
            "PYTHONNOUSERSITE",
            "1",
        ]
        command.extend(_python_environment(self.python_path))
        # A worker with no declared network grant gets its own empty network
        # namespace, so a plugin cannot reach a socket, a name server, or the
        # local bus regardless of what its code attempts.
        command.append("--share-net" if self.network else "--unshare-net")
        runtime_roots = [
            path for path in _SYSTEM_RUNTIME_ROOTS if Path(path).exists()
        ]
        runtime_roots.extend(executable_roots)
        runtime_roots.extend(self.runtime_paths)
        runtime_roots = list(_minimal_runtime_roots(runtime_roots))
        bindings = [
            *(path for path in runtime_roots),
            *self.read_paths,
            *self.write_paths,
            *self.device_paths,
        ]
        for directory in _target_directories(bindings):
            command.extend(("--dir", directory))
        for path in sorted(set(runtime_roots)):
            command.extend(("--ro-bind", path, path))
        for path in self.read_paths:
            command.extend(("--ro-bind-try", path, path))
        for path in self.write_paths:
            command.extend(("--bind-try", path, path))
        for path in self.device_paths:
            command.extend(("--dev-bind-try", path, path))
        command.extend(("--", *argv))
        return tuple(command)


def _network_granted(granted: Collection[str]) -> bool:
    """Whether the host network namespace is shared with the worker.

    A malformed scope is rejected rather than ignored: treating it as "no
    network" would hide a policy the author intended, and treating it as
    "network" would grant more than was declared.  Only ``outbound`` shares
    the host namespace; ``localhost`` is served by the private namespace the
    worker gets anyway.
    """
    enabled = False
    for permission in sorted(granted):
        action, separator, value = permission.partition(":")
        if action != NETWORK_ACTION:
            continue
        if not separator or value not in _NETWORK_SCOPES:
            raise SandboxPolicyError(
                f"network permission scope is not recognised: {permission}"
            )
        enabled = enabled or value == NETWORK_OUTBOUND
    return enabled


def _trusted_python_path(
    value: str | Path | None,
    runtime_paths: Collection[str],
) -> str | None:
    if value is None:
        return None
    trusted = _runtime_path(value)
    if trusted not in runtime_paths:
        raise SandboxPolicyError("sandbox Python path must be a declared runtime path")
    return trusted


def _python_environment(python_path: str | None) -> tuple[str, ...]:
    """Bootstrap only trusted OmniTensor code before worker argument parsing."""
    if python_path is None:
        return ()
    # The supervisor and worker must import the same OmniTensor code. This is
    # explicit because --clearenv removes a caller PYTHONPATH, and an editable
    # or relocated install otherwise points outside the mount namespace before
    # the worker can parse --import-path.
    return "--setenv", "PYTHONPATH", python_path


def _permission_path(action: str, value: str) -> str:
    if not value.startswith("/") or "\0" in value:
        raise SandboxPolicyError("filesystem permission path must be absolute")
    if action == "device" and "*" in value:
        raise SandboxPolicyError("device permission is not an accelerator node")
    if value.count("*") > 1 or ("*" in value and not value.endswith("/*")):
        raise SandboxPolicyError("filesystem permission allows only a trailing wildcard")
    path = value[:-2] if value.endswith("/*") else value
    if path == "/" or posixpath.normpath(path) != path or ".." in PurePosixPath(path).parts:
        raise SandboxPolicyError("filesystem permission path is not canonical")
    if action == "device":
        if _DEVICE_PATH.fullmatch(path) is None:
            raise SandboxPolicyError("device permission is not an accelerator node")
        return path
    if any(path == root or path.startswith(f"{root}/") for root in _FORBIDDEN_FILESYSTEM_ROOTS):
        raise SandboxPolicyError("filesystem permission exposes a host namespace")
    return path


def _runtime_path(value: str | Path) -> str:
    path = Path(value)
    if not path.is_absolute() or not path.exists():
        raise SandboxPolicyError("sandbox runtime paths must exist and be absolute")
    absolute = str(path.absolute())
    if absolute == "/":
        raise SandboxPolicyError("sandbox runtime path cannot expose the host root")
    return absolute


def _executable_runtime_paths(value: str | Path) -> tuple[str, ...]:
    current = Path(value).absolute()
    paths = []
    for _index in range(8):
        paths.append(_runtime_path(current))
        if not current.is_symlink():
            break
        target = Path(os.readlink(current))
        current = target if target.is_absolute() else current.parent / target
    resolved = _runtime_path(current.resolve())
    paths.append(resolved)
    return tuple(dict.fromkeys(paths))


def _minimal_runtime_roots(paths: Sequence[str]) -> tuple[str, ...]:
    kept = []
    for path in sorted(set(paths), key=lambda item: (item.count("/"), item)):
        if any(
            Path(parent).is_dir() and PurePosixPath(path).is_relative_to(parent)
            for parent in kept
        ):
            continue
        kept.append(path)
    return tuple(kept)


def _target_directories(paths: Sequence[str]) -> tuple[str, ...]:
    directories = set()
    for value in paths:
        path = PurePosixPath(value)
        parent = path if Path(value).is_dir() else path.parent
        directories.update(
            str(candidate)
            for candidate in (parent, *parent.parents)
            if str(candidate) != "/"
        )
    return tuple(sorted(directories, key=lambda item: (item.count("/"), item)))
