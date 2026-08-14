"""Named accelerator device, lease, and minimal Vulkan sysfs resources."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

ACCELERATOR_PERMISSIONS = frozenset({"accelerator:gpu", "accelerator:npu"})
VULKAN_METADATA_ROOTS = (
    Path("/usr/share/vulkan"),
    Path("/etc/vulkan"),
    Path("/usr/share/libdrm"),
)
SYS_CHAR_ROOT = Path("/sys/dev/char")
SYS_DEVICES_ROOT = Path("/sys/devices")


def accelerator_lease_path(
    root: Path | None,
    declared: frozenset[str],
    granted: frozenset[str],
) -> Path | None:
    permissions = sorted(ACCELERATOR_PERMISSIONS & declared & granted)
    if root is None or not permissions:
        return None
    if len(permissions) != 1:
        raise ValueError("a plugin worker must use exactly one accelerator lease")
    accelerator = permissions[0].partition(":")[2]
    path = root / f"_accelerator-{accelerator}.lock"
    path.touch(exist_ok=True)
    path.chmod(0o600)
    return path.resolve()


def accelerator_paths(
    declared: frozenset[str],
    granted: frozenset[str],
    devices: Mapping[str, Path],
) -> tuple[Path, ...]:
    paths = []
    for permission in sorted(ACCELERATOR_PERMISSIONS & declared & granted):
        accelerator = permission.partition(":")[2]
        path = devices.get(accelerator)
        if path is not None and path.exists():
            paths.append(path)
    return tuple(paths)


def vulkan_sysfs_resources(
    devices: Sequence[Path],
    *,
    char_root: Path = SYS_CHAR_ROOT,
    devices_root: Path = SYS_DEVICES_ROOT,
) -> tuple[tuple[Path, ...], tuple[tuple[str, Path], ...]]:
    """Expose only the selected DRM device identity needed by libdrm."""
    identities: list[Path] = []
    links: list[tuple[str, Path]] = []
    canonical_devices_root = devices_root.resolve()
    for device in devices:
        if device.parent != Path("/dev/dri") or not device.name.startswith("renderD"):
            continue
        status = device.stat()
        node_id = f"{os.major(status.st_rdev)}:{os.minor(status.st_rdev)}"
        candidate = char_root / node_id
        try:
            source = os.readlink(candidate)
            render_identity = candidate.resolve(strict=True)
            identity = (render_identity / "device").resolve(strict=True)
        except OSError:
            continue
        if (
            Path(source).is_absolute()
            or render_identity.name != device.name
            or not render_identity.is_relative_to(identity)
            or not identity.is_relative_to(canonical_devices_root)
        ):
            continue
        try:
            if (render_identity / "dev").read_text(encoding="ascii").strip() != node_id:
                continue
        except (OSError, UnicodeError):
            continue
        identities.append(identity)
        links.append((source, Path("/sys/dev/char") / node_id))
    return tuple(dict.fromkeys(identities)), tuple(dict.fromkeys(links))


__all__ = ["accelerator_lease_path", "accelerator_paths", "vulkan_sysfs_resources"]
