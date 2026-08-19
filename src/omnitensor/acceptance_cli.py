"""Default host orchestration and CLI for installation acceptance."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .acceptance_checks import (
    check_applet,
    check_applet_contract,
    check_backends,
    check_bus,
    check_confinement,
    check_executable,
    check_isolation,
    check_plugin_discovery,
    check_provider_imports,
    check_schemas,
    check_service,
    check_snapshot,
    check_workload_catalog,
    verify_installation,
)
from .acceptance_contracts import (
    ControlProbe,
    InstallationReport,
    ServiceProbe,
)
from .acceptance_probes import SocketApplyCommandProbe, SystemdUserServiceProbe
from .registry import bundled_workloads_path

DEFAULT_APPLET_ROOT = "~/.local/share/cinnamon/applets/cinnamon-xpuwlm@geraldo-netto"


def load_applet_checksums(path: Path) -> dict[str, str]:
    """Parse a ``SHA256SUMS`` payload manifest produced by the applet packager."""
    checksums: dict[str, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, relative = line.partition("  ")
        if not relative:
            raise ValueError(f"malformed checksum line: {line!r}")
        checksums[relative.strip()] = digest.strip()
    if not checksums:
        raise ValueError(f"no checksums in {path}")
    return checksums


def build_default_report(
    *,
    snapshot_path: Path,
    now_ms: int,
    applet_root: Path | None = None,
    applet_checksums: Path | None = None,
    service: ServiceProbe | None = None,
    bus: ControlProbe | None = None,
) -> InstallationReport:
    """The checks a real host runs after installing the service and applet."""
    from .plugins.discovery import discover_plugin_metadata  # noqa: PLC0415
    from .plugins.identity import resolve_plugin_identities  # noqa: PLC0415
    from .plugins.loading import external_worker_specs  # noqa: PLC0415

    bundled_root = bundled_workloads_path
    executable_check = check_executable
    service_check = check_service
    schemas_check = check_schemas
    workloads_check = check_workload_catalog
    discovery_check = check_plugin_discovery
    provider_import_check = check_provider_imports
    isolation_check = check_isolation
    bus_check = check_bus
    snapshot_check = check_snapshot
    backends_check = check_backends
    confinement_check = check_confinement
    systemd_probe = SystemdUserServiceProbe
    socket_probe = SocketApplyCommandProbe

    bundled = bundled_root()
    checks = [
        executable_check(),
        service_check(service or systemd_probe()),
        schemas_check(),
        workloads_check(bundled),
        discovery_check(lambda: discover_plugin_metadata(bundled_root=bundled)),
        provider_import_check(),
        isolation_check(
            external_worker_specs(
                resolve_plugin_identities(discover_plugin_metadata(bundled_root=bundled)).plugins
            )
        ),
        bus_check(bus or socket_probe()),
        snapshot_check(snapshot_path, now_ms=now_ms),
        backends_check(),
        confinement_check(),
    ]
    default_applet_root = DEFAULT_APPLET_ROOT
    root = Path(applet_root) if applet_root is not None else Path(default_applet_root).expanduser()
    if root.is_dir():
        applet_contract_check = check_applet_contract
        checks.append(applet_contract_check(root, snapshot_path))
        if applet_checksums is not None:
            checksum_loader = load_applet_checksums
            applet_check = check_applet
            checks.append(applet_check(root, checksum_loader(applet_checksums)))
    verifier = verify_installation
    return verifier(checks)


def main(argv: Sequence[str] | None = None) -> int:
    """Print the installation report; exit non-zero when any check fails."""
    import argparse  # noqa: PLC0415
    import os  # noqa: PLC0415
    import time  # noqa: PLC0415

    default_applet_root = DEFAULT_APPLET_ROOT
    parser = argparse.ArgumentParser(
        description="Verify an installed OmniTensor before provisioning use cases",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path(
            os.environ.get(
                "OMNITENSOR_STATE_PATH",
                "~/.local/state/xpu-workload-manager/state.json",
            )
        ).expanduser(),
    )
    parser.add_argument(
        "--applet-root",
        type=Path,
        default=None,
        help=f"installed applet to check against (default {default_applet_root})",
    )
    parser.add_argument("--applet-checksums", type=Path, default=None)
    arguments = parser.parse_args(argv)

    builder = build_default_report
    report = builder(
        snapshot_path=arguments.snapshot,
        now_ms=int(time.time() * 1000),
        applet_root=arguments.applet_root,
        applet_checksums=arguments.applet_checksums,
    )
    print(report.render())
    return 0 if report.ok else 1
