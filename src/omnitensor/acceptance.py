"""Verify that an installed OmniTensor is actually serving, before use cases.

A successful install is not evidence of a working one: the unit can be active
while the bus name is owned by something else, the schemas can be missing from
the installed package, the snapshot can be stale from a previous run, and the
applet can be a half-copied tree.  Each of those looks fine from ``systemctl``
and produces a desktop that silently shows nothing.

Every probe is injected, so the whole report is exercisable without a bus, a
service, or a desktop; the default adapters are wiring, not coupling.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import jsonschema

from .registry import (
    ManifestError,
    _schema_path,
    bundled_workloads_path,
    load_schema,
    load_workloads,
    validate_document,
)

# Every document the applet exchanges with the service.  A packaging mistake
# that drops one of these only surfaces when a user triggers that exact path.
REQUIRED_SCHEMAS = (
    "runtime-snapshot.schema.json",
    "runtime-command.schema.json",
    "runtime-acknowledgement.schema.json",
    "runtime-job-submit.schema.json",
    "runtime-job-cancel.schema.json",
    "runtime-job-acknowledgement.schema.json",
    "workload-manifest.schema.json",
    "plugin-inventory.schema.json",
    "runtime-refusal.schema.json",
    "runtime-job-result-request.schema.json",
    "runtime-job-result.schema.json",
)
DEFAULT_SNAPSHOT_MAX_AGE_MS = 30_000
# Publisher and verifier clocks are the same clock here, but they are read a
# moment apart, so a small negative age is ordinary rounding, not a fault.
FUTURE_TOLERANCE_MS = 2_000


@dataclass(frozen=True, slots=True)
class Check:
    """One named verification outcome with the evidence behind it."""

    name: str
    ok: bool
    detail: str


@dataclass(frozen=True, slots=True)
class InstallationReport:
    checks: tuple[Check, ...]

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    @property
    def failures(self) -> tuple[Check, ...]:
        return tuple(check for check in self.checks if not check.ok)

    def render(self) -> str:
        return "\n".join(
            f"{'PASS' if check.ok else 'FAIL'}  {check.name}: {check.detail}"
            for check in self.checks
        )


@runtime_checkable
class ServiceProbe(Protocol):
    """Whether the service unit is installed and currently running."""

    def unit_state(self) -> tuple[bool, str]: ...


@runtime_checkable
class BusProbe(Protocol):
    """Whether the D-Bus surface answers with a contract-valid reply."""

    def apply_command(self, text: str) -> str: ...


def check_executable(
    name: str = "omnitensor",
    *,
    resolver: Callable[[str], str | None] = shutil.which,
) -> Check:
    """The unit's ExecStart must resolve to something runnable."""
    resolved = resolver(name)
    if not resolved:
        return Check("executable", False, f"{name} is not on PATH")
    return Check("executable", True, f"{name} resolves to {resolved}")


def check_service(probe: ServiceProbe) -> Check:
    running, detail = probe.unit_state()
    return Check("service", running, detail)


def check_schemas(names: Sequence[str] = REQUIRED_SCHEMAS) -> Check:
    """Every canonical schema must be installed and be a valid schema.

    Present-but-invalid is the dangerous case: validation would raise on the
    first document instead of rejecting it, and the applet contract would fail
    at runtime rather than at install time.
    """
    missing: list[str] = []
    invalid: list[str] = []
    for name in names:
        try:
            _schema_path(name)
        except FileNotFoundError:
            missing.append(name)
            continue
        try:
            jsonschema.Draft202012Validator.check_schema(load_schema(name))
        except (jsonschema.SchemaError, ValueError, OSError):
            invalid.append(name)
    if missing or invalid:
        parts = []
        if missing:
            parts.append(f"missing {', '.join(sorted(missing))}")
        if invalid:
            parts.append(f"invalid {', '.join(sorted(invalid))}")
        return Check("schemas", False, "; ".join(parts))
    return Check("schemas", True, f"{len(names)} canonical schemas installed and valid")


def check_workload_catalog(root: Path | None = None) -> Check:
    """The bundled catalog must load; a bundled manifest failing is a packaging bug."""
    try:
        resolved = root or bundled_workloads_path()
    except FileNotFoundError as error:
        return Check("workloads", False, str(error))
    try:
        workloads = load_workloads(resolved)
    except ManifestError as error:
        return Check("workloads", False, f"bundled catalog is invalid: {error}")
    if not workloads:
        return Check("workloads", False, f"no bundled workloads under {resolved}")
    return Check("workloads", True, f"{len(workloads)} bundled workloads loaded")


def check_plugin_discovery(discover: Callable[[], Iterable]) -> Check:
    """Discovery must complete without importing plugin code."""
    try:
        candidates = tuple(discover())
    except Exception as error:  # noqa: BLE001 - any discovery failure is a failed install
        return Check("discovery", False, f"discovery raised {type(error).__name__}")
    return Check("discovery", True, f"{len(candidates)} plugin candidates discovered")


def check_isolation(specs: Iterable) -> Check:
    """Every external worker must be launched under a sandbox.

    A worker started without one runs with the service's own reach, so an
    install that silently loses the sandbox is worse than one that fails.
    """
    external = tuple(specs)
    unsandboxed = [spec.plugin_id for spec in external if getattr(spec, "sandbox", None) is None]
    if unsandboxed:
        return Check(
            "isolation",
            False,
            f"workers without a sandbox: {', '.join(sorted(unsandboxed))}",
        )
    return Check("isolation", True, f"{len(external)} external workers are sandboxed")


def check_bus(probe: BusProbe) -> Check:
    """The bus must answer, and its reply must satisfy the applet contract."""
    try:
        reply = probe.apply_command("not-json")
    except Exception as error:  # noqa: BLE001 - transport failures are arbitrary
        return Check("dbus", False, f"ApplyCommand raised {type(error).__name__}")
    try:
        acknowledgement = json.loads(reply)
    except ValueError:
        return Check("dbus", False, "ApplyCommand did not return JSON")
    violations = validate_document("runtime-acknowledgement.schema.json", acknowledgement)
    if violations:
        return Check("dbus", False, f"acknowledgement violates contract: {violations[0]}")
    if acknowledgement.get("status") != "rejected":
        return Check("dbus", False, "invalid command was not rejected")
    return Check("dbus", True, f"ApplyCommand answered at revision {acknowledgement['revision']}")


def check_snapshot(
    path: Path,
    *,
    now_ms: int,
    max_age_ms: int = DEFAULT_SNAPSHOT_MAX_AGE_MS,
) -> Check:
    """The published snapshot must exist, be contract-valid, and be fresh.

    A stale document is the failure that looks most like success: the applet
    renders it happily while the service behind it is no longer publishing.
    """
    try:
        document = json.loads(Path(path).read_bytes())
    except FileNotFoundError:
        return Check("snapshot", False, f"no snapshot at {path}")
    except (OSError, ValueError) as error:
        return Check("snapshot", False, f"snapshot is unreadable: {error}")
    violations = validate_document("runtime-snapshot.schema.json", document)
    if violations:
        return Check("snapshot", False, f"snapshot violates contract: {violations[0]}")
    generated_at = document["generatedAt"]
    if generated_at - now_ms > FUTURE_TOLERANCE_MS:
        # A future timestamp would otherwise read as permanently fresh, which
        # is exactly the staleness this check exists to catch: a publisher with
        # a wrong clock, or a document nobody is updating any more, would pass
        # forever.
        return Check(
            "snapshot",
            False,
            f"snapshot is stamped {generated_at - now_ms} ms in the future",
        )
    age_ms = max(0, now_ms - generated_at)
    if age_ms > max_age_ms:
        return Check("snapshot", False, f"snapshot is {age_ms} ms old; limit is {max_age_ms}")
    return Check("snapshot", True, f"snapshot is contract-valid and {age_ms} ms old")


def check_applet(root: Path, checksums: dict[str, str]) -> Check:
    """The installed applet tree must match the payload it was built from."""
    root = Path(root)
    if not root.is_dir():
        return Check("applet", False, f"no applet installed at {root}")
    missing: list[str] = []
    altered: list[str] = []
    for relative, digest in sorted(checksums.items()):
        installed = root / relative
        if not installed.is_file():
            missing.append(relative)
            continue
        if _sha256(installed) != digest:
            altered.append(relative)
    if missing or altered:
        parts = []
        if missing:
            parts.append(f"missing {len(missing)} file(s): {missing[0]}")
        if altered:
            parts.append(f"altered {len(altered)} file(s): {altered[0]}")
        return Check("applet", False, "; ".join(parts))
    return Check("applet", True, f"{len(checksums)} applet files match their checksums")


# Every accelerator lane and the runtime import that proves it can execute.
# There is no CPU entry by design: a CPU-only runtime is not a backend here.
BACKEND_RUNTIMES = (
    ("gpu", "ncnn", "install the [gpu] extra"),
    ("gpu", "onnxruntime", "install the [gpu-onnx-cuda] or [gpu-onnx-rocm] extra"),
    ("npu", "openvino", "install the [npu] extra"),
    ("tpu", "tflite_runtime", "install the [tpu] extra"),
)


def check_backends(runtimes: Sequence[tuple[str, str, str]] = BACKEND_RUNTIMES) -> Check:
    """At least one accelerator runtime must be importable.

    Without this the whole report can pass while the service can execute
    nothing: it starts, publishes a contract-valid snapshot, answers the bus,
    and refuses every job because no runtime resolves.  That is the most
    misleading state this install has, and it was reached by the documented
    install command, which omitted the accelerator extra.

    Importability is necessary, not sufficient — a present runtime may still
    find no device — so this reports what resolved rather than promising it
    will run.
    """
    import importlib.util  # noqa: PLC0415 - only needed for this probe

    resolved = []
    for backend, module, _remedy in runtimes:
        try:
            if importlib.util.find_spec(module) is not None:
                resolved.append(f"{backend}:{module}")
        except (ImportError, ValueError):
            # A broken or half-removed distribution is not an installed one.
            continue
    if not resolved:
        remedies = sorted({remedy for _backend, _module, remedy in runtimes})
        return Check(
            "backends",
            False,
            "no accelerator runtime is importable, so every profile is unavailable; "
            + "; ".join(remedies),
        )
    return Check("backends", True, f"accelerator runtimes available: {', '.join(resolved)}")


def verify_installation(checks: Sequence[Check]) -> InstallationReport:
    """Collect the individual probes into one ordered report."""
    return InstallationReport(tuple(checks))


def _sha256(path: Path) -> str:
    import hashlib  # noqa: PLC0415 - only needed when an applet is being verified

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SystemdUserServiceProbe:
    """:class:`ServiceProbe` over ``systemctl --user``."""

    def __init__(self, unit: str = "omnitensor.service", *, runner=None):
        self._unit = unit
        self._runner = runner or _run_command

    def unit_state(self) -> tuple[bool, str]:
        code, output = self._runner(["systemctl", "--user", "is-active", self._unit])
        state = output.strip() or "unknown"
        return code == 0 and state == "active", f"{self._unit} is {state}"


class DbusApplyCommandProbe:
    """:class:`BusProbe` over the session bus."""

    def __init__(self, *, bus_name: str = "org.cinnamon.OmniTensor1", timeout_s: float = 10.0):
        self._bus_name = bus_name
        self._timeout_s = timeout_s

    def apply_command(self, text: str) -> str:  # pragma: no cover - needs a live bus
        import asyncio  # noqa: PLC0415

        from dbus_fast import BusType  # noqa: PLC0415
        from dbus_fast.aio import MessageBus  # noqa: PLC0415

        object_path = "/" + self._bus_name.replace(".", "/")

        async def call() -> str:
            bus = await MessageBus(bus_type=BusType.SESSION).connect()
            try:
                introspection = await bus.introspect(self._bus_name, object_path)
                proxy = bus.get_proxy_object(self._bus_name, object_path, introspection)
                interface = proxy.get_interface(self._bus_name)
                return await interface.call_apply_command(text)
            finally:
                bus.disconnect()

        return asyncio.run(asyncio.wait_for(call(), timeout=self._timeout_s))


def _run_command(argv: Sequence[str]) -> tuple[int, str]:  # pragma: no cover - thin shim
    import subprocess  # noqa: PLC0415

    completed = subprocess.run(argv, capture_output=True, text=True, check=False)
    return completed.returncode, completed.stdout


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
    bus: BusProbe | None = None,
) -> InstallationReport:
    """The checks a real host runs after installing the service and applet."""
    from .plugins.discovery import discover_plugin_metadata  # noqa: PLC0415
    from .plugins.identity import resolve_plugin_identities  # noqa: PLC0415
    from .plugins.loading import external_worker_specs  # noqa: PLC0415

    bundled = bundled_workloads_path()
    checks = [
        check_executable(),
        check_service(service or SystemdUserServiceProbe()),
        check_schemas(),
        check_workload_catalog(bundled),
        check_plugin_discovery(lambda: discover_plugin_metadata(bundled_root=bundled)),
        check_isolation(
            external_worker_specs(
                resolve_plugin_identities(
                    discover_plugin_metadata(bundled_root=bundled)
                ).plugins
            )
        ),
        check_bus(bus or DbusApplyCommandProbe()),
        check_snapshot(snapshot_path, now_ms=now_ms),
        check_backends(),
    ]
    if applet_root is not None and applet_checksums is not None:
        checks.append(check_applet(applet_root, load_applet_checksums(applet_checksums)))
    return verify_installation(checks)


def main(argv: Sequence[str] | None = None) -> int:
    """Print the installation report; exit non-zero when any check fails."""
    import argparse  # noqa: PLC0415
    import os  # noqa: PLC0415
    import time  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        description="Verify an installed OmniTensor before provisioning use cases",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path(
            os.environ.get(
                "OMNITENSOR_STATE_PATH",
                "~/.local/state/tpu-workload-manager/state.json",
            )
        ).expanduser(),
    )
    parser.add_argument("--applet-root", type=Path, default=None)
    parser.add_argument("--applet-checksums", type=Path, default=None)
    arguments = parser.parse_args(argv)

    report = build_default_report(
        snapshot_path=arguments.snapshot,
        now_ms=int(time.time() * 1000),
        applet_root=arguments.applet_root,
        applet_checksums=arguments.applet_checksums,
    )
    print(report.render())
    return 0 if report.ok else 1
