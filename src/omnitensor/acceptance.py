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
# Stated rather than discovered: a check that verifies whatever it happens to
# find can never report one missing, which is the failure it exists for — an
# install whose schema directory did not ship.  ``test_acceptance_install.py``
# asserts this list equals the schemas in the tree, so it cannot fall behind a
# new contract the way it fell behind ``runtime-contract``.
#
# Written out rather than read from the directory, and that is the whole point:
# a check that verifies whatever it happens to find can never report something
# missing.  This is the one restatement of the contract that must stay a
# restatement — the applet's validator was derived from the schema precisely
# because it is the opposite case, a second description of the same rules.
REQUIRED_SCHEMAS = (
    "runtime-contract.schema.json",
    "runtime-snapshot.schema.json",
    "runtime-command.schema.json",
    "runtime-acknowledgement.schema.json",
    "runtime-job-submit.schema.json",
    "runtime-job-cancel.schema.json",
    "runtime-job-acknowledgement.schema.json",
    "workload-manifest.schema.json",
    "model-recipe.schema.json",
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


DEFAULT_APPLET_ROOT = "~/.local/share/cinnamon/applets/cinnamon-xpuwlm@geraldo-netto"


def check_confinement() -> Check:
    """Whether a plugin worker could be confined on this host.

    Workers fail closed: on a kernel that cannot install a seccomp filter, or
    on an architecture with no syscall table, every one of them refuses to
    start rather than run plugin code unconfined. That is the right outcome
    and it was undiagnosable — nothing said the kernel was the reason. This
    says it once, where an operator is already reading.

    Reported as a pass when nothing is confinable *and* nothing needs to be:
    an install with no external plugins is not broken by a kernel that cannot
    sandbox them.
    """
    from .plugins.seccomp import confinement_error  # noqa: PLC0415 - probe only

    reason = confinement_error()
    if not reason:
        return Check("confinement", True, "plugin workers can be confined on this host")
    return Check(
        "confinement",
        False,
        f"{reason}; external plugin workers will not start, and --no-seccomp is "
        "the deliberate override",
    )


def check_applet_contract(applet_root: Path, snapshot_path: Path) -> Check:
    """The installed applet must be able to read what this service publishes.

    The applet validates a snapshot as a closed record, so a field this service
    added and the installed payload does not know invalidates the *whole*
    document — and the applet reports that as "no runtime service is publishing
    state", which is indistinguishable from a service that has stopped.

    Optional fields do not prevent it and neither does landing the applet
    change first: what matters is the order the two are *deployed*, and nothing
    stated that anywhere.  This states it, against the snapshot actually on
    disk rather than against a fixture, so the mismatch is named at install
    time instead of appearing later as an outage.
    """
    import jsonschema  # noqa: PLC0415 - only needed for this probe

    schema_file = Path(applet_root) / "runtime-snapshot.schema.json"
    if not schema_file.is_file():
        return Check("applet-contract", False, f"applet ships no snapshot contract: {schema_file}")
    try:
        schema = json.loads(schema_file.read_text(encoding="utf-8"))
        snapshot = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return Check("applet-contract", False, f"could not compare contracts: {error}")
    violations = [
        f"{'/'.join(str(part) for part in error.path)}: {error.message}"
        for error in jsonschema.Draft202012Validator(schema).iter_errors(snapshot)
    ]
    if violations:
        return Check(
            "applet-contract",
            False,
            "the installed applet cannot read the snapshot this service publishes "
            f"({violations[0]}); install the applet built from these contracts",
        )
    return Check(
        "applet-contract",
        True,
        "the installed applet reads the snapshot this service publishes",
    )


# Every accelerator lane and the runtime import that proves it can execute.
# There is no CPU entry by design: a CPU-only runtime is not a backend here.
BACKEND_RUNTIMES = (
    ("gpu", "ncnn", "install the [gpu] extra"),
    ("gpu", "onnxruntime", "install the [gpu-onnx-cuda] or [gpu-onnx-rocm] extra"),
    ("npu", "openvino", "install the [npu] extra"),
    ("tpu", "tflite_runtime", "install the [tpu] extra"),
)


def check_backends(runtimes: Sequence[tuple[str, str, str]] = BACKEND_RUNTIMES) -> Check:
    """At least one accelerator runtime must be usable — not merely present.

    Without this the whole report can pass while the service can execute
    nothing: it starts, publishes a contract-valid snapshot, answers the bus,
    and refuses every job because no runtime resolves.  That is the most
    misleading state this install has, and it was reached by the documented
    install command, which omitted the accelerator extra.

    Importability was the wrong question.  A plain ``onnxruntime`` wheel ships
    ``CPUExecutionProvider`` only, and ``executors/gpu.py`` refuses it by
    design — there is no CPU backend here — so this check reported a GPU lane
    as available while dispatch answered ``runtime-unusable`` for the same
    install.  A gate telling an operator the opposite of what the runtime does
    is worse than no gate.

    So it asks the executors.  Each one already knows whether it can run, and
    says why when it cannot; a device that is simply absent is reported
    separately, because an install can be correct on a machine with no
    accelerator plugged in.
    """
    import importlib.util  # noqa: PLC0415 - only needed for this probe

    usable: list[str] = []
    present: list[str] = []
    reasons: list[str] = []
    for backend, module, _remedy in runtimes:
        try:
            if importlib.util.find_spec(module) is None:
                continue
        except (ImportError, ValueError):
            # A broken or half-removed distribution is not an installed one.
            continue
        present.append(f"{backend}:{module}")
        verdict = _runtime_verdict(module)
        if verdict is None:
            usable.append(f"{backend}:{module}")
        else:
            reasons.append(f"{backend}:{module} ({verdict})")
    if usable:
        return Check("backends", True, f"accelerator runtimes usable: {', '.join(usable)}")
    if present:
        # Installed and refused is a different problem from not installed, and
        # the executor's own words are the remedy.
        return Check(
            "backends",
            False,
            "no installed accelerator runtime is usable: " + "; ".join(reasons),
        )
    remedies = sorted({remedy for _backend, _module, remedy in runtimes})
    return Check(
        "backends",
        False,
        "no accelerator runtime is importable, so every profile is unavailable; "
        + "; ".join(remedies),
    )


def _executor_for(module: str):
    """The executor that would actually run this runtime.

    Asked per module rather than per backend: the GPU lane is a composite of
    Vulkan and ONNX Runtime, so asking it reports available whenever *either*
    works — which is exactly how a CPU-only ``onnxruntime`` came to be listed
    as a usable GPU runtime.
    """
    from .executors.gpu import GpuExecutor  # noqa: PLC0415 - probe-only imports
    from .executors.npu import NpuExecutor  # noqa: PLC0415
    from .executors.tpu import TpuExecutor  # noqa: PLC0415
    from .executors.vulkan import VulkanGpuExecutor  # noqa: PLC0415

    builders = {
        "ncnn": VulkanGpuExecutor,
        "onnxruntime": GpuExecutor,
        "openvino": NpuExecutor,
        "tflite_runtime": TpuExecutor,
    }
    builder = builders.get(module)
    return None if builder is None else builder(True)


def _runtime_verdict(module: str) -> str | None:
    """Why this runtime cannot be used, or ``None`` when it can.

    A device that is not plugged in is not an installation fault, so an absent
    device reports as usable: the runtime is correctly installed and waiting
    for hardware.  Anything else — a driver that will not load, a provider the
    executor refuses — is the install being wrong, and the executor's own
    sentence is the remedy.
    """
    from .executors.base import DEVICE_ABSENT  # noqa: PLC0415 - probe-only import

    executor = _executor_for(module)
    if executor is None:
        return None
    try:
        availability = executor.availability()
    except Exception as error:  # noqa: BLE001 - a probe must not fail the report
        return type(error).__name__
    if availability.available or availability.code == DEVICE_ABSENT:
        return None
    return availability.reason


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
        check_confinement(),
    ]
    # Run against whatever applet is actually installed, because the failure
    # this catches only exists on a host where both halves are present. A
    # service-only install skips it rather than failing for an applet nobody
    # asked for.
    root = Path(applet_root) if applet_root is not None else Path(DEFAULT_APPLET_ROOT).expanduser()
    if root.is_dir():
        checks.append(check_applet_contract(root, snapshot_path))
        if applet_checksums is not None:
            checks.append(check_applet(root, load_applet_checksums(applet_checksums)))
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
                "~/.local/state/xpu-workload-manager/state.json",
            )
        ).expanduser(),
    )
    parser.add_argument(
        "--applet-root",
        type=Path,
        default=None,
        help=f"installed applet to check against (default {DEFAULT_APPLET_ROOT})",
    )
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
