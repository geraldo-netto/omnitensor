"""Deterministic checks used by installation acceptance."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import jsonschema

from .acceptance_contracts import (
    BusProbe,
    Check,
    InstallationReport,
    ServiceProbe,
    legacy_acceptance_value,
)
from .acceptance_probes import _runtime_verdict
from .registry import (
    ManifestError,
    _schema_path,
    bundled_workloads_path,
    load_schema,
    load_workloads,
    validate_document,
)

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
    "training-report.schema.json",
    "numeric-training-report.schema.json",
    "document-model-report.schema.json",
    "low-light-configuration.schema.json",
    "plugin-inventory.schema.json",
    "runtime-refusal.schema.json",
    "runtime-job-result-request.schema.json",
    "runtime-job-result.schema.json",
    "media-transcription-result.schema.json",
    "event-extraction-result.schema.json",
    "document-question-answer.schema.json",
    "document-question-acceptance.schema.json",
    "document-question-result.schema.json",
    "selected-text-acceptance.schema.json",
    "selected-text-answer.schema.json",
    "selected-text-result.schema.json",
    "file-organizer-answer.schema.json",
    "file-organizer-result.schema.json",
    "collected-output.schema.json",
)
DEFAULT_SNAPSHOT_MAX_AGE_MS = 30_000
FUTURE_TOLERANCE_MS = 2_000

BACKEND_RUNTIMES = (
    ("gpu", "ncnn", "install the [gpu] extra"),
    ("gpu", "onnxruntime", "install the [gpu-onnx-cuda] or [gpu-onnx-rocm] extra"),
    ("npu", "openvino", "install the [npu] extra"),
    ("tpu", "tflite_runtime", "install the [tpu] extra"),
)


def check_executable(
    name: str = "omnitensor",
    *,
    resolver: Callable[[str], str | None] = shutil.which,
) -> Check:
    resolved = resolver(name)
    if not resolved:
        return Check("executable", False, f"{name} is not on PATH")
    return Check("executable", True, f"{name} resolves to {resolved}")


def check_service(probe: ServiceProbe) -> Check:
    running, detail = probe.unit_state()
    return Check("service", running, detail)


def check_schemas(names: Sequence[str] = REQUIRED_SCHEMAS) -> Check:
    missing: list[str] = []
    invalid: list[str] = []
    schema_path = legacy_acceptance_value("_schema_path", _schema_path)
    schema_loader = legacy_acceptance_value("load_schema", load_schema)
    for name in names:
        try:
            schema_path(name)
        except FileNotFoundError:
            missing.append(name)
            continue
        try:
            jsonschema.Draft202012Validator.check_schema(schema_loader(name))
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
    try:
        workload_root = legacy_acceptance_value(
            "bundled_workloads_path", bundled_workloads_path
        )
        resolved = root or workload_root()
    except FileNotFoundError as error:
        return Check("workloads", False, str(error))
    try:
        loader = legacy_acceptance_value("load_workloads", load_workloads)
        workloads = loader(resolved)
    except ManifestError as error:
        return Check("workloads", False, f"bundled catalog is invalid: {error}")
    if not workloads:
        return Check("workloads", False, f"no bundled workloads under {resolved}")
    return Check("workloads", True, f"{len(workloads)} bundled workloads loaded")


def check_plugin_discovery(discover: Callable[[], Iterable]) -> Check:
    try:
        candidates = tuple(discover())
    except Exception as error:  # noqa: BLE001 - any discovery failure is a failed install
        return Check("discovery", False, f"discovery raised {type(error).__name__}")
    return Check("discovery", True, f"{len(candidates)} plugin candidates discovered")


def check_isolation(specs: Iterable) -> Check:
    external = tuple(specs)
    unsandboxed = [
        spec.plugin_id for spec in external if getattr(spec, "sandbox", None) is None
    ]
    if unsandboxed:
        return Check(
            "isolation",
            False,
            f"workers without a sandbox: {', '.join(sorted(unsandboxed))}",
        )
    return Check("isolation", True, f"{len(external)} external workers are sandboxed")


def check_bus(probe: BusProbe) -> Check:
    try:
        reply = probe.apply_command("not-json")
    except Exception as error:  # noqa: BLE001 - transport failures are arbitrary
        return Check("dbus", False, f"ApplyCommand raised {type(error).__name__}")
    try:
        acknowledgement = json.loads(reply)
    except ValueError:
        return Check("dbus", False, "ApplyCommand did not return JSON")
    validator = legacy_acceptance_value("validate_document", validate_document)
    violations = validator("runtime-acknowledgement.schema.json", acknowledgement)
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
    try:
        document = json.loads(Path(path).read_bytes())
    except FileNotFoundError:
        return Check("snapshot", False, f"no snapshot at {path}")
    except (OSError, ValueError) as error:
        return Check("snapshot", False, f"snapshot is unreadable: {error}")
    validator = legacy_acceptance_value("validate_document", validate_document)
    violations = validator("runtime-snapshot.schema.json", document)
    if violations:
        return Check("snapshot", False, f"snapshot violates contract: {violations[0]}")
    generated_at = document["generatedAt"]
    if generated_at - now_ms > FUTURE_TOLERANCE_MS:
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
    from omnitensor.preparation import file_digest  # noqa: PLC0415 - acceptance-only import

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
        if file_digest(installed) != digest:
            altered.append(relative)
    if missing or altered:
        parts = []
        if missing:
            parts.append(f"missing {len(missing)} file(s): {missing[0]}")
        if altered:
            parts.append(f"altered {len(altered)} file(s): {altered[0]}")
        return Check("applet", False, "; ".join(parts))
    return Check("applet", True, f"{len(checksums)} applet files match their checksums")


def check_confinement() -> Check:
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


def check_backends(runtimes: Sequence[tuple[str, str, str]] = BACKEND_RUNTIMES) -> Check:
    import importlib.util  # noqa: PLC0415 - only needed for this probe

    usable: list[str] = []
    present: list[str] = []
    reasons: list[str] = []
    verdict_for = legacy_acceptance_value("_runtime_verdict", _runtime_verdict)
    for backend, module, _remedy in runtimes:
        try:
            if importlib.util.find_spec(module) is None:
                continue
        except (ImportError, ValueError):
            continue
        present.append(f"{backend}:{module}")
        verdict = verdict_for(module)
        if verdict is None:
            usable.append(f"{backend}:{module}")
        else:
            reasons.append(f"{backend}:{module} ({verdict})")
    if usable:
        return Check("backends", True, f"accelerator runtimes usable: {', '.join(usable)}")
    if present:
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


def verify_installation(checks: Sequence[Check]) -> InstallationReport:
    return InstallationReport(tuple(checks))


for _legacy_function in (
    check_executable,
    check_service,
    check_schemas,
    check_workload_catalog,
    check_plugin_discovery,
    check_isolation,
    check_bus,
    check_snapshot,
    check_applet,
    check_confinement,
    check_applet_contract,
    check_backends,
    verify_installation,
):
    _legacy_function.__module__ = "omnitensor.acceptance"
