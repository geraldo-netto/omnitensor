"""Deterministic checks used by installation acceptance."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import jsonschema

from .acceptance_contracts import (
    Check,
    ControlProbe,
    InstallationReport,
    ServiceProbe,
)
from .acceptance_probes import DEVICE_MISSING, UNUSABLE, USABLE, runtime_state
from .registry import (
    ManifestError,
    _schema_path,
    bundled_workloads_path,
    load_schema,
    load_workloads,
    validate_document,
)

REQUIRED_SCHEMAS = (
    "control-request.schema.json",
    "control-reply.schema.json",
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
    "document-translation-answer.schema.json",
    "document-translation-result.schema.json",
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
    schema_path = _schema_path
    schema_loader = load_schema
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
        workload_root = bundled_workloads_path
        resolved = root or workload_root()
    except FileNotFoundError as error:
        return Check("workloads", False, str(error))
    try:
        loader = load_workloads
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


def check_provider_imports(
    entry_points_provider: Callable[..., Iterable] | None = None,
) -> Check:
    """Load every installed provider entry point, and name what a skew broke.

    Provider distributions import module-level names from the service package,
    so reinstalling ``omnitensor`` without the providers built from the same
    tree leaves an installed wheel importing a name that no longer exists.
    Removing ``MAX_VISUALS`` and reinstalling only the root package left
    ``media-transcription`` dead at entry-point load, and the applet was told
    only ``worker handshake rejected: truncated-frame``.

    Discovery cannot catch this: it enumerates entry points deterministically
    and never calls ``load()``, which is the point at which the import happens.
    """
    from importlib import metadata  # noqa: PLC0415 - only needed for this probe

    from .plugins.discovery import PLUGIN_ENTRY_POINT_GROUP  # noqa: PLC0415

    provider = entry_points_provider or metadata.entry_points
    try:
        entry_points = tuple(provider(group=PLUGIN_ENTRY_POINT_GROUP))
    except Exception as error:  # noqa: BLE001 - a broken installation, not a crash here
        return Check(
            "provider-imports",
            False,
            f"entry-point enumeration failed: {type(error).__name__}: {error}",
        )
    broken: list[str] = []
    loaded = 0
    for entry_point in sorted(entry_points, key=lambda point: point.name):
        try:
            entry_point.load()
        except Exception as error:  # noqa: BLE001 - any failure to load is a failed install
            broken.append(
                f"{entry_point.name} from {_distribution_label(entry_point)}: "
                f"{type(error).__name__}: {error}"
            )
        else:
            loaded += 1
    if broken:
        return Check(
            "provider-imports",
            False,
            "installed providers that cannot be loaded — reinstall them from the same "
            "tree as the service: " + "; ".join(broken),
        )
    return Check("provider-imports", True, f"{loaded} installed providers import cleanly")


def _distribution_label(entry_point: object) -> str:
    """The distribution to reinstall, or the entry point's target when unknown."""
    distribution = getattr(entry_point, "dist", None)
    name = getattr(distribution, "name", None)
    version = getattr(distribution, "version", None)
    if name and version:
        return f"{name} {version}"
    return name or str(getattr(entry_point, "value", "an unknown distribution"))


def check_isolation(specs: Iterable) -> Check:
    external = tuple(specs)
    unsandboxed = [spec.plugin_id for spec in external if getattr(spec, "sandbox", None) is None]
    if unsandboxed:
        return Check(
            "isolation",
            False,
            f"workers without a sandbox: {', '.join(sorted(unsandboxed))}",
        )
    return Check("isolation", True, f"{len(external)} external workers are sandboxed")


def _transport_failure(error: BaseException) -> str:
    """Name the failure the operator can act on, not the deadline that hid it.

    The probe retries for its whole budget, so a service that never claimed the
    bus name surfaces as ``TimeoutError`` — true about the probe and useless
    about the cause. ``asyncio.wait_for`` also chains its own cancellation,
    which describes nothing.
    """
    import asyncio  # noqa: PLC0415

    cause = error.__cause__
    repeated = cause is not None and type(cause) is type(error) and str(cause) == str(error)
    if cause is None or repeated or isinstance(cause, asyncio.CancelledError):
        reported = cause if repeated else error
        detail = str(reported).strip()
        return f"{type(reported).__name__}: {detail}" if detail else type(reported).__name__
    detail = str(cause).strip()
    named = f"{type(error).__name__} after {type(cause).__name__}"
    return f"{named}: {detail}" if detail else named


def check_bus(probe: ControlProbe) -> Check:
    try:
        reply = probe.apply_command("not-json")
    except Exception as error:  # noqa: BLE001 - transport failures are arbitrary
        return Check("control", False, f"apply-command raised {_transport_failure(error)}")
    try:
        acknowledgement = json.loads(reply)
    except ValueError:
        return Check("control", False, "apply-command did not return JSON")
    validator = validate_document
    violations = validator("runtime-acknowledgement.schema.json", acknowledgement)
    if violations:
        return Check(
            "control",
            False,
            f"acknowledgement violates contract: {'; '.join(sorted(violations))}",
        )
    if acknowledgement.get("status") != "rejected":
        return Check("control", False, "invalid command was not rejected")
    return Check(
        "control", True, f"apply-command answered at revision {acknowledgement['revision']}"
    )


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
    validator = validate_document
    violations = validator("runtime-snapshot.schema.json", document)
    if violations:
        return Check(
            "snapshot",
            False,
            f"snapshot violates contract: {'; '.join(sorted(violations))}",
        )
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
            parts.append(f"missing {len(missing)} file(s): {', '.join(sorted(missing))}")
        if altered:
            parts.append(f"altered {len(altered)} file(s): {', '.join(sorted(altered))}")
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


# The fields the Cinnamon helper actually reads out of the snapshot. It ships
# no schema of its own any more — it draws an icon and five lines, and the
# client is what validates the document against the canonical schemas — so what
# has to hold between service and panel is that these fields are there and that
# both sides name the same file.
HELPER_SNAPSHOT_FIELDS = ("version", "generatedAt", "devices", "metrics", "alerts")


def _helper_state_path(applet_root: Path) -> str | None:
    """The path the installed helper reads, from its own settings schema."""
    try:
        settings = json.loads(
            (Path(applet_root) / "settings-schema.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None
    entry = settings.get("runtime-state-path")
    return entry.get("default") if isinstance(entry, dict) else None


def check_applet_contract(applet_root: Path, snapshot_path: Path) -> Check:
    """Whether the installed panel can read what this service publishes.

    Two shapes are accepted, because both have shipped. An applet that carries
    a mirrored `runtime-snapshot.schema.json` is validated against it, exactly
    as before: a field this service adds and that applet does not know
    invalidates the whole document, and the desktop reads as dead.

    The current helper carries no mirror. It reads a handful of fields to
    colour an icon, so its half of the contract is those fields being present
    and both sides naming the same file — a helper pointed at a path nobody
    writes reports "the runtime is not running", which looks exactly like a
    service that has stopped.
    """
    root = Path(applet_root)
    try:
        snapshot = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return Check("applet-contract", False, f"could not compare contracts: {error}")

    schema_file = root / "runtime-snapshot.schema.json"
    if schema_file.is_file():
        try:
            schema = json.loads(schema_file.read_text(encoding="utf-8"))
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
                f"({'; '.join(sorted(violations))}); "
                "install the applet built from these contracts",
            )
        return Check(
            "applet-contract",
            True,
            "the installed applet reads the snapshot this service publishes",
        )

    if not (root / "applet.js").is_file():
        return Check("applet-contract", False, f"no applet is installed at {root}")
    missing = [field for field in HELPER_SNAPSHOT_FIELDS if field not in snapshot]
    if missing:
        return Check(
            "applet-contract",
            False,
            f"the published snapshot omits what the panel reads: {', '.join(missing)}",
        )
    configured = _helper_state_path(root)
    published = str(Path(snapshot_path).expanduser())
    if configured is not None and str(Path(configured).expanduser()) != published:
        return Check(
            "applet-contract",
            False,
            f"the panel reads {configured} but this service publishes {published}; "
            "set the same path on both sides",
        )
    return Check(
        "applet-contract",
        True,
        "the installed panel reads the snapshot this service publishes",
    )


def _grouped_runtime_states(
    runtimes: Sequence[tuple[str, str, str]], find_spec
) -> dict[str, list[str]]:
    """Every importable runtime, labelled by what it can actually do."""
    grouped: dict[str, list[str]] = {USABLE: [], DEVICE_MISSING: [], UNUSABLE: []}
    for backend, module, _remedy in runtimes:
        try:
            if find_spec(module) is None:
                continue
        except (ImportError, ValueError):
            continue
        state, reason = runtime_state(module)
        label = f"{backend}:{module}" if state == USABLE else f"{backend}:{module} ({reason})"
        grouped[state].append(label)
    return grouped


def check_backends(runtimes: Sequence[tuple[str, str, str]] = BACKEND_RUNTIMES) -> Check:
    import importlib.util  # noqa: PLC0415 - only needed for this probe

    grouped = _grouped_runtime_states(runtimes, importlib.util.find_spec)
    usable = grouped[USABLE]
    waiting = grouped[DEVICE_MISSING]
    reasons = grouped[UNUSABLE]
    present = usable + waiting + reasons
    if usable or waiting:
        # An install waiting for hardware passes, but says so: reporting it as
        # "usable" promised a lane that refuses the first job it is given.
        parts = []
        if usable:
            parts.append(f"accelerator runtimes usable: {', '.join(usable)}")
        if waiting:
            parts.append(f"installed but waiting for hardware: {', '.join(waiting)}")
        return Check("backends", True, "; ".join(parts))
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
