"""Production composition root for the public OmniTensor service entry point."""

from __future__ import annotations

import os
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from . import paths
from .artifact_readiness import ArtifactResolver, plugin_accelerator_devices
from .control import ControlService
from .device_registry import DeviceRegistry
from .plugin_admission import DEFAULT_MAX_CONCURRENT, MAX_CONCURRENT_LIMIT
from .plugins.artifact_installation import ArtifactInstaller
from .plugins.grants import GrantLedger
from .plugins.job_results import JobResultStore
from .plugins.kernel_telemetry import UnixSocketAggregateSource
from .plugins.loading import InstalledPluginRuntime
from .plugins.settings import PluginSettingsStore
from .plugins.summaries import ResultSummaryRegistry
from .plugins.telemetry import PluginTelemetryRegistry
from .registry import bundled_workloads_path, load_workload_catalog
from .snapshot import MAX_PUBLISHED_INPUT_ROOTS
from .state import PolicyStore

DEFAULT_STATE_PATH = paths.SNAPSHOT_PATH
DEFAULT_POLICY_PATH = paths.POLICY_PATH
DEFAULT_WORKLOADS_PATH = paths.WORKLOADS_ROOT
DEFAULT_MODEL_BINDINGS_PATH = paths.MODEL_BINDINGS_ROOT
DEFAULT_ARTIFACT_ROOT = paths.ARTIFACT_ROOT
DEFAULT_GRANTS_PATH = paths.GRANTS_PATH


def _env_path(
    name: str,
    fallback: str,
    environ: Mapping[str, str] | None = None,
) -> Path:
    source = os.environ if environ is None else environ
    return Path(source.get(name, fallback)).expanduser()


def _env_paths(
    name: str,
    environ: Mapping[str, str] | None = None,
) -> tuple[Path, ...]:
    """Read colon-separated roots a caller may reference inputs from."""
    source = os.environ if environ is None else environ
    raw = source.get(name, "")
    return tuple(Path(part).expanduser() for part in raw.split(os.pathsep) if part)


def _env_input_roots(environ: Mapping[str, str] | None = None) -> tuple[Path, ...]:
    return _env_published_roots("OMNITENSOR_INPUT_ROOTS", environ)


def _env_selected_file_roots(environ: Mapping[str, str] | None = None) -> tuple[Path, ...]:
    """Where this unit can read a file a person selected.

    Configured rather than detected: `PrivateTmp=true` and
    `ProtectHome=read-only` are how the unit is run, and a service cannot
    enumerate what its own sandbox will let it open. Unset means it reads no
    selected file at all, which is what a client is then told.
    """
    return _env_published_roots("OMNITENSOR_SELECTED_FILE_ROOTS", environ)


def _env_published_roots(
    name: str,
    environ: Mapping[str, str] | None = None,
) -> tuple[Path, ...]:
    roots = tuple(dict.fromkeys(_env_paths(name, environ)))
    if len(roots) > MAX_PUBLISHED_INPUT_ROOTS:
        # A root the snapshot cannot name is one no client can use, so this is
        # a configuration error said at startup rather than half-honoured.
        raise ValueError(
            f"{name} contains {len(roots)} unique roots; "
            f"at most {MAX_PUBLISHED_INPUT_ROOTS} are supported"
        )
    return roots


def _env_accelerator_device_ids(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    source = os.environ if environ is None else environ
    names = {
        "gpu": "OMNITENSOR_GPU_DEVICE",
        "npu": "OMNITENSOR_NPU_DEVICE",
        "tpu": "OMNITENSOR_TPU_DEVICE",
    }
    return {
        backend: value for backend, name in names.items() if (value := source.get(name, "").strip())
    }


def _env_plugin_slots(environ: Mapping[str, str] | None = None) -> int:
    """How many plugin jobs may run at once, from the environment.

    An unset or unreadable value is the default rather than a startup failure:
    the pool is a tuning knob, and refusing to start over a typo in it would
    take the whole runtime down for a number that has a sane answer.
    """
    source = os.environ if environ is None else environ
    raw = source.get("OMNITENSOR_PLUGIN_SLOTS", "").strip()
    if not raw.isdigit():
        return DEFAULT_MAX_CONCURRENT
    return max(1, min(MAX_CONCURRENT_LIMIT, int(raw)))


@dataclass(frozen=True)
class ServiceEnvironment:
    """Validated path/device values at the process environment boundary."""

    snapshot_path: Path
    policy_path: Path
    workloads_path: Path
    artifact_root: Path
    model_bindings_path: Path
    grants_path: Path
    input_roots: tuple[Path, ...]
    selected_file_roots: tuple[Path, ...]
    accelerator_device_ids: dict[str, str]
    plugin_slots: int

    @classmethod
    def read(cls, environ: Mapping[str, str] | None = None) -> ServiceEnvironment:
        return cls(
            snapshot_path=_env_path("OMNITENSOR_STATE_PATH", DEFAULT_STATE_PATH, environ),
            policy_path=_env_path("OMNITENSOR_POLICY_PATH", DEFAULT_POLICY_PATH, environ),
            workloads_path=_env_path("OMNITENSOR_WORKLOADS", DEFAULT_WORKLOADS_PATH, environ),
            artifact_root=_env_path("OMNITENSOR_ARTIFACT_ROOT", DEFAULT_ARTIFACT_ROOT, environ),
            model_bindings_path=_env_path(
                "OMNITENSOR_MODEL_BINDINGS", DEFAULT_MODEL_BINDINGS_PATH, environ
            ),
            grants_path=_env_path("OMNITENSOR_GRANTS_PATH", DEFAULT_GRANTS_PATH, environ),
            input_roots=_env_input_roots(environ),
            selected_file_roots=_env_selected_file_roots(environ),
            accelerator_device_ids=_env_accelerator_device_ids(environ),
            plugin_slots=_env_plugin_slots(environ),
        )

    def service_options(self) -> dict[str, Any]:
        return {
            "snapshot_path": self.snapshot_path,
            "policy_path": self.policy_path,
            "workloads_path": self.workloads_path,
            "artifact_root": self.artifact_root,
            "model_bindings_path": self.model_bindings_path,
            "grants_path": self.grants_path,
            "input_roots": self.input_roots,
            "selected_file_roots": self.selected_file_roots,
            "accelerator_device_ids": self.accelerator_device_ids,
            "plugin_slots": self.plugin_slots,
        }


def _alert_id() -> str:
    return f"alert-{secrets.token_hex(16)}"


def build_adapters(options: Mapping[str, Any]) -> dict[str, Any]:
    """The production adapters the runtime used to construct for itself.

    Each of these is chosen, not derived: a different composition root may
    hold grants in memory, drop result summaries, or aggregate kernel
    telemetry over something other than a unix socket. The runtime only needs
    the port, so the choice belongs here and the runtime takes what it is
    given.
    """
    snapshot_path = Path(options["snapshot_path"])
    # The workload catalog and the artifact resolver need only the paths and
    # each other, so they are built here rather than by a constructor reading
    # its own arguments. The resolver is told where plugins come from by
    # whoever owns the plugin runtime, which is later than this.
    grants = GrantLedger(options["grants_path"])
    workloads = load_workload_catalog(
        options["workloads_path"],
        model_bindings_root=options.get("model_bindings_path"),
    )
    artifact_root = options.get("artifact_root")
    artifacts = ArtifactResolver(
        artifact_root,
        ArtifactInstaller(artifact_root) if artifact_root else None,
        workloads_of=lambda: workloads,
    )
    settings = PluginSettingsStore(snapshot_path.parent / "plugin-settings")
    devices = DeviceRegistry()
    # The two-way dependency the service used to hold inside its constructor,
    # broken where a composition root is allowed to break it: `control` is a
    # local this function assigns four lines down, and every closure below is
    # called long after that — when a worker is built, not while one is.
    control: ControlService | None = None
    plugin_runtime = InstalledPluginRuntime(
        bundled_workloads_path(),
        grant_source=grants,
        selected_files_root=snapshot_path.parent / "plugin-inputs",
        worker_state_root=snapshot_path.parent / "plugin-state",
        resolve_artifact=artifacts.resolve_plugin,
        accelerator_devices=lambda profile_id=None: plugin_accelerator_devices(
            devices.devices,
            None if profile_id is None else control.state.device_choices.get(profile_id),
        ),
        profile_accelerator_devices=lambda profile_id=None: plugin_accelerator_devices(
            devices.devices,
            None if profile_id is None else control.state.device_choices.get(profile_id),
        ),
        profile_model_choice=lambda profile_id: control.state.model_choices.get(profile_id),
        settings_store=settings,
    )
    artifacts.reads_plugins_from(lambda: plugin_runtime.snapshot.catalog.plugins)
    control = ControlService(
        PolicyStore(
            options["policy_path"],
            {
                workload_id: workload.default_policy()
                for workload_id, workload in workloads.items()
            },
        ),
        profile_exists=lambda profile_id: (
            profile_id in workloads or profile_id in plugin_runtime.plugin_ids()
        ),
        gpu_device_ids=devices.gpu_ids,
        profile_models=plugin_runtime.declared_artifacts,
        profile_configuration=plugin_runtime.configuration_spec,
        settings_store=settings,
    )
    return {
        "workloads": workloads,
        "devices": devices,
        "artifacts": artifacts,
        "plugin_settings": settings,
        "plugin_runtime": plugin_runtime,
        "control": control,
        "grants": grants,
        "job_results": JobResultStore(),
        "plugin_telemetry": PluginTelemetryRegistry(),
        "result_summaries": ResultSummaryRegistry(alert_id_factory=_alert_id),
        "kernel_telemetry_source": UnixSocketAggregateSource(),
        "cancellation_journal_path": snapshot_path.parent / "cancellations.json",
    }


def build_service(
    service_factory: Callable[..., object] | None = None,
    **options: Any,
):
    """Construct the production adapters, then the runtime that uses them.

    Callers pass the path/device options :meth:`ServiceEnvironment.service_options`
    produces; anything else given here overrides an adapter this root would
    otherwise choose, which is how a test substitutes one without reaching
    into the constructor's ``or ...`` defaults.
    """
    if service_factory is None:
        service_factory = import_module(".service", __package__).OmniTensorService
    adapters = build_adapters(options)
    adapters.update(options)
    return service_factory(**adapters)


def build_service_from_env(
    service_factory: Callable[..., object] | None = None,
    environ: Mapping[str, str] | None = None,
):
    """Build the production service while allowing a factory-boundary test."""
    return build_service(service_factory, **ServiceEnvironment.read(environ).service_options())
