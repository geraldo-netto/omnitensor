"""Production composition root for the public OmniTensor service entry point."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from . import paths
from .plugin_admission import DEFAULT_MAX_CONCURRENT, MAX_CONCURRENT_LIMIT
from .snapshot import MAX_PUBLISHED_INPUT_ROOTS

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
    roots = tuple(dict.fromkeys(_env_paths("OMNITENSOR_INPUT_ROOTS", environ)))
    if len(roots) > MAX_PUBLISHED_INPUT_ROOTS:
        raise ValueError(
            f"OMNITENSOR_INPUT_ROOTS contains {len(roots)} unique roots; "
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
            "accelerator_device_ids": self.accelerator_device_ids,
            "plugin_slots": self.plugin_slots,
        }


def build_service_from_env(
    service_factory: Callable[..., object] | None = None,
    environ: Mapping[str, str] | None = None,
):
    """Build the production service while allowing a factory-boundary test."""
    if service_factory is None:
        service_factory = import_module(".service", __package__).OmniTensorService
    return service_factory(**ServiceEnvironment.read(environ).service_options())
