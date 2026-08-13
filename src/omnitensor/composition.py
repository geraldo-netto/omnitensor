"""Production composition root for the public OmniTensor service entry point."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_STATE_PATH = "~/.local/state/xpu-workload-manager/state.json"
DEFAULT_POLICY_PATH = "~/.local/state/omnitensor/policy.json"
DEFAULT_WORKLOADS_PATH = "~/.local/share/omnitensor/workloads"
DEFAULT_MODEL_BINDINGS_PATH = "~/.local/share/omnitensor/model-bindings"
DEFAULT_ARTIFACT_ROOT = "~/.local/share/omnitensor/artifacts"
DEFAULT_GRANTS_PATH = "~/.local/state/omnitensor/grants.json"


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
        backend: value
        for backend, name in names.items()
        if (value := source.get(name, "").strip())
    }


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
            input_roots=_env_paths("OMNITENSOR_INPUT_ROOTS", environ),
            accelerator_device_ids=_env_accelerator_device_ids(environ),
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
        }


def build_service_from_env(
    service_factory: Callable[..., object] | None = None,
    environ: Mapping[str, str] | None = None,
):
    """Build the production service while allowing a factory-boundary test."""
    if service_factory is None:
        from .service import OmniTensorService

        service_factory = OmniTensorService
    return service_factory(**ServiceEnvironment.read(environ).service_options())
