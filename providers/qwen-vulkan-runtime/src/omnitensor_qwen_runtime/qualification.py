"""Fail-closed binding between accepted tasks, native bytes, models, and GPU."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.resources
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from omnitensor.plugins.generation import GenerationTask
from omnitensor.preparation import file_digest

_DIGEST = re.compile(r"^[a-f0-9]{64}$")
_MAX_RECEIPT_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class Qualification:
    device: str
    model_layers: int
    runtime_version: str
    runtime_binaries: tuple[tuple[str, str], ...]


def task_sha256(task: GenerationTask) -> str:
    raw = json.dumps(asdict(task), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def load_qualification(
    plugin_id: str,
    model_id: str,
    model_sha256: str,
    task: GenerationTask,
) -> Qualification:
    resource = importlib.resources.files("omnitensor_qwen_runtime").joinpath("qualification.json")
    raw = resource.read_bytes()
    if len(raw) > _MAX_RECEIPT_BYTES:
        raise RuntimeError("Qwen qualification receipt is oversized")
    try:
        document = json.loads(raw)
    except (UnicodeError, ValueError) as error:
        raise RuntimeError("Qwen qualification receipt is invalid") from error
    if not isinstance(document, dict) or set(document) != {
        "version",
        "recordedAt",
        "device",
        "runtime",
        "models",
        "workloads",
    }:
        raise RuntimeError("Qwen qualification receipt fields are invalid")
    if document["version"] != 1 or document["recordedAt"] != "2026-08-13":
        raise RuntimeError("Qwen qualification receipt version is invalid")
    device = document["device"]
    if not isinstance(device, str) or not device:
        raise RuntimeError("Qwen qualification device is invalid")
    runtime = _runtime(document["runtime"])
    model_layers = _model(document["models"], model_id, model_sha256)
    _workload(document["workloads"], plugin_id, model_id, task_sha256(task))
    return Qualification(device, model_layers, runtime[0], runtime[1])


def verify_native_runtime(qualification: Qualification) -> None:
    try:
        version = importlib.metadata.version("llama-cpp-python")
        package = importlib.resources.files("llama_cpp")
    except (importlib.metadata.PackageNotFoundError, ModuleNotFoundError) as error:
        raise RuntimeError("qualified llama.cpp runtime is not installed") from error
    if version != qualification.runtime_version:
        raise RuntimeError("llama.cpp runtime version differs from qualification")
    for name, digest in qualification.runtime_binaries:
        path = Path(str(package.joinpath("lib", name)))
        if not path.is_file() or file_digest(path) != digest:
            raise RuntimeError("llama.cpp native bytes differ from qualification")


def _runtime(value: object) -> tuple[str, tuple[tuple[str, str], ...]]:
    if not isinstance(value, dict) or set(value) != {
        "distribution",
        "version",
        "wheelSha256",
        "binaries",
    }:
        raise RuntimeError("Qwen qualification runtime is invalid")
    if value["distribution"] != "llama-cpp-python" or value["version"] != "0.3.34":
        raise RuntimeError("Qwen qualification runtime identity is invalid")
    if not _is_digest(value["wheelSha256"]):
        raise RuntimeError("Qwen qualification wheel digest is invalid")
    binaries = value["binaries"]
    if not isinstance(binaries, dict) or set(binaries) != {
        "libggml-vulkan.so",
        "libllama.so",
    }:
        raise RuntimeError("Qwen qualification native inventory is invalid")
    if any(not _is_digest(digest) for digest in binaries.values()):
        raise RuntimeError("Qwen qualification native digest is invalid")
    return str(value["version"]), tuple(sorted(binaries.items()))


def _model(value: object, model_id: str, digest: str) -> int:
    if not isinstance(value, dict) or model_id not in value:
        raise RuntimeError("Qwen model has no qualification")
    model = value[model_id]
    if not isinstance(model, dict) or set(model) != {
        "sha256",
        "fullyOffloadedLayers",
        "contextTokens",
        "keyCache",
        "valueCache",
    }:
        raise RuntimeError("Qwen model qualification is invalid")
    layers = model["fullyOffloadedLayers"]
    if (
        model["sha256"] != digest
        or model["contextTokens"] != 32_768
        or model["keyCache"] != "q8_0"
        or model["valueCache"] != "q8_0"
        or isinstance(layers, bool)
        or not isinstance(layers, int)
    ):
        raise RuntimeError("Qwen model differs from qualification")
    if layers < 1:
        raise RuntimeError("Qwen layer qualification is invalid")
    return layers


def _workload(value: object, plugin_id: str, model_id: str, task_digest: str) -> None:
    if not isinstance(value, dict) or plugin_id not in value:
        raise RuntimeError("Qwen workload has no qualification")
    workload = value[plugin_id]
    if not isinstance(workload, dict) or set(workload) != {
        "modelId",
        "taskSha256",
        "result",
    }:
        raise RuntimeError("Qwen workload qualification is invalid")
    if workload != {
        "modelId": model_id,
        "taskSha256": task_digest,
        "result": "passed",
    }:
        raise RuntimeError("Qwen workload differs from qualification")


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None
