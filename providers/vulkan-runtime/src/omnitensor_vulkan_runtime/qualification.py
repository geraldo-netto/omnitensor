"""Fail-closed binding between accepted tasks, native bytes, models, and GPU."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.resources
import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from omnitensor.plugins.generation import GenerationTask
from omnitensor.preparation import file_digest

_DIGEST = re.compile(r"^[a-f0-9]{64}$")
_RECORDED_AT = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# Version 2 keys each workload by model. Version 1 named exactly one, which is
# why a model could be installed, qualified as a model, and still unreachable:
# DictaLM was, for as long as this file could hold one Hebrew-capable model and
# no way to ask for it.
_RECEIPT_VERSION = 2
PASSED = "passed"
FAILED = "failed"
_MAX_RECEIPT_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class Qualification:
    device: str
    model_layers: int
    runtime_version: str
    runtime_binaries: tuple[tuple[str, str], ...]
    # Empty when a frozen acceptance run covers this exact workload, model and
    # task; otherwise the reason it does not. Carried so the answer can say so,
    # never consulted to decide whether the work may run.
    covers: str = ""


def task_sha256(task: GenerationTask) -> str:
    raw = json.dumps(asdict(task), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def load_qualification(
    plugin_id: str,
    model_id: str,
    model_sha256: str,
    task: GenerationTask,
) -> Qualification:
    """The receipt's record for this pair, and whether it covers this workload.

    Not a gate. It used to raise when the receipt did not list the pair, so a
    person who chose an unmeasured model got a worker that refused to start —
    and a task digest that moved without the receipt being reissued did the
    same to a model that had been measured, which is how event extraction was
    briefly unable to start at all. Choosing a model is the person's to make
    and theirs to own; what the receipt knows is reported, never enforced.
    """
    document = _qualification_document()
    qualification = _model_qualification(document, model_id, model_sha256)
    return replace(
        qualification,
        covers=_covers(document["workloads"], plugin_id, model_id, task_sha256(task)),
    )


def _covers(value: object, plugin_id: str, model_id: str, task_digest: str) -> str:
    """Empty when the receipt covers this exactly; otherwise why it does not.

    A malformed receipt still raises. That is not an unmeasured pair, it is a
    broken install — the same class as a missing dependency — and reporting it
    as "nobody measured this" would send a person off to run an acceptance
    pass against a file the reader cannot even parse.
    """
    if not isinstance(value, dict):
        raise RuntimeError("workload qualification is invalid")
    if plugin_id not in value:
        return f"{plugin_id} has no measured models"
    workload = _workload_entry(value, plugin_id)
    record = workload["models"].get(model_id)
    if record is None:
        return f"{model_id} was not measured for {plugin_id}"
    if record["result"] != PASSED:
        return f"{model_id} did not pass {plugin_id}: {record.get('reason', '')}".strip()
    if record["taskSha256"] != task_digest:
        return f"{plugin_id} has changed since {model_id} was measured"
    return ""


def load_model_qualification(model_id: str, model_sha256: str) -> Qualification:
    """Bind an operation-specific model to the same native GPU receipt."""
    return _model_qualification(_qualification_document(), model_id, model_sha256)


def _qualification_document() -> dict:
    resource = importlib.resources.files("omnitensor_vulkan_runtime").joinpath("qualification.json")
    raw = resource.read_bytes()
    if len(raw) > _MAX_RECEIPT_BYTES:
        raise RuntimeError("qualification receipt is oversized")
    try:
        document = json.loads(raw)
    except (UnicodeError, ValueError) as error:
        raise RuntimeError("qualification receipt is invalid") from error
    if not isinstance(document, dict) or set(document) != {
        "version",
        "recordedAt",
        "device",
        "runtime",
        "models",
        "workloads",
    }:
        raise RuntimeError("qualification receipt fields are invalid")
    if document["version"] != _RECEIPT_VERSION:
        raise RuntimeError("qualification receipt version is invalid")
    # A date, not one particular date: re-qualifying a pair is expected work,
    # and a literal here would mean every run edits the check that guards it.
    # Nothing is trusted because of this field; the digests do that.
    if not isinstance(document["recordedAt"], str) or not _RECORDED_AT.fullmatch(
        document["recordedAt"]
    ):
        raise RuntimeError("qualification receipt date is invalid")
    device = document["device"]
    if not isinstance(device, str) or not device:
        raise RuntimeError("qualification device is invalid")
    return document


def _model_qualification(document: dict, model_id: str, model_sha256: str) -> Qualification:
    device = document["device"]
    runtime = _runtime(document["runtime"])
    model_layers = _model(document["models"], model_id, model_sha256)
    return Qualification(device, model_layers, runtime[0], runtime[1])


def verify_native_runtime(qualification: Qualification) -> str:
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
    return f"llama-cpp-python-{version}"


def _runtime(value: object) -> tuple[str, tuple[tuple[str, str], ...]]:
    if not isinstance(value, dict) or set(value) != {
        "distribution",
        "version",
        "wheelSha256",
        "binaries",
    }:
        raise RuntimeError("qualification runtime is invalid")
    if value["distribution"] != "llama-cpp-python" or value["version"] != "0.3.34":
        raise RuntimeError("qualification runtime identity is invalid")
    if not _is_digest(value["wheelSha256"]):
        raise RuntimeError("qualification wheel digest is invalid")
    binaries = value["binaries"]
    if not isinstance(binaries, dict) or set(binaries) != {
        "libggml-vulkan.so",
        "libllama.so",
    }:
        raise RuntimeError("qualification native inventory is invalid")
    if any(not _is_digest(digest) for digest in binaries.values()):
        raise RuntimeError("qualification native digest is invalid")
    return str(value["version"]), tuple(sorted(binaries.items()))


def _model(value: object, model_id: str, digest: str) -> int:
    if not isinstance(value, dict) or model_id not in value:
        raise RuntimeError("model has no qualification")
    model = value[model_id]
    if not isinstance(model, dict) or set(model) != {
        "sha256",
        "fullyOffloadedLayers",
        "contextTokens",
        "keyCache",
        "valueCache",
    }:
        raise RuntimeError("model qualification is invalid")
    layers = model["fullyOffloadedLayers"]
    if (
        model["sha256"] != digest
        or model["contextTokens"] != 32_768
        or model["keyCache"] != "q8_0"
        or model["valueCache"] != "q8_0"
        or isinstance(layers, bool)
        or not isinstance(layers, int)
    ):
        raise RuntimeError("model differs from qualification")
    if layers < 1:
        raise RuntimeError("layer qualification is invalid")
    return layers


def qualified_models(plugin_id: str) -> tuple[str, ...]:
    """Every model this workload passed on, in the order the receipt lists.

    What a client may offer. A pair that failed is deliberately not here: an
    unqualified model is not a slower answer, it is a worker that refuses to
    start.
    """
    workload = _workload_entry(_qualification_document()["workloads"], plugin_id)
    return tuple(
        model_id for model_id, record in workload["models"].items() if record["result"] == PASSED
    )


def default_model(plugin_id: str) -> str:
    """The model this workload runs when nobody has chosen one."""
    return _workload_entry(_qualification_document()["workloads"], plugin_id)["default"]


def _workload_entry(value: object, plugin_id: str) -> dict:
    if not isinstance(value, dict) or plugin_id not in value:
        raise RuntimeError("workload has no qualification")
    workload = value[plugin_id]
    if not isinstance(workload, dict) or set(workload) != {"default", "models"}:
        raise RuntimeError("workload qualification is invalid")
    models = workload["models"]
    if not isinstance(models, dict) or not models:
        raise RuntimeError("workload qualification lists no model")
    for record in models.values():
        _workload_model(record)
    chosen = workload["default"]
    if chosen not in models or models[chosen]["result"] != PASSED:
        # A default nobody qualified would hand every job that did not choose
        # a model to a worker that cannot start.
        raise RuntimeError("workload default is not a passing model")
    return workload


def _workload_model(record: object) -> None:
    if not isinstance(record, dict) or not {"result", "taskSha256"} <= set(record):
        raise RuntimeError("workload qualification is invalid")
    if set(record) - {"result", "taskSha256", "reason"}:
        raise RuntimeError("workload qualification is invalid")
    if record["result"] not in (PASSED, FAILED):
        raise RuntimeError("workload result is invalid")
    if not _is_digest(record["taskSha256"]):
        raise RuntimeError("workload task digest is invalid")
    # A failure is recorded rather than dropped, so nobody re-derives it in six
    # months — and a recorded failure without its reason is a note that says
    # only "no".
    if record["result"] == FAILED and not str(record.get("reason", "")).strip():
        raise RuntimeError("workload failure has no reason")


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None
