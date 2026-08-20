"""What was measured on this workload, and which model it runs by default.

This file used to be a gate. It refused to build a worker when the mounted
GGUF was not one of the three the receipt lists, when its sha256 had moved,
or when `llama-cpp-python` was any version but 0.3.34 with two exact `.so`
digests — so replacing a model or upgrading the wheel turned all five
generation workloads off at once, with a message about qualification.

None of the refusals protected anything. `fullyOffloadedLayers` has no
reader; the context window and the cache types are `runtime.py`'s own
constants rather than anything read from here; the device and the version
are labels on the evidence a job carries. The no-CPU rule is enforced where
it can actually be observed — `validate_gpu_load` on the report of the load
that just happened.

So what is left is a record and a default: the receipt says what somebody
measured and which model a workload runs when nobody chose one, and it does
not decide what this machine is allowed to run.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.resources
import json
import re
from dataclasses import asdict, dataclass

from omnitensor.plugins.generation import GenerationTask

_DIGEST = re.compile(r"^[a-f0-9]{64}$")
_RECORDED_AT = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# Version 2 keys each workload by model. Version 1 named exactly one, which is
# why a model could be installed, qualified as a model, and still unreachable:
# DictaLM was, for as long as this file could hold one Hebrew-capable model and
# no way to ask for it.
_RECEIPT_VERSION = 2
PASSED = "passed"
FAILED = "failed"
# A pair the receipt declares and nobody has measured yet. It exists because
# the alternative is a deadlock: a workload cannot be qualified until its
# distribution is installed and runnable, and it could not be declared here
# until it was qualified. What it must never be is a claim — `_covers` reports
# it as unmeasured, and `declared_artifacts` in `omnitensor` decides what a
# client may offer, from the `selectable` flag each manifest declares.
UNMEASURED = "unmeasured"
_RESULTS = (PASSED, FAILED, UNMEASURED)
_MAX_RECEIPT_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class Qualification:
    device: str
    # Kept as a field so nothing downstream changes shape, and always empty
    # since 2026-08-20: the receipt stopped being a claim about the task. See
    # `load_qualification`.
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
    """What a load of this model reported: its device, layers and runtime.

    It no longer says whether an acceptance run covers this workload's task.
    The receipt bound a digest over the whole `GenerationTask`, so every edit
    to a prompt or a schema — including a refactor that changed nothing the
    model sees — moved it, while re-measuring needs the GPU and does not
    happen in the same commit. What that produced was a claim that was wrong
    within days and six tests that were simply red (OMNI-0501, OMNI-0565).
    The maintainer's decision on 2026-08-20 was to stop maintaining the claim
    rather than to keep a stale one: what pins a workload now is that it asks
    the model what it was reviewed asking (`tests/test_workload_prompt_contract.py`
    in `omnitensor`, and the message-assembly tests here).

    `task`, `model_id` and `model_sha256` stay in the signature: every caller
    has them, and the day this needs to say something about a particular pair
    again is the day it will want them.
    """
    return _model_qualification(_qualification_document(), model_id, model_sha256)


def load_model_qualification(model_id: str, model_sha256: str) -> Qualification:
    """The same record, for a model that is chosen per operation."""
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
    """The device the numbers came from, and whether they are about this model.

    A GGUF the receipt has never seen, or one whose digest has moved, is a
    model somebody installed — which is what the client's model chooser is
    for — and it runs. What it is not is measured, and the answer says so
    rather than the worker refusing to exist.
    """
    return Qualification(
        document["device"], _unmeasured(document["models"], model_id, model_sha256)
    )


def _unmeasured(models: object, model_id: str, model_sha256: str) -> str:
    """Empty when this exact model was measured; why not, when it was not."""
    if not isinstance(models, dict):
        return "the receipt lists no measured model"
    recorded = models.get(model_id)
    if not isinstance(recorded, dict):
        return f"{model_id} was not among the models measured on this device"
    if recorded.get("sha256") != model_sha256:
        return f"{model_id} has changed since it was measured on this device"
    return ""


def verify_native_runtime(_qualification: Qualification) -> str:
    """That llama.cpp is installed, and which version answered.

    The version and the two native digests used to have to equal the
    receipt's, so every upgrade of `llama-cpp-python` — including a security
    one — stopped all five generation workloads until somebody re-measured on
    a GPU. What the version is used for is labelling the evidence, so it is
    now read from what is installed and reported.
    """
    try:
        return f"llama-cpp-python-{importlib.metadata.version('llama-cpp-python')}"
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("the llama.cpp runtime is not installed") from error


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
    if chosen not in models or models[chosen]["result"] == FAILED:
        # A default that *failed* hands every job that did not choose a model
        # to a model somebody measured and rejected. A default nobody has
        # measured yet is a different thing: `load_qualification` stopped
        # gating on the receipt deliberately, so such a pair runs and the
        # answer carries "was not measured" rather than refusing to start.
        raise RuntimeError("workload default is a model that failed")
    return workload


def _workload_model(record: object) -> None:
    if not isinstance(record, dict) or "result" not in record:
        raise RuntimeError("workload qualification is invalid")
    if set(record) - {"result", "taskSha256", "reason"}:
        raise RuntimeError("workload qualification is invalid")
    if record["result"] not in _RESULTS:
        raise RuntimeError("workload result is invalid")
    if record["result"] == UNMEASURED:
        # Nothing ran, so there is no task to have a digest of. A digest here
        # would be a measurement nobody took.
        if "taskSha256" in record:
            raise RuntimeError("an unmeasured pair records no task digest")
        return
    if not _is_digest(record.get("taskSha256")):
        raise RuntimeError("workload task digest is invalid")
    # A failure is recorded rather than dropped, so nobody re-derives it in six
    # months — and a recorded failure without its reason is a note that says
    # only "no".
    if record["result"] == FAILED and not str(record.get("reason", "")).strip():
        raise RuntimeError("workload failure has no reason")


def _is_digest(value: object) -> bool:
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None
