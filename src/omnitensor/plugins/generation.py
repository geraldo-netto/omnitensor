"""Provider-neutral contracts for bounded accelerator-backed generation.

Private source content belongs to a workload worker.  The service routes only
an opaque reference plus a versioned task contract, and receives one bounded
structured document.  Native model SDKs therefore stay outside the core
process and can be terminated with their worker when cancellation is not
cooperative.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import jsonschema

from .protocol import CancellationToken, ProgressReporter

MAX_PROMPT_CHARACTERS = 32_768
MAX_OUTPUT_SCHEMA_BYTES = 64 * 1024
MAX_CONTEXT_TOKENS = 262_144
MAX_OUTPUT_TOKENS = 16_384
MAX_OUTPUT_BYTES = 1024 * 1024
MAX_CONTENT_REFERENCES = 32
_IDENTIFIER = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_DIGEST = re.compile(r"^[a-f0-9]{64}$")
_PRE_GENERATION_FAILURES = frozenset({"admission-refused", "model-load-failed"})
_PREFERENCES = frozenset({("gpu",), ("npu", "gpu")})


class GenerationError(RuntimeError):
    """Stable generation refusal safe to expose without private content."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


class ProviderGenerationError(GenerationError):
    """A provider refusal that records whether model generation began."""

    def __init__(self, code: str, detail: str, *, generation_started: bool):
        self.generation_started = generation_started
        super().__init__(code, detail)


@dataclass(frozen=True, slots=True)
class GenerationLimits:
    context_tokens: int
    # No ceiling by default. An output budget is a wrong answer waiting for a
    # long enough one: the generation stops mid-object and the truncation
    # surfaces as an unexplained invalid reply rather than as "there was more".
    # ``None`` means the only bound is the context window — what is left of it
    # after the prompt, which is arithmetic rather than a number somebody
    # chose. A task may still declare a budget; nothing requires one.
    output_tokens: int | None = None
    # No ceiling by default: a reply is bounded by the token budget that
    # produced it and by host pressure, not by a byte count nobody can predict
    # from the question. A task may still declare one; ``None`` means it did
    # not, and nothing here refuses on size alone.
    output_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class GenerationTask:
    task_id: str
    task_version: int
    prompt_id: str
    prompt_version: int
    system_prompt: str
    instruction_template: str
    modalities: tuple[str, ...]
    output_schema: dict
    limits: GenerationLimits


@dataclass(frozen=True, slots=True)
class ArtifactProvenance:
    model_id: str
    model_version: str
    sha256: str
    source_uri: str
    license_spdx: str


@dataclass(frozen=True, slots=True)
class GenerationProviderDescriptor:
    provider_id: str
    accelerator: str
    runtime: str
    qualified: bool
    provenance: ArtifactProvenance


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    request_id: str
    task_id: str
    content_references: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GenerationResult:
    request_id: str
    provider_id: str
    accelerator: str
    provenance: ArtifactProvenance
    document: dict


@runtime_checkable
class GenerationWorker(Protocol):
    """Killable private worker boundary around one native generation SDK."""

    @property
    def descriptor(self) -> GenerationProviderDescriptor: ...

    async def generate(
        self,
        task: GenerationTask,
        request: GenerationRequest,
        cancellation: CancellationToken,
        progress: ProgressReporter,
    ) -> str: ...

    async def terminate(self, request_id: str) -> None: ...


def parse_generation_task(document: Mapping[str, object]) -> GenerationTask:
    """Parse one closed task contract and validate its structured-output schema."""
    required = {
        "taskId",
        "taskVersion",
        "prompt",
        "modalities",
        "outputSchema",
        "limits",
    }
    if not isinstance(document, Mapping) or set(document) != required:
        raise GenerationError("task-invalid", "generation task fields do not match version 1")
    task_id = _bounded_identifier(document["taskId"], "task id")
    task_version = _bounded_integer(document["taskVersion"], "task version", 1, 65_535)
    prompt_id, prompt_version, system_prompt, instruction_template = _parse_prompt(
        document["prompt"]
    )
    modalities = _parse_modalities(document["modalities"])
    output_schema = _parse_output_schema(document["outputSchema"])
    limits = _parse_limits(document["limits"])
    return GenerationTask(
        task_id,
        task_version,
        prompt_id,
        prompt_version,
        system_prompt,
        instruction_template,
        modalities,
        output_schema,
        limits,
    )


def parse_provider_descriptor(document: Mapping[str, object]) -> GenerationProviderDescriptor:
    """Parse one accelerator-qualified provider and its immutable model identity."""
    if not isinstance(document, Mapping) or set(document) != {
        "providerId",
        "accelerator",
        "runtime",
        "qualified",
        "provenance",
    }:
        raise GenerationError("provider-invalid", "provider fields do not match version 1")
    provider_id = _bounded_identifier(document["providerId"], "provider id")
    accelerator = document["accelerator"]
    if accelerator not in {"gpu", "npu"}:
        raise GenerationError("provider-invalid", "generation providers must use GPU or NPU")
    runtime = _bounded_text(document["runtime"], "runtime", 120)
    if not isinstance(document["qualified"], bool):
        raise GenerationError("provider-invalid", "qualified must be boolean")
    provenance = document["provenance"]
    if not isinstance(provenance, Mapping) or set(provenance) != {
        "modelId",
        "modelVersion",
        "sha256",
        "sourceUri",
        "licenseSpdx",
    }:
        raise GenerationError(
            "provider-invalid", "artifact provenance fields do not match version 1"
        )
    model_id = _bounded_identifier(provenance["modelId"], "model id")
    model_version = _bounded_text(provenance["modelVersion"], "model version", 80)
    digest = provenance["sha256"]
    if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
        raise GenerationError("provider-invalid", "artifact digest must be lower-case SHA-256")
    source_uri = _bounded_text(provenance["sourceUri"], "source URI", 2048)
    if not source_uri.startswith("https://"):
        raise GenerationError("provider-invalid", "artifact source URI must use HTTPS")
    license_spdx = _bounded_text(provenance["licenseSpdx"], "license", 80)
    return GenerationProviderDescriptor(
        provider_id,
        accelerator,
        runtime,
        document["qualified"],
        ArtifactProvenance(model_id, model_version, digest, source_uri, license_spdx),
    )


def generation_request(
    request_id: str,
    task_id: str,
    content_references: Sequence[str],
) -> GenerationRequest:
    """Build the private-worker request without accepting source content inline."""
    request = _bounded_text(request_id, "request id", 120)
    task = _bounded_identifier(task_id, "task id")
    if (
        isinstance(content_references, (str, bytes))
        or not 1 <= len(content_references) <= MAX_CONTENT_REFERENCES
    ):
        raise GenerationError(
            "request-invalid", f"content references must contain 1-{MAX_CONTENT_REFERENCES} items"
        )
    references = tuple(
        _bounded_text(reference, "content reference", 240) for reference in content_references
    )
    if len(set(references)) != len(references):
        raise GenerationError("request-invalid", "content references must be unique")
    return GenerationRequest(request, task, references)


def _looks_truncated(raw: str) -> bool:
    """Whether the reply stops mid-structure rather than being malformed.

    A closed JSON object is the contract, so text that opens one and never
    closes it is output that ran out of room — the ordinary result of asking a
    long question of a task with a bounded token budget.
    """
    text = raw.strip()
    return bool(text) and text.startswith(("{", "[")) and not text.endswith(("}", "]"))


def validate_structured_output(task: GenerationTask, raw: str) -> dict:
    """Decode one provider reply within the task's exact byte and schema limits."""
    if not isinstance(raw, str):
        raise GenerationError("provider-output-invalid", "provider output must be JSON text")
    if task.limits.output_bytes is not None:
        encoded = raw.encode("utf-8")
        if len(encoded) > task.limits.output_bytes:
            raise GenerationError(
                "provider-output-invalid", "provider output exceeds its byte limit"
            )
    try:
        document = json.loads(raw, parse_constant=lambda value: (_raise_json_constant(value)))
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        # Told apart because they need different answers from the person: a
        # model that ran out of output budget mid-object is asking for a
        # narrower question, while genuinely malformed JSON is a provider that
        # does not honour its own contract.
        if _looks_truncated(raw):
            raise GenerationError(
                "provider-output-truncated",
                "provider stopped before its JSON was complete",
            ) from error
        raise GenerationError(
            "provider-output-invalid", "provider output is not strict JSON"
        ) from error
    violations = sorted(
        jsonschema.Draft202012Validator(task.output_schema).iter_errors(document),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    )
    if violations:
        raise GenerationError("provider-output-invalid", violations[0].message)
    assert isinstance(document, dict)
    return document


class GenerationRouter:
    """Select qualified workers and permit only pre-generation NPU fallback."""

    def __init__(
        self,
        workers: Sequence[GenerationWorker],
        *,
        accelerator_preference: Sequence[str] = ("gpu",),
    ) -> None:
        preference = tuple(accelerator_preference)
        if preference not in _PREFERENCES:
            raise GenerationError(
                "preference-invalid", "generation preference must be gpu or explicitly npu,gpu"
            )
        by_accelerator: dict[str, GenerationWorker] = {}
        for worker in workers:
            if not isinstance(worker, GenerationWorker):
                raise GenerationError(
                    "provider-invalid", "worker does not implement generation port"
                )
            descriptor = worker.descriptor
            if descriptor.accelerator not in {"gpu", "npu"}:
                raise GenerationError("provider-invalid", "CPU generation providers are forbidden")
            if not descriptor.qualified:
                continue
            if descriptor.accelerator in by_accelerator:
                raise GenerationError(
                    "provider-invalid", f"multiple providers claim {descriptor.accelerator}"
                )
            by_accelerator[descriptor.accelerator] = worker
        self._workers = by_accelerator
        self._preference = preference

    def require_ready(self) -> None:
        """Require one qualified provider whose local artifacts pass preflight."""
        failure: GenerationError | None = None
        for accelerator in self._preference:
            worker = self._workers.get(accelerator)
            if worker is None:
                continue
            preflight = getattr(worker, "preflight", None)
            if not callable(preflight):
                return
            try:
                preflight()
            except GenerationError as error:
                failure = error
                continue
            return
        if failure is not None:
            raise failure
        raise GenerationError("provider-unavailable", "no qualified provider is available")

    async def run(
        self,
        task: GenerationTask,
        request: GenerationRequest,
        cancellation: CancellationToken,
        progress: ProgressReporter,
    ) -> GenerationResult:
        if request.task_id != task.task_id:
            raise GenerationError("task-mismatch", "request does not name the supplied task")
        last_failure: ProviderGenerationError | None = None
        for accelerator in self._preference:
            worker = self._workers.get(accelerator)
            if worker is None:
                continue
            cancellation.raise_if_cancelled()
            try:
                raw = await worker.generate(task, request, cancellation, progress)
            except ProviderGenerationError as error:
                last_failure = error
                if (
                    accelerator == "npu"
                    and not error.generation_started
                    and error.code in _PRE_GENERATION_FAILURES
                ):
                    continue
                raise
            except BaseException:
                await worker.terminate(request.request_id)
                raise
            document = validate_structured_output(task, raw)
            return GenerationResult(
                request.request_id,
                worker.descriptor.provider_id,
                accelerator,
                worker.descriptor.provenance,
                document,
            )
        if last_failure is not None:
            raise last_failure
        raise GenerationError("provider-unavailable", "no qualified provider is available")


def _bounded_identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) > 120 or _IDENTIFIER.fullmatch(value) is None:
        raise GenerationError("contract-invalid", f"{label} is invalid")
    return value


def _parse_prompt(value: object) -> tuple[str, int, str, str]:
    if not isinstance(value, Mapping) or set(value) != {
        "id",
        "version",
        "system",
        "instructionTemplate",
    }:
        raise GenerationError("task-invalid", "prompt fields do not match version 1")
    prompt_id = _bounded_identifier(value["id"], "prompt id")
    prompt_version = _bounded_integer(value["version"], "prompt version", 1, 65_535)
    system_prompt = _bounded_text(value["system"], "system prompt", MAX_PROMPT_CHARACTERS)
    instruction_template = _bounded_text(
        value["instructionTemplate"], "instruction template", MAX_PROMPT_CHARACTERS
    )
    if "{{UNTRUSTED_CONTENT}}" not in instruction_template:
        raise GenerationError(
            "task-invalid", "instruction template must mark untrusted content explicitly"
        )
    return prompt_id, prompt_version, system_prompt, instruction_template


def _parse_modalities(value: object) -> tuple[str, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not 1 <= len(value) <= 2
        or any(item not in {"text", "image"} for item in value)
        or len(set(value)) != len(value)
    ):
        raise GenerationError("task-invalid", "modalities must be unique text/image values")
    return tuple(value)


def _parse_output_schema(value: object) -> dict:
    if not isinstance(value, dict):
        raise GenerationError("task-invalid", "output schema must be an object")
    encoded = _canonical_json(value, "output schema")
    if len(encoded) > MAX_OUTPUT_SCHEMA_BYTES:
        raise GenerationError("task-invalid", "output schema exceeds its byte limit")
    if value.get("type") != "object" or value.get("additionalProperties") is not False:
        raise GenerationError("task-invalid", "output schema must be a closed object")
    try:
        jsonschema.Draft202012Validator.check_schema(value)
    except jsonschema.SchemaError as error:
        raise GenerationError(
            "task-invalid", f"output schema is invalid: {error.message}"
        ) from error
    return json.loads(encoded)


def _parse_limits(value: object) -> GenerationLimits:
    # ``outputTokens`` and ``outputBytes`` are both optional, and both being
    # absent is the ordinary case: a task states the context it needs and lets
    # the answer be as long as the answer is. A task written when they were
    # required stays valid rather than being rejected for carrying a ceiling
    # this no longer asks for.
    if not isinstance(value, Mapping) or not {"contextTokens"} <= set(value) <= {
        "contextTokens",
        "outputTokens",
        "outputBytes",
    }:
        raise GenerationError("task-invalid", "generation limits do not match version 1")
    context_tokens = _bounded_integer(
        value["contextTokens"], "context token limit", 1, MAX_CONTEXT_TOKENS
    )
    output_tokens = (
        _bounded_integer(value["outputTokens"], "output token limit", 1, MAX_OUTPUT_TOKENS)
        if "outputTokens" in value
        else None
    )
    output_bytes = (
        _bounded_integer(value["outputBytes"], "output byte limit", 2, MAX_OUTPUT_BYTES)
        if "outputBytes" in value
        else None
    )
    if output_tokens is not None and output_tokens >= context_tokens:
        raise GenerationError("task-invalid", "output token limit must be below context limit")
    return GenerationLimits(context_tokens, output_tokens, output_bytes)


def _bounded_integer(value: object, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise GenerationError("contract-invalid", f"{label} must be in [{minimum}, {maximum}]")
    return value


def _bounded_text(value: object, label: str, maximum: int) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise GenerationError("contract-invalid", f"{label} must contain 1-{maximum} characters")
    return value


def _canonical_json(document: object, label: str) -> bytes:
    try:
        return json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise GenerationError("contract-invalid", f"{label} is not strict JSON") from error


def _raise_json_constant(value: str):
    raise ValueError(f"invalid JSON constant: {value}")
