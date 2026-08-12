"""In-process llama.cpp/Vulkan adapter with bounded private prompt assembly."""

from __future__ import annotations

import asyncio
import copy
import fcntl
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import BinaryIO

from omnitensor.plugins.event_workload import MemoryFragmentStore
from omnitensor.plugins.generation import GenerationRequest, GenerationTask
from omnitensor.plugins.protocol import CancellationToken, ProgressReporter
from omnitensor.plugins.qwen import NativeLoadReport, ProviderGenerationError

MAX_RUNTIME_CONTEXT_TOKENS = 32_768
MAX_RUNTIME_OUTPUT_TOKENS = 4_096
_OFFLOAD = re.compile(r"offloaded\s+(\d+)/(\d+)\s+layers\s+to\s+GPU", re.IGNORECASE)
_DEVICE = re.compile(r"using device Vulkan\d+ \((.+)\) \([0-9a-fA-F:.]+\)", re.IGNORECASE)
_SPAN_REFERENCE = re.compile(r":span:(\d+)-(\d+)$")


class LlamaVulkanRuntime:
    """Load Qwen only while holding the host's cross-worker GPU lease."""

    def __init__(self, store: MemoryFragmentStore, lease_path: Path) -> None:
        if not isinstance(store, MemoryFragmentStore):
            raise TypeError("store must be MemoryFragmentStore")
        if not isinstance(lease_path, Path) or not lease_path.is_file():
            raise ValueError("accelerator lease is unavailable")
        self._store = store
        self._lease_path = lease_path
        self._model_path: Path | None = None
        self._llama = None
        self._lease: BinaryIO | None = None
        self._load_report: NativeLoadReport | None = None
        self._physical_device = ""
        self._active_request = ""

    async def load(self, artifacts: tuple[Path, ...], accelerator: str) -> NativeLoadReport:
        if accelerator != "gpu" or len(artifacts) != 1:
            raise ProviderGenerationError(
                "model-load-failed", "Qwen Vulkan requires one GPU GGUF", generation_started=False
            )
        model = Path(artifacts[0])
        if model.suffix != ".gguf" or not model.is_file():
            raise ProviderGenerationError(
                "model-load-failed", "Qwen GGUF is unavailable", generation_started=False
            )
        if self._llama is not None:
            assert self._load_report is not None
            return self._load_report
        self._model_path = model
        try:
            return await asyncio.to_thread(self._acquire_and_load)
        except ProviderGenerationError:
            self._release()
            raise
        except Exception as error:
            self._release()
            raise ProviderGenerationError(
                "model-load-failed", "Vulkan model loading failed", generation_started=False
            ) from error

    async def generate(
        self,
        task: GenerationTask,
        request: GenerationRequest,
        cancellation: CancellationToken,
        progress: ProgressReporter,
    ) -> str:
        if self._model_path is None:
            raise ProviderGenerationError(
                "model-load-failed", "Qwen model was not loaded", generation_started=False
            )
        if self._llama is None:
            await self.load((self._model_path,), "gpu")
        cancellation.raise_if_cancelled()
        self._active_request = request.request_id
        await progress.report(_generation_progress(request.request_id, 0.15))
        try:
            raw = await asyncio.to_thread(self._generate_sync, task, request, cancellation)
            cancellation.raise_if_cancelled()
            await progress.report(_generation_progress(request.request_id, 0.95))
            return raw
        except asyncio.CancelledError:
            raise
        except ProviderGenerationError:
            raise
        except Exception as error:
            raise ProviderGenerationError(
                "generation-failed", "native generation failed", generation_started=True
            ) from error
        finally:
            self._active_request = ""
            self._release()

    async def terminate(self, request_id: str) -> None:
        if request_id == self._active_request or request_id == "__startup__":
            await asyncio.to_thread(self._release)

    def _acquire_and_load(self) -> NativeLoadReport:
        self._lease_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        lease = self._lease_path.open("a+b")
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX)
        self._lease = lease
        try:
            from llama_cpp import Llama, llama_cpp  # type: ignore[import-not-found]
        except ImportError as error:
            raise ProviderGenerationError(
                "model-load-failed",
                "install the Vulkan llama-cpp-python runtime",
                generation_started=False,
            ) from error
        logs: list[str] = []

        @llama_cpp.llama_log_callback
        def callback(_level, text, _data):
            logs.append(text.decode("utf-8", errors="replace"))

        self._log_callback = callback
        llama_cpp.llama_log_set(callback, None)
        self._llama = Llama(
            model_path=str(self._model_path),
            n_ctx=MAX_RUNTIME_CONTEXT_TOKENS,
            n_gpu_layers=-1,
            main_gpu=0,
            offload_kqv=True,
            op_offload=True,
            flash_attn=True,
            verbose=False,
        )
        match = _OFFLOAD.search("".join(logs))
        device = _DEVICE.search("".join(logs))
        if match is None or device is None or int(match.group(1)) != int(match.group(2)):
            raise ProviderGenerationError(
                "model-load-failed",
                "llama.cpp did not prove full Vulkan layer offload",
                generation_started=False,
            )
        layers = int(match.group(2))
        self._physical_device = device.group(1)
        self._load_report = NativeLoadReport("llama.cpp-vulkan", "Vulkan", layers, layers, False)
        return self._load_report

    @property
    def physical_device(self) -> str:
        return self._physical_device

    def _generate_sync(
        self,
        task: GenerationTask,
        request: GenerationRequest,
        cancellation: CancellationToken,
    ) -> str:
        llama = self._llama
        if llama is None:
            raise RuntimeError("model not loaded")
        content = _private_content(self._store, request)
        instruction = task.instruction_template.replace("{{UNTRUSTED_CONTENT}}", content)

        reply = llama.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"{task.system_prompt}\nThe output requestId must be exactly "
                        f"{json.dumps(request.request_id)}. Copy fragment metadata exactly.\n"
                        "/no_think"
                    ),
                },
                {"role": "user", "content": instruction},
            ],
            temperature=0.0,
            top_p=1.0,
            seed=0,
            max_tokens=_output_token_limit(task.limits.output_tokens),
            response_format={
                "type": "json_object",
                "schema": grammar_schema(task.output_schema),
            },
            stream=True,
        )
        chunks = []
        for reply_chunk in reply:
            cancellation.raise_if_cancelled()
            choices = reply_chunk.get("choices") if isinstance(reply_chunk, Mapping) else None
            delta = choices[0].get("delta") if isinstance(choices, list) and choices else None
            text = delta.get("content") if isinstance(delta, Mapping) else None
            if isinstance(text, str) and text:
                chunks.append(text)
        if not chunks:
            raise RuntimeError("llama.cpp returned no JSON content")
        return "".join(chunks)

    def _release(self) -> None:
        llama = self._llama
        self._llama = None
        if llama is not None:
            close = getattr(llama, "close", None)
            if callable(close):
                close()
        lease = self._lease
        self._lease = None
        if lease is not None:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
            lease.close()


def grammar_schema(schema: Mapping[str, object]) -> dict:
    """Project unsupported large string bounds for grammar only.

    The canonical task schema remains unchanged and validates the final reply.
    """
    projected = copy.deepcopy(dict(schema))
    definitions = projected.get("$defs", {})
    if not isinstance(definitions, dict):
        raise ValueError("generation schema definitions are invalid")
    projected = _inline_local_refs(projected, definitions, ())
    projected.pop("$defs", None)
    _project_grammar_keywords(projected)
    return projected


def _project_grammar_keywords(value) -> None:
    if isinstance(value, list):
        for child in value:
            _project_grammar_keywords(child)
        return
    if not isinstance(value, dict):
        return
    value.pop("maxLength", None)
    value.pop("pattern", None)
    properties = value.get("properties")
    if isinstance(properties, dict):
        confirmation = properties.get("confirmation")
        if isinstance(confirmation, dict) and "pending" in confirmation.get("enum", ()):
            properties["confirmation"] = {"const": "pending"}
    for child in value.values():
        _project_grammar_keywords(child)


def _inline_local_refs(value, definitions: Mapping[str, object], stack: tuple[str, ...]):
    """Inline bounded local definitions unsupported by llama.cpp grammars."""
    if isinstance(value, list):
        return [_inline_local_refs(child, definitions, stack) for child in value]
    if not isinstance(value, dict):
        return value
    reference = value.get("$ref")
    if reference is not None:
        prefix = "#/$defs/"
        if not isinstance(reference, str) or not reference.startswith(prefix):
            raise ValueError("generation schema uses an unsupported reference")
        name = reference.removeprefix(prefix)
        if name in stack or name not in definitions:
            raise ValueError("generation schema reference is invalid")
        target = definitions[name]
        if not isinstance(target, dict):
            raise ValueError("generation schema definition is invalid")
        merged = copy.deepcopy(target)
        merged.update({key: child for key, child in value.items() if key != "$ref"})
        return _inline_local_refs(merged, definitions, (*stack, name))
    return {key: _inline_local_refs(child, definitions, stack) for key, child in value.items()}


def _private_content(store: MemoryFragmentStore, request: GenerationRequest) -> str:
    fragments = []
    for reference in request.content_references:
        fragment = store.resolve(request.request_id, reference)
        document = {
            "sourceRef": fragment.reference,
            "sourceSha256": fragment.source_sha256,
            "page": fragment.page,
            "text": fragment.text,
            "textSha256": fragment.text_sha256,
        }
        match = _SPAN_REFERENCE.search(fragment.reference)
        document["span"] = (
            {"start": int(match.group(1)), "end": int(match.group(2))}
            if match is not None
            else {"start": 0, "end": len(fragment.text)}
        )
        fragments.append(document)
    return json.dumps(fragments, ensure_ascii=False, separators=(",", ":"))


def _output_token_limit(requested: int) -> int:
    if requested > MAX_RUNTIME_OUTPUT_TOKENS:
        raise ProviderGenerationError(
            "admission-refused",
            f"Qwen Vulkan output limit is {MAX_RUNTIME_OUTPUT_TOKENS} tokens",
            generation_started=False,
        )
    return requested


def _generation_progress(request_id: str, fraction: float):
    from omnitensor.plugins.protocol import PluginProgress

    return PluginProgress(request_id, "native-generation", fraction, "", 1)
