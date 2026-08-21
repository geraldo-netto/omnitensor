"""In-process llama.cpp/Vulkan adapter: load, lease, decode, and nothing else.

What each workload tells the model arrives through the `TaskPrompting` port
and is stated beside that workload's task; which references may be cited lives
in :mod:`hints` and what is bound back out of an answer in :mod:`grounding`.
This module holds the GPU lease and the token stream.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from omnitensor.plugins.acceptance_kit import NativeLoadReport
from omnitensor.plugins.fragments import (
    PrivateFragmentStore,
)
from omnitensor.plugins.generation import (
    GenerationRequest,
    GenerationTask,
    ProviderGenerationError,
)
from omnitensor.plugins.prompting import NO_PROMPTING, TaskPrompting
from omnitensor.plugins.protocol import CancellationToken, ProgressReporter
from omnitensor.plugins.tuning import WorkloadTuning

from .grammar import grammar_schema
from .grounding import bind_grounding_metadata, fragment_span
from .hints import citable_hint

MAX_RUNTIME_CONTEXT_TOKENS = 32_768
# How many model layers to place on the GPU. -1 is llama.cpp's "all of them",
# which is right for every model that fits — and was hard-coded, so a model
# larger than VRAM could not even ask for fewer. The environment variable is
# the person's knob (OMNI-0586): set it and the load carries however many
# layers fit, with the real offloaded/total split published in the receipt.
GPU_LAYERS_VARIABLE = "OMNITENSOR_GPU_LAYERS"


def _gpu_layer_budget() -> int:
    configured = os.environ.get(GPU_LAYERS_VARIABLE, "").strip()
    if not configured:
        return -1
    try:
        return int(configured)
    except ValueError:
        # "-1, meaning all layers" is the exact opposite of what somebody who
        # set a budget asked for; a typo must refuse, not maximise.
        raise ProviderGenerationError(
            "model-load-failed",
            f"{GPU_LAYERS_VARIABLE} must be an integer, not {configured!r}",
            generation_started=False,
        ) from None


_OFFLOAD = re.compile(r"offloaded\s+(\d+)/(\d+)\s+layers\s+to\s+GPU", re.IGNORECASE)
_DEVICE = re.compile(r"using device Vulkan\d+ \((.+)\) \([0-9a-fA-F:.]+\)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class _LoadFacts:
    """What llama.cpp states about a load through its API rather than its log.

    ``None`` means the installed build does not answer that question, which is
    not the same as answering zero: only a real zero is evidence.
    """

    layers: int | None = None
    devices: int | None = None


def _structured_load_facts(llama: object, llama_cpp: object) -> _LoadFacts:
    """Ask the loaded model what it can be asked, and settle for less if not.

    The layer count and the number of devices holding layers are stable C
    entry points; the device *name* is not exposed by any of them, which is
    why the log regex survives as the only source for that one field.
    """
    handle = getattr(getattr(llama, "_model", None), "model", None)
    if handle is None:
        handle = getattr(llama, "model", None)
    if handle is None:
        return _LoadFacts()
    return _LoadFacts(
        layers=_native_count(llama_cpp, "llama_model_n_layer", handle),
        devices=_native_count(llama_cpp, "llama_model_n_devices", handle),
    )


def _native_count(llama_cpp: object, name: str, handle: object) -> int | None:
    entry = getattr(llama_cpp, name, None)
    if entry is None:
        return None
    try:
        value = int(entry(handle))
    except Exception:  # noqa: BLE001 - an older build may bind a different shape
        return None
    return value if value >= 0 else None


def _log_offload(text: str) -> tuple[int | None, int | None]:
    """The offloaded and total layer counts, if this build's log states them."""
    match = _OFFLOAD.search(text)
    if match is None:
        return None, None
    return int(match.group(1)), int(match.group(2))


class LlamaVulkanRuntime:
    """Load any GGUF only while holding the host's cross-worker GPU lease."""

    def __init__(
        self,
        store: PrivateFragmentStore,
        lease_path: Path,
        *,
        prompting: TaskPrompting = NO_PROMPTING,
    ) -> None:
        if not isinstance(store, PrivateFragmentStore):
            raise TypeError("store must satisfy PrivateFragmentStore")
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
        self._tuning = WorkloadTuning()
        # What the workload adds to the prompt, asked for rather than known:
        # this adapter loads a model and decodes tokens, and none of that
        # changes when a workload is added.
        self._prompting = prompting
        # The model is native memory shared between the event loop and the
        # worker thread running a decode.  `terminate` used to call `_release`
        # -- and so `llama.close()` -- while `create_chat_completion` was still
        # reading the same context, freeing it under an in-flight decode; two
        # concurrent `generate` calls did the same to each other through their
        # `finally`.  Every use and every free now holds `_in_use`, and
        # `_stopping` is how a waiting `terminate` asks the decode to stop
        # rather than pull the model out from under it.
        self._in_use = threading.RLock()
        self._stopping = threading.Event()
        self._log_callback = None

    def tune(self, tuning: WorkloadTuning) -> None:
        """Hold the tuning this workload was configured with.

        Held rather than passed per request because a worker reads its
        configuration once, at `start()`, and is replaced when it changes.
        """
        self._tuning = tuning

    async def load(self, artifacts: tuple[Path, ...], accelerator: str) -> NativeLoadReport:
        if accelerator != "gpu" or len(artifacts) != 1:
            raise ProviderGenerationError(
                "model-load-failed",
                "Vulkan generation requires one GPU GGUF",
                generation_started=False,
            )
        model = Path(artifacts[0])
        if model.suffix != ".gguf" or not model.is_file():
            raise ProviderGenerationError(
                "model-load-failed", "generation GGUF is unavailable", generation_started=False
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
                "model-load-failed", "generation model was not loaded", generation_started=False
            )
        try:
            cancellation.raise_if_cancelled()
            if self._llama is None:
                await self.load((self._model_path,), "gpu")
            cancellation.raise_if_cancelled()
            self._active_request = request.request_id
            await progress.report(_generation_progress(request.request_id, 0.15))
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
            # Ask first, free second: the decode stops at its next chunk and
            # drops the lock, and only then is the model closed.
            self._stopping.set()
            try:
                await asyncio.to_thread(self._release)
            finally:
                self._stopping.clear()

    def _acquire_and_load(self) -> NativeLoadReport:
        # Loading writes `_llama` and `_lease`, so it takes the same lock a
        # decode and a release do: a load must not run beside either.
        with self._in_use:
            return self._acquire_and_load_locked()

    def _acquire_and_load_locked(self) -> NativeLoadReport:
        self._lease_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        lease = self._lease_path.open("a+b")
        try:
            fcntl.flock(lease.fileno(), fcntl.LOCK_EX)
        except BaseException:
            # `_release` closes `self._lease`, which is not set until the lock
            # is held — so an interrupted acquisition used to leave the file
            # open with nobody able to close it, and the lease held until the
            # process exited.
            lease.close()
            raise
        self._lease = lease
        try:
            from llama_cpp import (  # type: ignore[import-not-found]  # noqa: PLC0415 - deferred: an optional or heavy dependency
                Llama,
                llama_cpp,
            )
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
        try:
            self._llama = self._load_llama(Llama, llama_cpp)
        finally:
            # llama_log_set is global and the callback outlives the load it was
            # made for: it used to stay registered for the process lifetime,
            # appending every later log line to this load's list -- growing
            # without bound, and, with a second runtime alive, capturing the
            # other one's load logs. Hand the global sink back to something
            # that keeps nothing.
            self._discard_llama_logs(llama_cpp)
        offloaded, total = self._prove_gpu_offload(llama_cpp, "".join(logs))
        self._load_report = NativeLoadReport("llama.cpp-vulkan", "Vulkan", total, offloaded, False)
        logs.clear()
        return self._load_report

    def _prove_gpu_offload(self, llama_cpp, text: str) -> tuple[int, int]:
        """How many layers reached the GPU, out of how many — or a refusal.

        This used to demand every layer and refuse otherwise, which capped
        what the machine may run: a model larger than VRAM with its experts
        in RAM is real work, not a fallback (OMNI-0586). The refusals that
        remain are the honest ones — a load that reached no device at all,
        and a build whose answers cannot be read either way.

        The structured calls answer first where the installed build binds
        them; the log is the fallback, and a log that cannot be read at all is
        reported as unverified rather than as a proven partial offload.
        """
        facts = _structured_load_facts(self._llama, llama_cpp)
        offloaded, total = _log_offload(text)
        if facts.devices == 0:
            # The API said it outright: no device holds a layer. Nothing the
            # log says can make that a GPU load.
            raise ProviderGenerationError(
                "model-load-failed",
                "llama.cpp did not prove full Vulkan layer offload",
                generation_started=False,
            )
        if offloaded is None and facts.devices and facts.layers:
            # The structured calls prove *a* GPU load — some device holds
            # layers — but `llama_model_n_layer` is the model's total, which
            # says nothing about placement: claiming `total/total` here would
            # report a partial offload as full the day a build binds
            # `llama_model_n_devices` (0.3.34 does not, so this branch is
            # currently unreachable). Without a readable split the split is
            # unverified, and the honest answer is to say so (OMNI-0592).
            raise ProviderGenerationError(
                "model-load-unverified",
                "llama.cpp placed layers on a device but reported no offload split",
                generation_started=False,
            )
        if offloaded is None or total is None:
            raise ProviderGenerationError(
                "model-load-unverified",
                "llama.cpp reported neither a device count nor a readable offload log",
                generation_started=False,
            )
        if offloaded < 1:
            raise ProviderGenerationError(
                "model-load-failed",
                "llama.cpp placed no model layer on a Vulkan device",
                generation_started=False,
            )
        device = _DEVICE.search(text)
        if device is None:
            # Which card ran it is part of the evidence a qualification keeps,
            # and no public llama.cpp call returns it — only the log does.
            raise ProviderGenerationError(
                "model-load-unverified",
                "llama.cpp did not name the Vulkan device it loaded on",
                generation_started=False,
            )
        self._physical_device = device.group(1)
        return offloaded, total

    def _discard_llama_logs(self, llama_cpp) -> None:
        @llama_cpp.llama_log_callback
        def discard(_level, _text, _data):
            return None

        # Kept on the instance because llama.cpp holds the pointer, not the
        # object: a callback that is garbage collected is a dangling call.
        self._log_callback = discard
        llama_cpp.llama_log_set(discard, None)

    def _load_llama(self, Llama, llama_cpp):  # noqa: N803 - the class is llama-cpp's name
        return Llama(
            model_path=str(self._model_path),
            n_ctx=MAX_RUNTIME_CONTEXT_TOKENS,
            n_gpu_layers=_gpu_layer_budget(),
            main_gpu=0,
            offload_kqv=True,
            op_offload=True,
            flash_attn=True,
            type_k=llama_cpp.GGML_TYPE_Q8_0,
            type_v=llama_cpp.GGML_TYPE_Q8_0,
            verbose=False,
        )

    @property
    def physical_device(self) -> str:
        return self._physical_device

    def _generate_sync(
        self,
        task: GenerationTask,
        request: GenerationRequest,
        cancellation: CancellationToken,
    ) -> str:
        with self._in_use:
            llama = self._llama
            if llama is None:
                raise RuntimeError("model not loaded")
            return self._generate_locked(llama, task, request, cancellation)

    def _generate_locked(
        self,
        llama,
        task: GenerationTask,
        request: GenerationRequest,
        cancellation: CancellationToken,
    ) -> str:
        content = _private_content(self._store, request)
        instruction = task.instruction_template.replace("{{UNTRUSTED_CONTENT}}", content)
        hint = citable_hint(task, request) + self._prompting.hint(task, request, self._store)

        messages = [
            {
                "role": "system",
                "content": (
                    f"{task.system_prompt}\n{hint}"
                    "The output requestId must be exactly "
                    f"{json.dumps(request.request_id)}. Copy fragment metadata exactly.\n"
                    "/no_think"
                ),
            },
            {"role": "user", "content": instruction},
        ]
        # After the request rather than inside the system turn: the grounding
        # and citation rules are the service's and stay authoritative, and a
        # preference somebody typed may not quietly outrank them.
        tuning_message = self._tuning.message()
        if tuning_message is not None:
            messages.append(tuning_message)
        raw = _complete_json(llama, messages, task, cancellation, self._stopping)
        reconsideration = self._prompting.reconsideration(task, hint, raw, request, self._store)
        if reconsideration is not None:
            messages.extend(
                (
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": reconsideration},
                )
            )
            raw = _complete_json(llama, messages, task, cancellation, self._stopping)
        return bind_grounding_metadata(raw, task, request, self._store)

    def _release(self) -> None:
        with self._in_use:
            self._release_locked()

    def _release_locked(self) -> None:
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


def _complete_json(
    llama,
    messages: list[dict[str, str]],
    task: GenerationTask,
    cancellation: CancellationToken,
    stopping: threading.Event | None = None,
) -> str:
    reply = llama.create_chat_completion(
        messages=messages,
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
        if stopping is not None and stopping.is_set():
            # `terminate` is waiting to close this model: stop reading it now,
            # rather than making the caller wait out a whole generation.
            raise asyncio.CancelledError
        choices = reply_chunk.get("choices") if isinstance(reply_chunk, Mapping) else None
        delta = choices[0].get("delta") if isinstance(choices, list) and choices else None
        text = delta.get("content") if isinstance(delta, Mapping) else None
        if isinstance(text, str) and text:
            chunks.append(text)
    if not chunks:
        raise RuntimeError("llama.cpp returned no JSON content")
    return "".join(chunks)


def _private_content(store: PrivateFragmentStore, request: GenerationRequest) -> str:
    fragments = []
    for reference in request.content_references:
        fragment = store.resolve(request.request_id, reference)
        document = {
            "sourceRef": fragment.reference,
            "sourceSha256": fragment.source_sha256,
            "page": fragment.page,
            "text": fragment.text,
            "textSha256": fragment.text_sha256,
            "span": fragment_span(fragment),
        }
        fragments.append(document)
    return json.dumps(fragments, ensure_ascii=False, separators=(",", ":"))


def _output_token_limit(requested: int | None) -> int | None:
    """What to ask llama.cpp for, which is nothing when nothing was asked.

    ``None`` lets generation run to the model's own stop token or to the edge
    of the context, whichever comes first. That is the only real bound: a
    smaller one refuses or truncates an answer somebody needed, and this
    provider used to refuse outright above 4,096 tokens — a policy number
    wearing the shape of a hardware limit.
    """
    return requested


def _generation_progress(request_id: str, fraction: float):
    from omnitensor.plugins.protocol import PluginProgress  # noqa: PLC0415

    return PluginProgress(request_id, "native-generation", fraction, "", 1)
