"""Cheap entry-point factories for the four independently identified wheels."""

from __future__ import annotations

from pathlib import Path

from omnitensor.plugins.document_qa import DocumentQuestionPlugin, document_question_task
from omnitensor.plugins.event_workload import (
    EventExtractionPlugin,
    EventRecoveryJournal,
    MemoryFragmentStore,
    event_generation_task,
)
from omnitensor.plugins.file_organizer import FileOrganizerPlugin, file_organizer_task
from omnitensor.plugins.generation import (
    ArtifactProvenance,
    GenerationProviderDescriptor,
    GenerationRouter,
)
from omnitensor.plugins.protocol import (
    CancellationToken,
    PluginContext,
    PluginHealth,
    PluginRequest,
    PluginResult,
    ProgressReporter,
    WorkloadPlugin,
)
from omnitensor.plugins.qwen import LlamaCppVulkanQwenWorker, _validate_gpu_load
from omnitensor.plugins.selected_text import SelectedTextPlugin, selected_text_task
from omnitensor.sdk import current_plugin_bootstrap

from .bge import BgeVulkanEmbedder
from .qualification import Qualification, load_qualification, verify_native_runtime
from .runtime import LlamaVulkanRuntime

QWEN_ARTIFACT_ID = "qwen3-0-6b-q8-0"
QWEN_LARGE_ARTIFACT_ID = "qwen3-4b-q4-k-m"
BGE_ARTIFACT_ID = "bge-small-en-v1-5-ask-gpu"
QWEN_SOURCE = (
    "https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/"
    "1208e45d782fe18602c5eaf10e5758d5b0f24c03/Qwen3-0.6B-Q8_0.gguf"
)
QWEN_LARGE_SOURCE = (
    "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/"
    "bc640142c66e1fdd12af0bd68f40445458f3869b/Qwen3-4B-Q4_K_M.gguf"
)
_LARGE_WORKLOADS = frozenset({"event-extraction", "file-organizer"})


class QualifiedWorkload:
    """Prove native GPU startup after handshake, then delegate the workload."""

    def __init__(
        self,
        plugin: WorkloadPlugin,
        runtime: LlamaVulkanRuntime,
        model_path: Path,
        qualification: Qualification,
        embedder: BgeVulkanEmbedder | None = None,
    ) -> None:
        self.plugin_id = plugin.plugin_id
        self._plugin = plugin
        self._runtime = runtime
        self._model_path = model_path
        self._qualification = qualification
        self._embedder = embedder

    async def start(self, context: PluginContext) -> None:
        await self._plugin.start(context)
        try:
            verify_native_runtime(self._qualification)
            report = await self._runtime.load((self._model_path,), "gpu")
            _validate_gpu_load(report)
            if (
                report.total_model_layers != self._qualification.model_layers
                or self._runtime.physical_device != self._qualification.device
            ):
                raise RuntimeError("native Qwen load differs from qualification")
            await self._runtime.terminate("__startup__")
            if self._embedder is not None:
                await self._embedder.preflight()
        except BaseException:
            await self._runtime.terminate("__startup__")
            await self._plugin.stop()
            raise

    async def health(self) -> PluginHealth:
        return await self._plugin.health()

    async def stop(self) -> None:
        await self._runtime.terminate("__startup__")
        await self._plugin.stop()

    async def execute(
        self,
        request: PluginRequest,
        cancellation: CancellationToken,
        progress: ProgressReporter,
    ) -> PluginResult:
        return await self._plugin.execute(request, cancellation, progress)


def _generation(plugin_id: str):
    bootstrap = current_plugin_bootstrap(plugin_id)
    if bootstrap.accelerator_lease_path is None:
        raise RuntimeError("GPU accelerator grant is unavailable")
    model_id = QWEN_LARGE_ARTIFACT_ID if plugin_id in _LARGE_WORKLOADS else QWEN_ARTIFACT_ID
    model = bootstrap.require_artifact(model_id)
    qualification = load_qualification(
        plugin_id,
        model.id,
        model.sha256,
        _TASKS[plugin_id](),
    )
    store = MemoryFragmentStore()
    runtime = LlamaVulkanRuntime(store, bootstrap.accelerator_lease_path)
    descriptor = GenerationProviderDescriptor(
        provider_id="qwen3-workloads-gpu",
        accelerator="gpu",
        runtime="llama.cpp-vulkan",
        qualified=True,
        provenance=ArtifactProvenance(
            model_id,
            model.version,
            model.sha256,
            QWEN_LARGE_SOURCE if plugin_id in _LARGE_WORKLOADS else QWEN_SOURCE,
            "Apache-2.0",
        ),
    )
    worker = LlamaCppVulkanQwenWorker(descriptor, runtime, (model.path,))
    return bootstrap, store, runtime, model.path, qualification, GenerationRouter((worker,))


_TASKS = {
    "ask-selected-files": document_question_task,
    "event-extraction": event_generation_task,
    "file-organizer": file_organizer_task,
    "selected-text-tools": selected_text_task,
}


def create_event_extraction() -> QualifiedWorkload:
    bootstrap, store, runtime, model, qualification, router = _generation("event-extraction")
    if bootstrap.state_path is None:
        raise RuntimeError("event recovery state is unavailable")
    plugin = EventExtractionPlugin(
        router,
        store,
        EventRecoveryJournal(bootstrap.state_path / "event-recovery.json"),
    )
    return QualifiedWorkload(plugin, runtime, model, qualification)


def create_ask_selected_files() -> QualifiedWorkload:
    bootstrap, store, runtime, model, qualification, router = _generation("ask-selected-files")
    bge = bootstrap.require_artifact(BGE_ARTIFACT_ID)
    companions = dict(bge.companions)
    if set(companions) != {"model.bin", "tokenizer.json"}:
        raise RuntimeError("BGE companions are incomplete")
    embedder = BgeVulkanEmbedder(
        bge.path,
        bge.path.parent / "tokenizer.json",
        bge.sha256,
        bootstrap.accelerator_lease_path,
    )
    return QualifiedWorkload(
        DocumentQuestionPlugin(embedder, router, store),
        runtime,
        model,
        qualification,
        embedder,
    )


def create_selected_text_tools() -> QualifiedWorkload:
    _bootstrap, store, runtime, model, qualification, router = _generation("selected-text-tools")
    return QualifiedWorkload(SelectedTextPlugin(router, store), runtime, model, qualification)


def create_file_organizer() -> QualifiedWorkload:
    _bootstrap, store, runtime, model, qualification, router = _generation("file-organizer")
    return QualifiedWorkload(FileOrganizerPlugin(router, store), runtime, model, qualification)
