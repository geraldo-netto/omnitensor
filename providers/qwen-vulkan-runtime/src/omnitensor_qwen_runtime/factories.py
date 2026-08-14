"""Cheap entry-point factories for the four independently identified wheels."""

from __future__ import annotations

from pathlib import Path

from omnitensor.plugins.acceptance_kit import validate_gpu_load
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
from omnitensor.plugins.qwen import LlamaCppVulkanQwenWorker
from omnitensor.plugins.selected_text import SelectedTextPlugin, selected_text_task
from omnitensor.sdk import current_plugin_bootstrap

from .bge import BgeVulkanEmbedder
from .hebrew import HebrewTranslationRuntime
from .qualification import (
    Qualification,
    load_model_qualification,
    load_qualification,
    verify_native_runtime,
)
from .runtime import LlamaVulkanRuntime

QWEN_ARTIFACT_ID = "qwen3-8b-q4-k-m"
BGE_ARTIFACT_ID = "bge-small-en-v1-5-ask-gpu"
HEBREW_ARTIFACT_ID = "dictalm2-hebrew-q4-k-m"
QWEN_SOURCE = (
    "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/"
    "7c41481f57cb95916b40956ab2f0b139b296d974/Qwen3-8B-Q4_K_M.gguf"
)
HEBREW_SOURCE = (
    "https://huggingface.co/dicta-il/dictalm2.0-instruct-GGUF/resolve/"
    "9ed3346a1440643825287abf86e28800005ec9b3/dictalm2.0-instruct-Q4_K_M.gguf"
)


class QualifiedWorkload:
    """Prove native GPU startup after handshake, then delegate the workload."""

    def __init__(
        self,
        plugin: WorkloadPlugin,
        runtime: LlamaVulkanRuntime,
        model_path: Path,
        qualification: Qualification,
        embedder: BgeVulkanEmbedder | None = None,
        additional_runtimes: tuple[
            tuple[LlamaVulkanRuntime, Path, Qualification], ...
        ] = (),
    ) -> None:
        self.plugin_id = plugin.plugin_id
        self._plugin = plugin
        self._runtime = runtime
        self._model_path = model_path
        self._qualification = qualification
        self._runtimes = ((runtime, model_path, qualification), *additional_runtimes)
        self._embedder = embedder

    async def start(self, context: PluginContext) -> None:
        await self._plugin.start(context)
        try:
            for runtime, model_path, qualification in self._runtimes:
                verify_native_runtime(qualification)
                report = await runtime.load((model_path,), "gpu")
                validate_gpu_load(report)
                if (
                    report.total_model_layers != qualification.model_layers
                    or runtime.physical_device != qualification.device
                ):
                    raise RuntimeError("native model load differs from qualification")
                await runtime.terminate("__startup__")
            if self._embedder is not None:
                await self._embedder.preflight()
        except BaseException:
            for runtime, _model_path, _qualification in self._runtimes:
                await runtime.terminate("__startup__")
            await self._plugin.stop()
            raise

    async def health(self) -> PluginHealth:
        return await self._plugin.health()

    async def stop(self) -> None:
        for runtime, _model_path, _qualification in self._runtimes:
            await runtime.terminate("__startup__")
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
    model = bootstrap.require_artifact(QWEN_ARTIFACT_ID)
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
            model.id,
            model.version,
            model.sha256,
            QWEN_SOURCE,
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
    bootstrap, store, runtime, model, qualification, router = _generation("selected-text-tools")
    hebrew_model = bootstrap.require_artifact(HEBREW_ARTIFACT_ID)
    hebrew_qualification = load_model_qualification(
        hebrew_model.id, hebrew_model.sha256
    )
    assert bootstrap.accelerator_lease_path is not None
    hebrew_runtime = HebrewTranslationRuntime(store, bootstrap.accelerator_lease_path)
    hebrew_descriptor = GenerationProviderDescriptor(
        provider_id="dictalm2-hebrew-gpu",
        accelerator="gpu",
        runtime="llama.cpp-vulkan",
        qualified=True,
        provenance=ArtifactProvenance(
            hebrew_model.id,
            hebrew_model.version,
            hebrew_model.sha256,
            HEBREW_SOURCE,
            "Apache-2.0",
        ),
    )
    hebrew_worker = LlamaCppVulkanQwenWorker(
        hebrew_descriptor, hebrew_runtime, (hebrew_model.path,)
    )
    plugin = SelectedTextPlugin(
        router,
        store,
        translation_routes={"Hebrew": GenerationRouter((hebrew_worker,))},
    )
    return QualifiedWorkload(
        plugin,
        runtime,
        model,
        qualification,
        additional_runtimes=((hebrew_runtime, hebrew_model.path, hebrew_qualification),),
    )


def create_file_organizer() -> QualifiedWorkload:
    _bootstrap, store, runtime, model, qualification, router = _generation("file-organizer")
    return QualifiedWorkload(FileOrganizerPlugin(router, store), runtime, model, qualification)
