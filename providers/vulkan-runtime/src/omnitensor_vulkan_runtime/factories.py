"""Cheap entry-point factories for the four independently identified wheels."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

from omnitensor.atomicio import remove_durable, write_json_atomic
from omnitensor.plugins.acceptance_kit import validate_gpu_load
from omnitensor.plugins.document_qa import DocumentQuestionPlugin, document_question_task
from omnitensor.plugins.event_workload import (
    EventExtractionPlugin,
    EventRecoveryJournal,
    event_generation_task,
)
from omnitensor.plugins.fragments import MemoryFragmentStore
from omnitensor.plugins.file_organizer import FileOrganizerPlugin, file_organizer_task
from omnitensor.plugins.generation import (
    ArtifactProvenance,
    GenerationProviderDescriptor,
    GenerationRouter,
    GenerationTask,
)
from omnitensor.plugins.generation_workers import LlamaCppVulkanWorker
from omnitensor.plugins.protocol import (
    CancellationToken,
    PluginContext,
    PluginHealth,
    PluginRequest,
    PluginResult,
    ProgressReporter,
    WorkloadPlugin,
)
from omnitensor.plugins.selected_text import SelectedTextPlugin, selected_text_task
from omnitensor.plugins.selected_text_acceptance import (
    SELECTED_TEXT_WORKER_LOAD_RECEIPT,
    ModelEvidence,
    build_selected_text_worker_load_receipt,
)
from omnitensor.sdk import current_plugin_bootstrap

from .bge import BgeVulkanEmbedder
from .hebrew import HebrewTranslationRuntime
from .qualification import (
    Qualification,
    default_model,
    load_model_qualification,
    load_qualification,
    verify_native_runtime,
)
from .runtime import LlamaVulkanRuntime

# The model this provider shipped with. It is no longer the default —
# `default_model()` reads that from the receipt — and remains only because
# the installer pins this exact file.
SHIPPED_ARTIFACT_ID = "qwen3-8b-q4-k-m"
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
        additional_runtimes: tuple[tuple[LlamaVulkanRuntime, Path, Qualification], ...] = (),
        load_receipt_path: Path | None = None,
        load_receipt_models: tuple[tuple[str, str], ...] = (),
    ) -> None:
        runtimes = ((runtime, model_path, qualification), *additional_runtimes)
        if (load_receipt_path is None) == bool(load_receipt_models) or (
            load_receipt_path is not None
            and (
                tuple(role for role, _digest in load_receipt_models)
                != ("primary", "hebrewTranslation")
                or len(load_receipt_models) != len(runtimes)
            )
        ):
            raise ValueError("selected-text load receipt configuration is invalid")
        self.plugin_id = plugin.plugin_id
        self._plugin = plugin
        self._runtime = runtime
        self._model_path = model_path
        self._qualification = qualification
        self._runtimes = runtimes
        self._embedder = embedder
        self._load_receipt_path = load_receipt_path
        self._load_receipt_models = load_receipt_models

    async def start(self, context: PluginContext) -> None:
        self._clear_load_receipt()
        await self._plugin.start(context)
        measured: list[ModelEvidence] = []
        try:
            for index, (runtime, model_path, qualification) in enumerate(self._runtimes):
                runtime_version = verify_native_runtime(qualification)
                report = await runtime.load((model_path,), "gpu")
                validate_gpu_load(report)
                if (
                    report.total_model_layers != qualification.model_layers
                    or runtime.physical_device != qualification.device
                ):
                    raise RuntimeError("native model load differs from qualification")
                if self._load_receipt_path is not None:
                    _role, digest = self._load_receipt_models[index]
                    measured.append(
                        ModelEvidence(digest, runtime_version, runtime.physical_device, report)
                    )
                await runtime.terminate("__startup__")
            if self._embedder is not None:
                await self._embedder.preflight()
            self._publish_load_receipt(measured)
        except BaseException:
            await self._cleanup_failed_start()
            raise

    def _clear_load_receipt(self) -> None:
        if self._load_receipt_path is not None:
            remove_durable(self._load_receipt_path)

    def _publish_load_receipt(self, measured: list[ModelEvidence]) -> None:
        if self._load_receipt_path is None:
            return
        receipt = build_selected_text_worker_load_receipt(*measured)
        write_json_atomic(
            self._load_receipt_path,
            receipt.document(),
            prefix=".selected-text-worker-load-",
        )

    async def _cleanup_failed_start(self) -> None:
        await self._cleanup(clear_receipt_first=False)

    async def health(self) -> PluginHealth:
        return await self._plugin.health()

    async def stop(self) -> None:
        failure = await self._cleanup(clear_receipt_first=True)
        if failure is not None:
            raise failure

    async def _cleanup(self, *, clear_receipt_first: bool) -> BaseException | None:
        failure = None
        if clear_receipt_first:
            failure = _capture_sync_failure(self._clear_load_receipt, failure)
        for runtime, _model_path, _qualification in self._runtimes:
            failure = await _capture_async_failure(runtime.terminate("__startup__"), failure)
        failure = await _capture_async_failure(self._plugin.stop(), failure)
        if not clear_receipt_first:
            failure = _capture_sync_failure(self._clear_load_receipt, failure)
        return failure

    async def execute(
        self,
        request: PluginRequest,
        cancellation: CancellationToken,
        progress: ProgressReporter,
    ) -> PluginResult:
        return await self._plugin.execute(request, cancellation, progress)


def _capture_sync_failure(
    operation: Callable[[], None], failure: BaseException | None
) -> BaseException | None:
    try:
        operation()
    except BaseException as error:
        return failure if failure is not None else error
    return failure


async def _capture_async_failure(
    operation: Awaitable[object], failure: BaseException | None
) -> BaseException | None:
    try:
        await operation
    except BaseException as error:
        return failure if failure is not None else error
    return failure


def _generation(plugin_id: str):
    bootstrap = current_plugin_bootstrap(plugin_id)
    if bootstrap.accelerator_lease_path is None:
        raise RuntimeError("GPU accelerator grant is unavailable")
    # The model a person chose, or this workload's default. The default is read
    # from the receipt rather than kept as a constant here: a second copy of it
    # went stale the day the default moved to the 9B, and nothing would have
    # said so. Its receipt is verified for *this* workload below, so an
    # unqualified pair refuses to start rather than quietly answering with
    # something nobody measured.
    model = bootstrap.chosen_or(default_model(plugin_id))
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
    worker = LlamaCppVulkanWorker(descriptor, runtime, (model.path,))
    return bootstrap, store, runtime, model, qualification, GenerationRouter((worker,))


_TASKS = {
    "ask-selected-files": document_question_task,
    "event-extraction": event_generation_task,
    "file-organizer": file_organizer_task,
    "selected-text-tools": selected_text_task,
}


def workload_tasks() -> dict[str, Callable[[], GenerationTask]]:
    """The task each workload runs, by workload id.

    Published because something outside this distribution needs it: the
    benchmark measures the tasks that actually run, and was reaching into
    `_TASKS` — an underscore name this package never promised, which a patch
    release could have renamed without breaking any stated contract.

    A copy, so a caller cannot alter what the workloads run by editing what it
    was handed.
    """
    return dict(_TASKS)


def create_event_extraction() -> QualifiedWorkload:
    bootstrap, store, runtime, model, qualification, router = _generation("event-extraction")
    if bootstrap.state_path is None:
        raise RuntimeError("event recovery state is unavailable")
    plugin = EventExtractionPlugin(
        router,
        store,
        EventRecoveryJournal(bootstrap.state_path / "event-recovery.json"),
    )
    return QualifiedWorkload(plugin, runtime, model.path, qualification)


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
        model.path,
        qualification,
        embedder,
    )


def create_selected_text_tools() -> QualifiedWorkload:
    bootstrap, store, runtime, model, qualification, router = _generation("selected-text-tools")
    if bootstrap.state_path is None:
        raise RuntimeError("selected-text load receipt state is unavailable")
    hebrew_model = bootstrap.require_artifact(HEBREW_ARTIFACT_ID)
    hebrew_qualification = load_model_qualification(hebrew_model.id, hebrew_model.sha256)
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
    hebrew_worker = LlamaCppVulkanWorker(hebrew_descriptor, hebrew_runtime, (hebrew_model.path,))
    plugin = SelectedTextPlugin(
        router,
        store,
        translation_routes={"Hebrew": GenerationRouter((hebrew_worker,))},
    )
    return QualifiedWorkload(
        plugin,
        runtime,
        model.path,
        qualification,
        additional_runtimes=((hebrew_runtime, hebrew_model.path, hebrew_qualification),),
        load_receipt_path=bootstrap.state_path / SELECTED_TEXT_WORKER_LOAD_RECEIPT,
        load_receipt_models=(
            ("primary", model.sha256),
            ("hebrewTranslation", hebrew_model.sha256),
        ),
    )


def create_file_organizer() -> QualifiedWorkload:
    _bootstrap, store, runtime, model, qualification, router = _generation("file-organizer")
    return QualifiedWorkload(FileOrganizerPlugin(router, store), runtime, model.path, qualification)
