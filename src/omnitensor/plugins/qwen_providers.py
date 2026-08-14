"""Fail-closed native Qwen generation workers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from ..preparation import file_digest
from .acceptance_kit import NativeLoadReport, validate_gpu_load, validate_npu_load
from .generation import (
    GenerationProviderDescriptor,
    GenerationRequest,
    GenerationTask,
    ProviderGenerationError,
)
from .protocol import CancellationToken, ProgressReporter
from .qwen_contracts import NativeQwenRuntime, QwenProviderError, legacy_qwen_callback

_LEGACY_MODULE = "omnitensor.plugins.qwen"


class _QwenWorker:
    accelerator: str
    runtime_name: str

    def __init__(
        self,
        descriptor: GenerationProviderDescriptor,
        runtime: NativeQwenRuntime,
        artifacts: Sequence[Path | str],
        *,
        companion_sha256: Mapping[str, str] | None = None,
    ) -> None:
        if not isinstance(runtime, NativeQwenRuntime):
            raise QwenProviderError("runtime-invalid", "native runtime does not implement its port")
        if descriptor.accelerator != self.accelerator or descriptor.runtime != self.runtime_name:
            raise QwenProviderError("provider-invalid", "provider descriptor names another lane")
        if not descriptor.qualified:
            raise QwenProviderError("provider-unqualified", "provider has no accepted evidence")
        paths = tuple(Path(path) for path in artifacts)
        if not paths:
            raise QwenProviderError("artifact-invalid", "at least one model artifact is required")
        self._descriptor = descriptor
        self._runtime = runtime
        self._artifacts = paths
        self._companion_sha256 = dict(companion_sha256 or {})
        self._load_report: NativeLoadReport | None = None

    @property
    def descriptor(self) -> GenerationProviderDescriptor:
        return self._descriptor

    def preflight(self) -> None:
        """Verify pinned local model bytes without loading a native runtime."""
        self._verify_artifacts()

    async def generate(
        self,
        task: GenerationTask,
        request: GenerationRequest,
        cancellation: CancellationToken,
        progress: ProgressReporter,
    ) -> str:
        cancellation.raise_if_cancelled()
        if self._load_report is None:
            self._verify_artifacts()
            try:
                report = await self._runtime.load(self._artifacts, self.accelerator)
            except ProviderGenerationError:
                raise
            except Exception as error:
                raise ProviderGenerationError(
                    "model-load-failed", "native model loading failed", generation_started=False
                ) from error
            self._validate_load(report)
            self._load_report = report
        cancellation.raise_if_cancelled()
        try:
            return await self._runtime.generate(task, request, cancellation, progress)
        except ProviderGenerationError as error:
            if error.generation_started:
                await self.terminate(request.request_id)
            raise

    async def terminate(self, request_id: str) -> None:
        self._load_report = None
        await self._runtime.terminate(request_id)

    def _verify_artifacts(self) -> None:
        primary = self._artifacts[0]
        if not primary.is_file() or file_digest(primary) != self._descriptor.provenance.sha256:
            raise ProviderGenerationError(
                "model-load-failed",
                "primary model artifact is absent or changed",
                generation_started=False,
            )
        expected_names = set(self._companion_sha256)
        actual_names = {path.name for path in self._artifacts[1:]}
        if actual_names != expected_names:
            raise ProviderGenerationError(
                "model-load-failed", "model companion set is incomplete", generation_started=False
            )
        for path in self._artifacts[1:]:
            if not path.is_file() or file_digest(path) != self._companion_sha256[path.name]:
                raise ProviderGenerationError(
                    "model-load-failed",
                    "model companion is absent or changed",
                    generation_started=False,
                )

    def _validate_load(self, report: NativeLoadReport) -> None:
        raise NotImplementedError


class LlamaCppVulkanQwenWorker(_QwenWorker):
    """Qwen GGUF provider that requires every model layer on Vulkan."""

    accelerator = "gpu"
    runtime_name = "llama.cpp-vulkan"

    def _validate_load(self, report: NativeLoadReport) -> None:
        legacy_qwen_callback("_validate_gpu_load", validate_gpu_load)(report)


class OpenVinoNpuQwenWorker(_QwenWorker):
    """Explicit local OpenVINO GenAI provider that must stay on NPU."""

    accelerator = "npu"
    runtime_name = "openvino-genai-npu"

    def __init__(self, *args, explicitly_enabled: bool = False, **kwargs) -> None:
        if not explicitly_enabled:
            raise QwenProviderError(
                "provider-disabled", "NPU generation needs explicit local configuration"
            )
        super().__init__(*args, **kwargs)

    def _validate_load(self, report: NativeLoadReport) -> None:
        legacy_qwen_callback("_validate_npu_load", validate_npu_load)(report)


for _legacy_type in (_QwenWorker, LlamaCppVulkanQwenWorker, OpenVinoNpuQwenWorker):
    _legacy_type.__module__ = _LEGACY_MODULE


__all__ = [
    "LlamaCppVulkanQwenWorker",
    "OpenVinoNpuQwenWorker",
]
