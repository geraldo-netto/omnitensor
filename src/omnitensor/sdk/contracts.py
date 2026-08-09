"""Public extension contracts that do not expose service implementation types."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..plugins.artifacts import ArtifactReference, ArtifactResolution
from ..plugins.pipeline import (
    CollectedOutput,
    DeliveredOutput,
    InferenceOutput,
    PostprocessedOutput,
    PreprocessedOutput,
)
from ..plugins.protocol import CancellationToken, PluginProgress, PluginResult


@runtime_checkable
class ArtifactProvider(Protocol):
    """Resolve one declared immutable artifact without revealing host roots."""

    def resolve(self, reference: ArtifactReference) -> ArtifactResolution: ...


@runtime_checkable
class PipelineHooks(Protocol):
    """Plugin-owned transformations around host-owned artifact inference."""

    async def preprocess(
        self,
        collected: CollectedOutput,
        cancellation: CancellationToken,
    ) -> PreprocessedOutput: ...

    async def postprocess(
        self,
        inference: InferenceOutput,
        cancellation: CancellationToken,
    ) -> PostprocessedOutput: ...

    async def deliver(
        self,
        output: PostprocessedOutput,
        cancellation: CancellationToken,
    ) -> DeliveredOutput: ...


@runtime_checkable
class ResultSink(Protocol):
    """Authorized terminal result consumer supplied by the host."""

    async def publish(self, result: PluginResult) -> None: ...


@runtime_checkable
class ProgressSink(Protocol):
    """Authorized progress consumer supplied by the host."""

    async def publish(self, progress: PluginProgress) -> None: ...
