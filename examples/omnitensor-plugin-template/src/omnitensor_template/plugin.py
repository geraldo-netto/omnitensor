"""Minimal bounded plugin built only on the public OmniTensor SDK."""

from __future__ import annotations

import math
from collections.abc import Mapping

from omnitensor.sdk import (
    CancellationToken,
    CollectedOutput,
    CollectorReadiness,
    ConfigurationView,
    DeliveredOutput,
    InferenceOutput,
    ManagedPlugin,
    PluginHealth,
    PluginHealthStatus,
    PluginProgress,
    PluginRequest,
    PluginResult,
    PostprocessedOutput,
    PreprocessedOutput,
    ProgressReporter,
    SDKContractError,
    SourceStatus,
    Trigger,
    TriggerKind,
    cancelled_result,
    succeeded_result,
)

MAX_VALUES = 32
MAX_ABSOLUTE_VALUE = 1_000_000.0


class TemplateCollector:
    """Collect one bounded numeric vector from a manual trigger."""

    async def readiness(self) -> CollectorReadiness:
        return CollectorReadiness(SourceStatus.READY, "manual input ready", 0)

    async def collect(self, trigger: Trigger) -> CollectedOutput:
        return CollectedOutput({"values": _values(trigger.payload)})


class TemplatePipeline:
    """Demonstrate typed preprocessing/postprocessing without host imports."""

    def __init__(self, scale: float) -> None:
        if isinstance(scale, bool) or not isinstance(scale, (int, float)):
            raise SDKContractError("invalid-scale", "scale must be numeric")
        if not math.isfinite(scale) or scale <= 0 or scale > 1000:
            raise SDKContractError("invalid-scale", "scale must be finite and in (0, 1000]")
        self._scale = float(scale)

    async def preprocess(
        self,
        collected: CollectedOutput,
        cancellation: CancellationToken,
    ) -> PreprocessedOutput:
        cancellation.raise_if_cancelled()
        values = _values(collected.payload)
        mean = sum(values) / len(values)
        return PreprocessedOutput({"mean": mean / self._scale}, {"count": len(values)})

    async def postprocess(
        self,
        inference: InferenceOutput,
        cancellation: CancellationToken,
    ) -> PostprocessedOutput:
        cancellation.raise_if_cancelled()
        score = inference.tensors.get("score")
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(score)
            or score < 0
            or score > 1
        ):
            raise SDKContractError("invalid-score", "score must be finite and in [0, 1]")
        return PostprocessedOutput({"score": float(score)})


class TemplateResultConsumer:
    """Bounded no-side-effect consumer used by the template and its tests."""

    async def deliver(
        self,
        output: PostprocessedOutput,
        cancellation: CancellationToken,
    ) -> DeliveredOutput:
        cancellation.raise_if_cancelled()
        return DeliveredOutput(output.output, "template result accepted")


class TemplatePlugin(ManagedPlugin):
    """Executable example entry point; production models use host inference."""

    plugin_id = "template-workload"

    async def on_health(self) -> PluginHealth:
        return PluginHealth(PluginHealthStatus.READY, "template ready", 0)

    async def execute(
        self,
        request: PluginRequest,
        cancellation: CancellationToken,
        progress: ProgressReporter,
    ) -> PluginResult:
        if cancellation.cancelled:
            return cancelled_result(
                request,
                "cancelled before execution",
                completed_at_ms=request.submitted_at_ms,
            )
        cancellation.raise_if_cancelled()
        scale = ConfigurationView(self.context.configuration).optional("scale", object, 1.0)
        pipeline = TemplatePipeline(scale)
        collector = TemplateCollector()
        consumer = TemplateResultConsumer()
        trigger = Trigger(
            request.plugin_id,
            request.job_id,
            TriggerKind(request.trigger),
            request.payload,
            request.submitted_at_ms,
        )
        await progress.report(
            PluginProgress(request.job_id, "collect", 0.0, "", request.submitted_at_ms)
        )
        collected = await collector.collect(trigger)
        await progress.report(
            PluginProgress(request.job_id, "preprocess", 0.4, "", request.submitted_at_ms)
        )
        preprocessed = await pipeline.preprocess(collected, cancellation)
        normalized = preprocessed.tensors["mean"]
        score = max(0.0, min(1.0, (float(normalized) + 1.0) / 2.0))
        postprocessed = await pipeline.postprocess(
            InferenceOutput({"score": score}),
            cancellation,
        )
        await progress.report(
            PluginProgress(request.job_id, "deliver", 0.8, "", request.submitted_at_ms)
        )
        delivered = await consumer.deliver(postprocessed, cancellation)
        await progress.report(
            PluginProgress(request.job_id, "terminal", 1.0, "", request.submitted_at_ms)
        )
        return succeeded_result(
            request,
            delivered.output,
            detail=delivered.detail,
            completed_at_ms=request.submitted_at_ms,
        )


def _values(payload: Mapping) -> list[float]:
    values = payload.get("values")
    if not isinstance(values, list) or not 1 <= len(values) <= MAX_VALUES:
        raise SDKContractError("invalid-values", f"values must contain 1-{MAX_VALUES} items")
    converted = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or abs(value) > MAX_ABSOLUTE_VALUE
        ):
            raise SDKContractError(
                "invalid-values",
                f"values must be finite numbers within ±{int(MAX_ABSOLUTE_VALUE)}",
            )
        converted.append(float(value))
    return converted
