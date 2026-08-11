"""Offline, user-owned model training and accelerator packaging.

Nothing in this package is imported by the service.  Training dependencies,
local history, and compiler toolchains stay in a separate producer process;
the service receives only immutable artifacts and a schema-valid binding.
"""

from .clip_production import (
    ClipGateEvidence,
    ClipHoldout,
    ProducedClipSource,
    TorchScriptClipOnnxExporter,
    clip_resize_geometry,
    evaluate_clip_gate,
    normalize_clip_rgb,
    produce_clip_source,
)
from .compilers import (
    CompilationRequest,
    CompiledTarget,
    CompilerCapability,
    CompilerError,
    EdgeTpuTargetCompiler,
    NcnnTargetCompiler,
    OpenVinoTargetCompiler,
    TargetCompiler,
    default_target_compilers,
)
from .contracts import TrainingError, TrainingReport, TrainingSpec
from .embedding_production import (
    EmbeddingGateEvidence,
    EmbeddingHoldout,
    ProducedEmbeddingSource,
    SentenceEmbeddingOnnxExporter,
    evaluate_embedding_gate,
    pool_sentence_embedding,
    produce_sentence_embedding,
)
from .forecast import ForecastTrainer
from .foundation_forecast_production import (
    ForecastHoldout,
    FoundationForecastEvidence,
    ProducedFoundationForecast,
    TorchFoundationForecastOnnxExporter,
    evaluate_foundation_forecast,
    produce_foundation_forecast,
    select_median_quantile,
    select_point_forecast,
)
from .installation import InstalledTraining, install_training
from .numeric_promotion import (
    NativeParityEvidence,
    NumericTrainingReport,
    promote_numeric_training,
)

__all__ = [
    "ForecastTrainer",
    "CompilationRequest",
    "ClipGateEvidence",
    "ClipHoldout",
    "CompiledTarget",
    "CompilerCapability",
    "CompilerError",
    "EdgeTpuTargetCompiler",
    "EmbeddingGateEvidence",
    "EmbeddingHoldout",
    "ForecastHoldout",
    "FoundationForecastEvidence",
    "InstalledTraining",
    "NcnnTargetCompiler",
    "NativeParityEvidence",
    "NumericTrainingReport",
    "OpenVinoTargetCompiler",
    "ProducedClipSource",
    "ProducedEmbeddingSource",
    "ProducedFoundationForecast",
    "SentenceEmbeddingOnnxExporter",
    "TargetCompiler",
    "TorchScriptClipOnnxExporter",
    "TorchFoundationForecastOnnxExporter",
    "TrainingError",
    "TrainingReport",
    "TrainingSpec",
    "default_target_compilers",
    "clip_resize_geometry",
    "evaluate_clip_gate",
    "evaluate_embedding_gate",
    "evaluate_foundation_forecast",
    "install_training",
    "normalize_clip_rgb",
    "pool_sentence_embedding",
    "promote_numeric_training",
    "produce_clip_source",
    "produce_foundation_forecast",
    "produce_sentence_embedding",
    "select_median_quantile",
    "select_point_forecast",
]
