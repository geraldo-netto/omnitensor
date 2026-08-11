"""Offline, user-owned model training and accelerator packaging.

Nothing in this package is imported by the service.  Training dependencies,
local history, and compiler toolchains stay in a separate producer process;
the service receives only immutable artifacts and a schema-valid binding.
"""

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
from .installation import InstalledTraining, install_training
from .numeric_promotion import (
    NativeParityEvidence,
    NumericTrainingReport,
    promote_numeric_training,
)

__all__ = [
    "ForecastTrainer",
    "CompilationRequest",
    "CompiledTarget",
    "CompilerCapability",
    "CompilerError",
    "EdgeTpuTargetCompiler",
    "EmbeddingGateEvidence",
    "EmbeddingHoldout",
    "InstalledTraining",
    "NcnnTargetCompiler",
    "NativeParityEvidence",
    "NumericTrainingReport",
    "OpenVinoTargetCompiler",
    "ProducedEmbeddingSource",
    "SentenceEmbeddingOnnxExporter",
    "TargetCompiler",
    "TrainingError",
    "TrainingReport",
    "TrainingSpec",
    "default_target_compilers",
    "evaluate_embedding_gate",
    "install_training",
    "pool_sentence_embedding",
    "promote_numeric_training",
    "produce_sentence_embedding",
]
