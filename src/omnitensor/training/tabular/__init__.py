"""Shared deterministic mechanics for bounded numeric model training."""

from .fit import (
    binary_auc,
    binary_class_counts,
    fit_balanced_logistic,
    fit_output,
    normalization,
    normalized_features,
    population_normalization,
    ranked_auc,
    require_binary_class_floor,
    stable_sigmoid,
    unit_interval,
)
from .jsonl import JsonlPolicy, load_bounded_jsonl, parse_jsonl_line
from .model import (
    BuildAdvisorModel,
    BuildExample,
    NormalizedLogisticModel,
    TabularExample,
)
from .onnx import (
    checked_model,
    load_onnx_dependency,
    normalized_logistic_model,
    save_onnx_atomic,
)
from .report import numeric_training_report, publish_numeric_training
from .split import chronological_split, floor_chronological_split

__all__ = [
    "BuildAdvisorModel",
    "BuildExample",
    "JsonlPolicy",
    "NormalizedLogisticModel",
    "TabularExample",
    "binary_auc",
    "binary_class_counts",
    "checked_model",
    "chronological_split",
    "fit_balanced_logistic",
    "fit_output",
    "floor_chronological_split",
    "load_bounded_jsonl",
    "load_onnx_dependency",
    "normalization",
    "normalized_features",
    "normalized_logistic_model",
    "numeric_training_report",
    "parse_jsonl_line",
    "population_normalization",
    "publish_numeric_training",
    "ranked_auc",
    "require_binary_class_floor",
    "save_onnx_atomic",
    "stable_sigmoid",
    "unit_interval",
]
