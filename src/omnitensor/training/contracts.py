"""Stable contracts shared by training recipes and artifact installers."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

from omnitensor.preparation import ArtifactReference, artifact_reference_error, file_digest

from ..atomicio import JsonTooLargeError, read_json_bounded, write_json_atomic
from ..registry import validate_document

TRAINING_REPORT_VERSION = 1
TRAINING_RECIPE = "forecast-v1"
TRAINING_REPORT_SCHEMA = "training-report.schema.json"
NUMERIC_TRAINING_REPORT_SCHEMA = "numeric-training-report.schema.json"
MAX_REPORT_BYTES = 64 * 1024
MAX_FEATURES = 128
MAX_FEATURE_NAME = 64
MAX_WINDOW = 128
MAX_INPUT_WIDTH = 512
_PROFILE_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class TrainingError(ValueError):
    """Stable training failure safe to show to a local operator."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def validate_training_report_document(document: object, *, numeric: bool = False) -> None:
    """Apply the canonical standalone schema to one training report."""
    schema = NUMERIC_TRAINING_REPORT_SCHEMA if numeric else TRAINING_REPORT_SCHEMA
    violations = validate_document(schema, document)
    if violations:
        label = "numeric training report" if numeric else "training report"
        raise TrainingError("report-invalid", f"{label} violates schema: {violations[0]}")


def write_training_report(
    path: Path | str,
    document: dict,
    *,
    prefix: str,
    numeric: bool = False,
) -> None:
    """Validate one report before publishing its unchanged compact JSON document."""
    validate_training_report_document(document, numeric=numeric)
    write_json_atomic(Path(path), document, prefix)


@dataclass(frozen=True, slots=True)
class TrainingSpec:
    """Identity and exact tensor shape of one local forecasting fit."""

    profile_id: str
    artifact_id: str
    artifact_version: str
    feature_names: tuple[str, ...]
    target_feature: str
    window: int
    horizon: int = 1

    def __post_init__(self) -> None:
        _validate_identity(self)
        _validate_features(self)
        _validate_bounds(self)

    @property
    def input_width(self) -> int:
        return len(self.feature_names) * self.window

    @property
    def tensor_contract(self) -> dict:
        return {
            "inputs": [
                {
                    "shape": [1, self.input_width],
                    "dtype": "float32",
                    "layout": "NC",
                }
            ]
        }

    @property
    def output_contract(self) -> dict:
        return {"kind": "raw"}

    @property
    def feature_contract(self) -> dict:
        """Machine-readable ordering and meaning of the flattened input."""
        return {
            "version": 1,
            "recipe": TRAINING_RECIPE,
            "featureNames": list(self.feature_names),
            "targetFeature": self.target_feature,
            "window": self.window,
            "horizon": self.horizon,
            "observationOrder": "oldest-first",
            "flattenOrder": "observations-then-features",
        }

    def document(self) -> dict:
        return {
            "profileId": self.profile_id,
            "artifactId": self.artifact_id,
            "artifactVersion": self.artifact_version,
            "featureNames": list(self.feature_names),
            "targetFeature": self.target_feature,
            "window": self.window,
            "horizon": self.horizon,
        }

    @classmethod
    def from_document(cls, document: object) -> TrainingSpec:
        if not isinstance(document, dict):
            raise TrainingError("report-invalid", "training spec must be an object")
        required = (
            "profileId",
            "artifactId",
            "artifactVersion",
            "featureNames",
            "targetFeature",
            "window",
            "horizon",
        )
        if any(name not in document for name in required) or not isinstance(
            document.get("featureNames"), list
        ):
            raise TrainingError("report-invalid", "training spec fields are invalid")
        return cls(
            document["profileId"],
            document["artifactId"],
            document["artifactVersion"],
            tuple(document["featureNames"]),
            document["targetFeature"],
            document["window"],
            document["horizon"],
        )


def _validate_identity(spec: TrainingSpec) -> None:
    invalid = artifact_reference_error(
        ArtifactReference(spec.artifact_id, spec.artifact_version, "onnx", "0" * 64)
    )
    if invalid or len(spec.artifact_id) > 116:
        invalid = invalid or "artifact id is too long for a target suffix"
        raise TrainingError("identity-invalid", invalid)
    if (
        not isinstance(spec.profile_id, str)
        or len(spec.profile_id) > 80
        or _PROFILE_ID.fullmatch(spec.profile_id) is None
    ):
        raise TrainingError("identity-invalid", "profile id is invalid")


def _validate_features(spec: TrainingSpec) -> None:
    if (
        not isinstance(spec.feature_names, tuple)
        or not 1 <= len(spec.feature_names) <= MAX_FEATURES
    ):
        raise TrainingError(
            "features-invalid", f"between 1 and {MAX_FEATURES} features are required"
        )
    try:
        unique_features = len(set(spec.feature_names))
    except TypeError as error:
        raise TrainingError(
            "features-invalid", "feature names must be bounded strings"
        ) from error
    if unique_features != len(spec.feature_names):
        raise TrainingError("features-invalid", "feature names must be unique")
    for name in spec.feature_names:
        if not isinstance(name, str) or not 1 <= len(name) <= MAX_FEATURE_NAME:
            raise TrainingError("features-invalid", "feature names must be bounded strings")
    if spec.target_feature != spec.feature_names[0]:
        raise TrainingError("features-invalid", "target feature must be the first feature")


def _validate_bounds(spec: TrainingSpec) -> None:
    for name, value in (("window", spec.window), ("horizon", spec.horizon)):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_WINDOW:
            raise TrainingError(
                "bounds-invalid", f"{name} must be an integer between 1 and {MAX_WINDOW}"
            )
    if spec.input_width > MAX_INPUT_WIDTH:
        raise TrainingError(
            "bounds-invalid", f"at most {MAX_INPUT_WIDTH} input values are supported"
        )


@dataclass(frozen=True, slots=True)
class TrainingReport:
    """Reproducible evidence connecting corpus, fit, and portable bytes."""

    spec: TrainingSpec
    samples: int
    corpus_sha256: str
    model_sha256: str
    quality: dict

    def __post_init__(self) -> None:
        if isinstance(self.samples, bool) or not isinstance(self.samples, int) or self.samples < 1:
            raise TrainingError("report-invalid", "sample count must be positive")
        for label, digest in (
            ("corpus", self.corpus_sha256),
            ("model", self.model_sha256),
        ):
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise TrainingError("report-invalid", f"{label} digest is invalid")
        if not isinstance(self.quality, dict) or not _quality_is_finite(self.quality):
            raise TrainingError("report-invalid", "quality metrics are invalid")

    def document(self) -> dict:
        return {
            "version": TRAINING_REPORT_VERSION,
            "recipe": TRAINING_RECIPE,
            "spec": self.spec.document(),
            "samples": self.samples,
            "corpusSha256": self.corpus_sha256,
            "model": {
                "format": "onnx",
                "filename": "model.onnx",
                "sha256": self.model_sha256,
            },
            "quality": self.quality,
            "tensorContract": self.spec.tensor_contract,
            "outputContract": self.spec.output_contract,
        }

    @classmethod
    def load(cls, path: Path | str) -> TrainingReport:
        report_path = Path(path)
        try:
            document = read_json_bounded(report_path, MAX_REPORT_BYTES)
        except (OSError, ValueError, JsonTooLargeError) as error:
            raise TrainingError(
                "report-invalid", f"cannot read training report: {error}"
            ) from error
        if not isinstance(document, dict):
            raise TrainingError("report-invalid", "training report fields are invalid")
        if (
            document.get("version") != TRAINING_REPORT_VERSION
            or document.get("recipe") != TRAINING_RECIPE
        ):
            raise TrainingError(
                "report-invalid", "training report version or recipe is unsupported"
            )
        spec = TrainingSpec.from_document(document.get("spec"))
        if document.get("tensorContract") != spec.tensor_contract:
            raise TrainingError("report-invalid", "tensor contract disagrees with training spec")
        if document.get("outputContract") != spec.output_contract:
            raise TrainingError("report-invalid", "output contract disagrees with training spec")
        model = document.get("model")
        if (
            not isinstance(model, dict)
            or model.get("format") != "onnx"
            or model.get("filename") != "model.onnx"
            or not isinstance(model.get("sha256"), str)
        ):
            raise TrainingError("report-invalid", "portable model declaration is invalid")
        validate_training_report_document(document)
        report = cls(
            spec,
            document["samples"],
            document["corpusSha256"],
            model["sha256"],
            document["quality"],
        )
        model_path = report_path.parent / "model.onnx"
        if not model_path.is_file() or file_digest(model_path) != report.model_sha256:
            raise TrainingError("model-invalid", "portable model does not match training report")
        return report


def _quality_is_finite(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return isinstance(value, (bool, str)) or value is None
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _quality_is_finite(item) for key, item in value.items())
    if isinstance(value, list):
        return all(_quality_is_finite(item) for item in value)
    return False
