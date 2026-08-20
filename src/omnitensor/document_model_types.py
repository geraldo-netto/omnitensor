"""Stable values and contracts for the document-model producer."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

PROFILE_ID = "document-intelligence"
RECIPE_ID = "bge-small-en-v1-5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
SEQUENCE_LENGTH = 128
MAX_TEXT_BYTES = 2 * 1024 * 1024
MAX_CORPUS_BYTES = 256 * 1024
MIN_NATIVE_COSINE = 0.999
MIN_RETRIEVAL_OVERLAP = 0.9
MAX_NATIVE_ABSOLUTE_ERROR = 0.001
DEFAULT_SOURCE_ROOT = "~/.local/share/omnitensor/model-sources"
DEFAULT_BUILD_ROOT = "~/.local/share/omnitensor/document-model-build"
DEFAULT_ARTIFACT_ROOT = "~/.local/share/omnitensor/artifacts"
DEFAULT_BINDINGS_ROOT = "~/.local/share/omnitensor/model-bindings"
DOCUMENT_MODEL_REPORT_SCHEMA = "document-model-report.schema.json"

_PORTABLE_INPUT_NAMES = ("input_ids", "attention_mask", "token_type_ids")
_NATIVE_DTYPES = {
    "input_ids": "int32",
    "attention_mask": "float32",
    "token_type_ids": "int32",
}


class DocumentModelError(ValueError):
    """Stable producer failure safe to show to a local operator."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class TokenizedText:
    """Three fixed tensors in the exact order the native graph declares."""

    input_ids: tuple[int, ...]
    attention_mask: tuple[float, ...]
    token_type_ids: tuple[int, ...]

    def inputs(self) -> list[list[list[int | float]]]:
        return [
            [list(self.input_ids)],
            [list(self.attention_mask)],
            [list(self.token_type_ids)],
        ]


@dataclass(frozen=True, slots=True)
class DocumentModelEvidence:
    """Portable/native and retrieval evidence from one named Vulkan device."""

    device_index: int
    device_name: str
    samples: int
    minimum_cosine_similarity: float
    minimum_top10_overlap: float
    maximum_absolute_error: float
    expected_top_hits: int


@dataclass(frozen=True, slots=True)
class InstalledDocumentModel:
    """Paths and evidence published by the one-shot producer."""

    artifact_path: Path
    binding_path: Path
    report_path: Path
    evidence: DocumentModelEvidence

    def document(self) -> dict:
        return {
            "profileId": PROFILE_ID,
            "artifactPath": str(self.artifact_path),
            "bindingPath": str(self.binding_path),
            "reportPath": str(self.report_path),
            "device": {
                "index": self.evidence.device_index,
                "name": self.evidence.device_name,
            },
            "minimumCosineSimilarity": self.evidence.minimum_cosine_similarity,
            "minimumTop10Overlap": self.evidence.minimum_top10_overlap,
            "maximumAbsoluteError": self.evidence.maximum_absolute_error,
            "restartRequired": True,
        }


def native_tensor_contract(recipe) -> dict:
    """Derive ncnn's fixed input ABI from the portable recipe contract.

    The recipe owns input order, shapes, layouts, and portable ONNX dtypes.
    The named conversion below is the explicit ncnn boundary: integer token
    tensors narrow to int32 and the additive attention mask becomes float32,
    matching the graph that was measured on Vulkan.
    """
    try:
        producer = recipe.producer
        contract = recipe.tensor_contract
        names = tuple(producer["inputNames"])
        inputs = contract["inputs"]
    except (AttributeError, KeyError, TypeError) as error:
        raise DocumentModelError(
            "recipe-incompatible", "recipe has no fixed BGE tensor contract"
        ) from error
    if names != _PORTABLE_INPUT_NAMES or not isinstance(inputs, list) or len(inputs) != 3:
        raise DocumentModelError("recipe-incompatible", "recipe has no fixed BGE tensor contract")
    converted = []
    for name, item in zip(names, inputs, strict=True):
        if (
            not isinstance(item, dict)
            or item.get("shape") != [1, SEQUENCE_LENGTH]
            or item.get("dtype") != "int64"
            or item.get("layout") != "NC"
        ):
            raise DocumentModelError(
                "recipe-incompatible", "recipe has no fixed BGE tensor contract"
            )
        converted.append(
            {
                "shape": copy.deepcopy(item["shape"]),
                "dtype": _NATIVE_DTYPES[name],
                "layout": item["layout"],
            }
        )
    return {"inputs": converted}
