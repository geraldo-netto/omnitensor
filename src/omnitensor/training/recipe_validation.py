"""Cross-field recipe validation and producer-kind policy registry."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType

import jsonschema

from ..registry import workload_model_contract_schemas
from .recipe_model import ModelRecipeError
from .recipe_uri_policy import revision_is_path_segment, validate_https_uri

MAX_TOTAL_SOURCE_BYTES = 4 * 1024 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ProducerValidator:
    contract: Callable[[dict], None]
    semantics: Callable[[dict, dict, list], None]


def validate_recipe_semantics(
    document: dict,
    *,
    maximum_total_source_bytes: int = MAX_TOTAL_SOURCE_BYTES,
) -> None:
    sources = document["sources"]
    roles = validate_source_inventory(
        sources,
        maximum_total_source_bytes=maximum_total_source_bytes,
    )
    for source in sources:
        validate_https_uri(source["uri"], "source URI")
        if not revision_is_path_segment(source["uri"], source["revision"]):
            raise ModelRecipeError(
                "recipe-invalid", f"source URI does not pin revision {source['revision']}"
            )
    validate_https_uri(document["license"]["termsUri"], "license terms URI")
    validate_preprocessing_sources(document, roles)
    validate_contracts(document)
    validate_producer(document)
    if not finite_json(document["evaluation"]):
        raise ModelRecipeError("recipe-invalid", "evaluation metrics must be finite")
    tpu = document["targets"]["tpu"]
    if tpu["status"] == "validated" and not tpu["fullyQuantized"]:
        raise ModelRecipeError(
            "recipe-invalid", "validated TPU compatibility requires full quantization"
        )


def validate_source_inventory(
    sources: list[dict],
    *,
    maximum_total_source_bytes: int = MAX_TOTAL_SOURCE_BYTES,
) -> tuple[str, ...]:
    roles = [source["role"] for source in sources]
    filenames = [source["filename"] for source in sources]
    if roles.count("model") != 1:
        raise ModelRecipeError("recipe-invalid", "exactly one model source is required")
    if len(set(filenames)) != len(filenames):
        raise ModelRecipeError("recipe-invalid", "source filenames must be unique")
    if len(set(roles)) != len(roles):
        raise ModelRecipeError("recipe-invalid", "source roles must be unique")
    total = sum(source["sizeBytes"] for source in sources)
    if total > maximum_total_source_bytes:
        raise ModelRecipeError("recipe-invalid", "combined source size exceeds 4 GiB")
    return tuple(roles)


def validate_preprocessing_sources(document: dict, roles: tuple[str, ...]) -> None:
    required_sources = set((document.get("preprocessing") or {}).get("artifacts", ()))
    missing_sources = sorted(required_sources - set(roles))
    if missing_sources:
        raise ModelRecipeError(
            "recipe-invalid",
            f"preprocessing artifact has no pinned source: {missing_sources[0]}",
        )


def validate_contracts(document: dict) -> None:
    contract_schemas = workload_model_contract_schemas()
    for field in ("tensorContract", "outputContract"):
        violations = sorted(
            jsonschema.Draft202012Validator(contract_schemas[field]).iter_errors(document[field]),
            key=str,
        )
        if violations:
            detail = "; ".join(
                f"{'/'.join(str(part) for part in error.absolute_path) or '/'}: {error.message}"
                for error in violations
            )
            raise ModelRecipeError(
                "recipe-invalid", f"model contract is invalid at {field}: {detail}"
            )


def validate_producer(document: dict) -> None:
    producer = document.get("producer")
    if producer is None:
        return
    inputs = document["tensorContract"]["inputs"]
    if len(producer["inputNames"]) != len(inputs):
        raise ModelRecipeError(
            "recipe-invalid", "producer inputNames must match tensorContract input order"
        )
    validator = PRODUCER_VALIDATORS[producer["kind"]]
    validator.contract(document)
    if producer["outputShape"][0] != 1:
        raise ModelRecipeError("recipe-invalid", "producer outputShape must have batch size 1")
    validator.semantics(document, producer, inputs)


def _accept_any_contract(_document: dict) -> None:
    return None


def _embedding_contract(document: dict) -> None:
    if document["outputContract"]["kind"] != "embedding":
        raise ModelRecipeError(
            "recipe-invalid", "embedding producers require an embedding output contract"
        )


def _retinexformer_contract(document: dict) -> None:
    if document["family"] != "low-light" or document["outputContract"]["kind"] != "raw":
        raise ModelRecipeError(
            "recipe-invalid", "image enhancement producers require a raw low-light contract"
        )


def _timeseries_contract(document: dict) -> None:
    if document["family"] != "forecast" or document["outputContract"]["kind"] != "raw":
        raise ModelRecipeError(
            "recipe-invalid", "time-series producers require a raw forecast output contract"
        )


def _identity_producer(_document: dict, producer: dict, _inputs: list) -> None:
    if producer["sourceOutputShape"] != producer["outputShape"]:
        raise ModelRecipeError(
            "recipe-invalid", "identity producer cannot change the source output shape"
        )


def _sentence_embedding_producer(document: dict, producer: dict, inputs: list) -> None:
    expected_names = ["input_ids", "attention_mask", "token_type_ids"]
    if document["sourceFormat"] != "onnx" or producer["inputNames"] != expected_names:
        raise ModelRecipeError(
            "recipe-invalid", "sentence embedding source must be ONNX with canonical input order"
        )
    if producer["sourceOutputName"] != "last_hidden_state":
        raise ModelRecipeError(
            "recipe-invalid", "sentence embedding source output must be last_hidden_state"
        )
    shapes = [item["shape"] for item in inputs]
    if len(inputs) != 3 or len({tuple(shape) for shape in shapes}) != 1:
        raise ModelRecipeError(
            "recipe-invalid", "sentence embedding inputs must share one fixed token shape"
        )
    if any(item["dtype"] != "int64" for item in inputs):
        raise ModelRecipeError("recipe-invalid", "sentence embedding inputs must be int64")
    batch, sequence = shapes[0]
    source_shape = producer["sourceOutputShape"]
    output_shape = producer["outputShape"]
    if (
        producer["postprocessing"] not in {"attention-mask-mean-pool-l2", "cls-token-l2"}
        or source_shape[:2] != [batch, sequence]
        or len(source_shape) != 3
        or output_shape != [batch, source_shape[2]]
    ):
        raise ModelRecipeError(
            "recipe-invalid", "sentence embedding pooling shapes or postprocessing disagree"
        )


def _clip_image_producer(document: dict, producer: dict, inputs: list) -> None:
    if document["sourceFormat"] != "torchscript" or producer["inputNames"] != ["image"]:
        raise ModelRecipeError(
            "recipe-invalid", "CLIP image embedding source must be TorchScript with image input"
        )
    if len(inputs) != 1 or inputs[0]["shape"] != [1, 3, 224, 224]:
        raise ModelRecipeError(
            "recipe-invalid", "CLIP image embedding input must be fixed NCHW 224x224"
        )
    if (
        producer["sourceOutputName"] != "encode_image"
        or producer["postprocessing"] != "l2-normalize"
        or producer["sourceOutputShape"] != producer["outputShape"]
    ):
        raise ModelRecipeError(
            "recipe-invalid", "CLIP image output must preserve and L2-normalize encode_image"
        )


def _retinexformer_producer(document: dict, producer: dict, inputs: list) -> None:
    roles = {source["role"] for source in document["sources"]}
    if document["sourceFormat"] != "pytorch-state-dict" or not {
        "architecture",
        "config",
    }.issubset(roles):
        raise ModelRecipeError(
            "recipe-invalid",
            "Retinexformer requires a state dict plus pinned architecture and config",
        )
    expected_input = {
        "shape": [1, 3, 256, 256],
        "dtype": "float32",
        "layout": "NCHW",
        "preprocess": {
            "channelOrder": "RGB",
            "mean": [0, 0, 0],
            "scale": [1 / 255, 1 / 255, 1 / 255],
            "resize": {"filter": "bicubic", "fit": "exact"},
        },
    }
    if producer["inputNames"] != ["image"] or inputs != [expected_input]:
        raise ModelRecipeError(
            "recipe-invalid", "Retinexformer input must be fixed RGB NCHW 256x256"
        )
    expected_shape = [1, 3, 256, 256]
    if (
        producer["sourceOutputName"] != "enhanced"
        or producer["sourceOutputShape"] != expected_shape
        or producer["outputShape"] != expected_shape
        or producer["postprocessing"] != "clamp-unit-range"
    ):
        raise ModelRecipeError(
            "recipe-invalid", "Retinexformer export must clamp its fixed enhanced image"
        )


def _timeseries_producer(document: dict, producer: dict, inputs: list) -> None:
    if document["sourceFormat"] != "safetensors" or producer["inputNames"] != ["context"]:
        raise ModelRecipeError(
            "recipe-invalid", "pretrained time-series source must be safetensors with context input"
        )
    expected_input = {"shape": [1, 512], "dtype": "float32", "layout": "NC"}
    if inputs != [expected_input] or producer["outputShape"] != [1, 1]:
        raise ModelRecipeError(
            "recipe-invalid", "time-series export must map one 512-value context to one scalar"
        )
    if producer["kind"] == "timeseries-point-forecast":
        valid = (
            producer["sourceOutputName"] == "prediction_outputs"
            and producer["sourceOutputShape"] == [1, 96, 1]
            and producer["postprocessing"] == "first-horizon-point"
        )
    else:
        valid = (
            producer["sourceOutputName"] == "quantile_preds"
            and producer["sourceOutputShape"] == [1, 9, 64]
            and producer["postprocessing"] == "first-horizon-median-quantile"
        )
    if not valid:
        raise ModelRecipeError(
            "recipe-invalid", "time-series source output or scalar selection disagrees"
        )


PRODUCER_VALIDATORS = MappingProxyType(
    {
        "identity": ProducerValidator(_accept_any_contract, _identity_producer),
        "sentence-embedding": ProducerValidator(_embedding_contract, _sentence_embedding_producer),
        "clip-image-embedding": ProducerValidator(_embedding_contract, _clip_image_producer),
        "retinexformer-image-enhancement": ProducerValidator(
            _retinexformer_contract, _retinexformer_producer
        ),
        "timeseries-point-forecast": ProducerValidator(_timeseries_contract, _timeseries_producer),
        "timeseries-quantile-forecast": ProducerValidator(
            _timeseries_contract, _timeseries_producer
        ),
    }
)


def finite_json(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(finite_json(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and finite_json(item) for key, item in value.items())
    return False


__all__ = ["PRODUCER_VALIDATORS", "ProducerValidator", "validate_recipe_semantics"]
