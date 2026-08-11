"""Strict offline intake for portable, accelerator-independent model sources."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import jsonschema

from ..atomicio import JsonTooLargeError, read_json_bounded, write_json_atomic
from ..registry import load_schema, validate_document

MODEL_RECIPE_VERSION = 1
MAX_RECIPE_BYTES = 256 * 1024
MAX_SOURCE_BYTES = 2 * 1024 * 1024 * 1024
MAX_TOTAL_SOURCE_BYTES = 4 * 1024 * 1024 * 1024
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 60.0
RECEIPT_FILENAME = "source-receipt.json"
MAX_BUNDLED_RECIPES = 128
_RECIPE_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_GOOGLE_DRIVE_FILE_ID = re.compile(r"^[A-Za-z0-9_-]{20,100}$")
REFUSED_BUNDLED_RECIPES = {
    "zero-dce": (
        "the official Zero-DCE code and weights are CC-BY-NC-4.0 for academic "
        "research only; OmniTensor accepts only redistributable commercial-use sources"
    )
}


class ModelRecipeError(ValueError):
    """Stable producer failure safe to show to a local operator."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class ModelSource:
    """One immutable file belonging to a portable model release."""

    role: str
    uri: str
    revision: str
    filename: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ModelLicense:
    """Weight-specific terms, deliberately separate from repository code."""

    spdx: str
    terms_uri: str
    attribution: str


@dataclass(frozen=True, slots=True)
class TargetClaim:
    """What is actually known about one native compilation lane."""

    status: str
    compiler: str | None
    fully_quantized: bool
    reason: str
    evidence_sha256: str | None


@dataclass(frozen=True, slots=True)
class ModelRecipe:
    """Validated portable source identity and its target-independent semantics."""

    id: str
    version: str
    family: str
    profile_ids: tuple[str, ...]
    source_format: str
    sources: tuple[ModelSource, ...]
    license: ModelLicense
    tensor_contract: dict
    output_contract: dict
    preprocessing: dict | None
    producer: dict | None
    evaluation: tuple[dict, ...]
    targets: dict[str, TargetClaim]
    document: dict
    document_sha256: str

    @property
    def model_source(self) -> ModelSource:
        return next(source for source in self.sources if source.role == "model")


@dataclass(frozen=True, slots=True)
class FetchedModelSource:
    """Atomic local source installation and its persisted receipt."""

    recipe: ModelRecipe
    root: Path
    receipt_path: Path

    def document(self) -> dict:
        return {
            "recipeId": self.recipe.id,
            "recipeVersion": self.recipe.version,
            "recipeSha256": self.recipe.document_sha256,
            "root": str(self.root),
            "receipt": str(self.receipt_path),
            "files": [source.filename for source in self.recipe.sources],
        }


class SourceTransport(Protocol):
    """Yield bytes from one source URI without owning recipe policy."""

    def chunks(self, uri: str, maximum_bytes: int) -> Iterable[bytes]: ...


class HttpsSourceTransport:
    """Bounded HTTPS adapter used only by the explicit producer command."""

    def __init__(self, timeout_seconds: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS):
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("download timeout must be a positive finite number")
        self._timeout_seconds = timeout_seconds

    def chunks(self, uri: str, maximum_bytes: int) -> Iterable[bytes]:
        download_uri = _source_download_uri(uri)
        try:
            response = urllib.request.urlopen(  # noqa: S310
                download_uri, timeout=self._timeout_seconds
            )
        except (OSError, urllib.error.URLError) as error:
            raise ModelRecipeError("source-unavailable", f"cannot fetch source: {error}") from error
        with response:
            _validate_download_response_uri(uri, response.geturl())
            declared = response.headers.get("Content-Length")
            if declared is not None:
                try:
                    declared_size = int(declared)
                except ValueError as error:
                    raise ModelRecipeError(
                        "source-invalid", "source Content-Length is invalid"
                    ) from error
                if declared_size < 0 or declared_size > maximum_bytes:
                    raise ModelRecipeError("source-too-large", "source exceeds its declared bound")
            read = 0
            while True:
                chunk = response.read(min(DOWNLOAD_CHUNK_BYTES, maximum_bytes - read + 1))
                if not chunk:
                    return
                read += len(chunk)
                if read > maximum_bytes:
                    raise ModelRecipeError("source-too-large", "source exceeds its declared bound")
                yield chunk


def load_model_recipe(path: Path | str) -> ModelRecipe:
    """Load one bounded recipe and enforce cross-field source/model semantics."""
    try:
        document = read_json_bounded(Path(path), MAX_RECIPE_BYTES)
    except (OSError, ValueError, JsonTooLargeError) as error:
        raise ModelRecipeError("recipe-invalid", f"cannot read model recipe: {error}") from error
    violations = validate_document("model-recipe.schema.json", document)
    if violations:
        raise ModelRecipeError("recipe-invalid", "; ".join(violations))
    assert isinstance(document, dict)  # schema proves this
    _validate_recipe_semantics(document)
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"))
    sources = tuple(
        ModelSource(
            item["role"],
            item["uri"],
            item["revision"],
            item["filename"],
            item["sha256"],
            item["sizeBytes"],
        )
        for item in document["sources"]
    )
    license_document = document["license"]
    targets = {
        accelerator: TargetClaim(
            claim["status"],
            claim["compiler"],
            claim["fullyQuantized"],
            claim["reason"],
            claim.get("evidenceSha256"),
        )
        for accelerator, claim in document["targets"].items()
    }
    return ModelRecipe(
        document["id"],
        document["version"],
        document["family"],
        tuple(document["profileIds"]),
        document["sourceFormat"],
        sources,
        ModelLicense(
            license_document["spdx"],
            license_document["termsUri"],
            license_document["attribution"],
        ),
        document["tensorContract"],
        document["outputContract"],
        document.get("preprocessing"),
        document.get("producer"),
        tuple(document["evaluation"]),
        targets,
        document,
        hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


def bundled_model_recipe_root() -> Path:
    """Return the source-checkout or installed-wheel recipe catalog."""
    package_root = Path(__file__).resolve().parents[1]
    for candidate in (package_root / "model-recipes", package_root.parents[1] / "model-recipes"):
        if candidate.is_dir():
            return candidate
    raise ModelRecipeError("recipe-catalog-missing", "bundled model recipe catalog is absent")


def load_bundled_model_recipes() -> tuple[ModelRecipe, ...]:
    """Load the complete bounded catalog and reject ambiguous package contents."""
    root = bundled_model_recipe_root()
    paths = sorted(root.glob("*.json"))
    if not paths:
        raise ModelRecipeError("recipe-catalog-invalid", "bundled model recipe catalog is empty")
    if len(paths) > MAX_BUNDLED_RECIPES:
        raise ModelRecipeError(
            "recipe-catalog-invalid", "bundled model recipe catalog is too large"
        )
    recipes: list[ModelRecipe] = []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ModelRecipeError("recipe-catalog-invalid", f"unsafe recipe entry: {path.name}")
        recipe = load_model_recipe(path)
        if path.name != f"{recipe.id}.json":
            raise ModelRecipeError(
                "recipe-catalog-invalid", f"recipe filename does not match id: {path.name}"
            )
        recipes.append(recipe)
    return tuple(recipes)


def resolve_model_recipe_path(reference: str | Path) -> Path:
    """Resolve an explicit path or one traversal-safe bundled recipe id."""
    path = Path(reference).expanduser()
    if path.is_file():
        return path
    text = str(reference)
    if not _RECIPE_ID.fullmatch(text):
        return path
    if text in REFUSED_BUNDLED_RECIPES:
        raise ModelRecipeError("recipe-refused", REFUSED_BUNDLED_RECIPES[text])
    bundled = bundled_model_recipe_root() / f"{text}.json"
    if not bundled.is_file() or bundled.is_symlink():
        raise ModelRecipeError("recipe-not-found", f"no bundled model recipe is named {text}")
    return bundled


def fetch_model_sources(
    recipe_path: Path | str,
    destination_root: Path | str,
    *,
    accepted_license: str,
    transport: SourceTransport | None = None,
) -> FetchedModelSource:
    """Fetch every pinned source, verify bytes, then publish one atomic version."""
    recipe = load_model_recipe(recipe_path)
    if accepted_license != recipe.license.spdx:
        raise ModelRecipeError(
            "license-not-accepted",
            f"pass --accept-license {recipe.license.spdx} after reviewing "
            f"{recipe.license.terms_uri}",
        )
    destination = Path(destination_root) / recipe.id / recipe.version
    if destination.exists():
        if _installed_source_matches(destination, recipe):
            return FetchedModelSource(recipe, destination, destination / RECEIPT_FILENAME)
        raise ModelRecipeError(
            "source-conflict", f"existing source version does not match recipe: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".model-source-", dir=destination.parent))
    adapter = transport or HttpsSourceTransport()
    try:
        for source in recipe.sources:
            _fetch_one(source, stage / source.filename, adapter)
        write_json_atomic(
            stage / RECEIPT_FILENAME,
            _receipt_document(recipe),
            prefix=".source-receipt-",
        )
        try:
            os.rename(stage, destination)
        except FileExistsError:
            if _installed_source_matches(destination, recipe):
                return FetchedModelSource(recipe, destination, destination / RECEIPT_FILENAME)
            raise ModelRecipeError(
                "source-conflict", f"source version appeared concurrently: {destination}"
            ) from None
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return FetchedModelSource(recipe, destination, destination / RECEIPT_FILENAME)


def _validate_recipe_semantics(document: dict) -> None:
    sources = document["sources"]
    roles = _validate_source_inventory(sources)
    for source in sources:
        _validate_https_uri(source["uri"], "source URI")
        path_segments = urllib.parse.unquote(urllib.parse.urlsplit(source["uri"]).path).split("/")
        if source["revision"] not in path_segments:
            raise ModelRecipeError(
                "recipe-invalid", f"source URI does not pin revision {source['revision']}"
            )
    _validate_https_uri(document["license"]["termsUri"], "license terms URI")
    _validate_preprocessing_sources(document, roles)
    _validate_contracts(document)
    _validate_producer(document)
    if not _finite_json(document["evaluation"]):
        raise ModelRecipeError("recipe-invalid", "evaluation metrics must be finite")
    tpu = document["targets"]["tpu"]
    if tpu["status"] == "validated" and not tpu["fullyQuantized"]:
        raise ModelRecipeError(
            "recipe-invalid", "validated TPU compatibility requires full quantization"
        )


def _validate_source_inventory(sources: list[dict]) -> tuple[str, ...]:
    roles = [source["role"] for source in sources]
    filenames = [source["filename"] for source in sources]
    if roles.count("model") != 1:
        raise ModelRecipeError("recipe-invalid", "exactly one model source is required")
    if len(set(filenames)) != len(filenames):
        raise ModelRecipeError("recipe-invalid", "source filenames must be unique")
    if len(set(roles)) != len(roles):
        raise ModelRecipeError("recipe-invalid", "source roles must be unique")
    total = sum(source["sizeBytes"] for source in sources)
    if total > MAX_TOTAL_SOURCE_BYTES:
        raise ModelRecipeError("recipe-invalid", "combined source size exceeds 4 GiB")
    return tuple(roles)


def _validate_preprocessing_sources(document: dict, roles: tuple[str, ...]) -> None:
    required_sources = set((document.get("preprocessing") or {}).get("artifacts", ()))
    missing_sources = sorted(required_sources - set(roles))
    if missing_sources:
        raise ModelRecipeError(
            "recipe-invalid",
            f"preprocessing artifact has no pinned source: {missing_sources[0]}",
        )


def _validate_contracts(document: dict) -> None:
    model_schema = load_schema("workload-manifest.schema.json")["$defs"]["model"]
    for field in ("tensorContract", "outputContract"):
        schema = model_schema["properties"][field]
        violations = sorted(
            jsonschema.Draft202012Validator(schema).iter_errors(document[field]), key=str
        )
        if violations:
            detail = "; ".join(
                f"{'/'.join(str(part) for part in error.absolute_path) or '/'}: {error.message}"
                for error in violations
            )
            raise ModelRecipeError(
                "recipe-invalid", f"model contract is invalid at {field}: {detail}"
            )


def _validate_producer(document: dict) -> None:
    producer = document.get("producer")
    if producer is None:
        return
    inputs = document["tensorContract"]["inputs"]
    if len(producer["inputNames"]) != len(inputs):
        raise ModelRecipeError(
            "recipe-invalid", "producer inputNames must match tensorContract input order"
        )
    kind = producer["kind"]
    _validate_producer_contract(document, kind)
    if producer["outputShape"][0] != 1:
        raise ModelRecipeError("recipe-invalid", "producer outputShape must have batch size 1")
    _validate_specialized_producer(document, producer, inputs, kind)


def _validate_specialized_producer(
    document: dict, producer: dict, inputs: list, kind: str
) -> None:
    if kind == "sentence-embedding":
        _validate_sentence_embedding_producer(document, producer, inputs)
    elif kind == "clip-image-embedding":
        _validate_clip_image_producer(document, producer, inputs)
    elif kind == "retinexformer-image-enhancement":
        _validate_retinexformer_producer(document, producer, inputs)
    elif kind in {"timeseries-point-forecast", "timeseries-quantile-forecast"}:
        _validate_timeseries_producer(document, producer, inputs)
    elif producer["sourceOutputShape"] != producer["outputShape"]:
        raise ModelRecipeError(
            "recipe-invalid", "identity producer cannot change the source output shape"
        )


def _validate_producer_contract(document: dict, kind: str) -> None:
    if kind in {"sentence-embedding", "clip-image-embedding"}:
        if document["outputContract"]["kind"] != "embedding":
            raise ModelRecipeError(
                "recipe-invalid", "embedding producers require an embedding output contract"
            )
    elif kind == "retinexformer-image-enhancement" and (
        document["family"] != "low-light" or document["outputContract"]["kind"] != "raw"
    ):
        raise ModelRecipeError(
            "recipe-invalid", "image enhancement producers require a raw low-light contract"
        )
    elif kind in {"timeseries-point-forecast", "timeseries-quantile-forecast"} and (
        document["family"] != "forecast" or document["outputContract"]["kind"] != "raw"
    ):
        raise ModelRecipeError(
            "recipe-invalid", "time-series producers require a raw forecast output contract"
        )


def _validate_sentence_embedding_producer(document: dict, producer: dict, inputs: list) -> None:
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
        producer["postprocessing"]
        not in {"attention-mask-mean-pool-l2", "cls-token-l2"}
        or source_shape[:2] != [batch, sequence]
        or len(source_shape) != 3
        or output_shape != [batch, source_shape[2]]
    ):
        raise ModelRecipeError(
            "recipe-invalid", "sentence embedding pooling shapes or postprocessing disagree"
        )


def _validate_clip_image_producer(document: dict, producer: dict, inputs: list) -> None:
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


def _validate_retinexformer_producer(document: dict, producer: dict, inputs: list) -> None:
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


def _validate_timeseries_producer(document: dict, producer: dict, inputs: list) -> None:
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


def _validate_https_uri(uri: str, label: str) -> None:
    parsed = urllib.parse.urlsplit(uri)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ModelRecipeError(
            "recipe-invalid",
            f"{label} must be an HTTPS URL without credentials, query, or fragment",
        )


def _source_download_uri(uri: str) -> str:
    """Map one immutable official Drive share path to its byte-download endpoint."""
    parsed = urllib.parse.urlsplit(uri)
    parts = parsed.path.strip("/").split("/")
    if parsed.hostname != "drive.google.com" or len(parts) != 4 or parts[:2] != ["file", "d"]:
        return uri
    file_id = parts[2]
    if parts[3] != "view" or not _GOOGLE_DRIVE_FILE_ID.fullmatch(file_id):
        raise ModelRecipeError("source-invalid", "Google Drive source path is invalid")
    query = urllib.parse.urlencode({"id": file_id, "export": "download", "confirm": "t"})
    return urllib.parse.urlunsplit(
        ("https", "drive.usercontent.google.com", "/download", query, "")
    )


def _validate_download_response_uri(declared_uri: str, response_uri: str) -> None:
    mapped = _source_download_uri(declared_uri)
    if response_uri == mapped:
        return
    _validate_https_uri(response_uri, "redirected source URI")


def _fetch_one(source: ModelSource, destination: Path, transport: SourceTransport) -> None:
    digest = hashlib.sha256()
    size = 0
    try:
        with destination.open("xb") as handle:
            for chunk in transport.chunks(source.uri, min(source.size_bytes, MAX_SOURCE_BYTES)):
                if not isinstance(chunk, bytes) or not chunk:
                    raise ModelRecipeError("source-invalid", "transport returned an invalid chunk")
                size += len(chunk)
                if size > source.size_bytes:
                    raise ModelRecipeError(
                        "source-too-large", f"{source.filename} exceeds sizeBytes"
                    )
                digest.update(chunk)
                handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    if size != source.size_bytes:
        destination.unlink(missing_ok=True)
        raise ModelRecipeError(
            "source-size-mismatch",
            f"{source.filename} has {size} bytes; expected {source.size_bytes}",
        )
    if digest.hexdigest() != source.sha256:
        destination.unlink(missing_ok=True)
        raise ModelRecipeError("source-digest-mismatch", f"{source.filename} digest does not match")


def _receipt_document(recipe: ModelRecipe) -> dict:
    return {
        "version": MODEL_RECIPE_VERSION,
        "recipe": {
            "id": recipe.id,
            "version": recipe.version,
            "sha256": recipe.document_sha256,
        },
        "license": {
            "spdx": recipe.license.spdx,
            "termsUri": recipe.license.terms_uri,
            "attribution": recipe.license.attribution,
        },
        "sources": [
            {
                "role": source.role,
                "revision": source.revision,
                "filename": source.filename,
                "sha256": source.sha256,
                "sizeBytes": source.size_bytes,
            }
            for source in recipe.sources
        ],
    }


def _installed_source_matches(destination: Path, recipe: ModelRecipe) -> bool:
    receipt_path = destination / RECEIPT_FILENAME
    if receipt_path.is_symlink():
        return False
    try:
        receipt = read_json_bounded(receipt_path, MAX_RECIPE_BYTES)
    except (OSError, ValueError, JsonTooLargeError):
        return False
    if receipt != _receipt_document(recipe):
        return False
    for source in recipe.sources:
        if not _source_file_matches(destination / source.filename, source):
            return False
    return True


def _source_file_matches(path: Path, source: ModelSource) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        if path.stat().st_size != source.size_bytes:
            return False
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(DOWNLOAD_CHUNK_BYTES), b""):
                digest.update(chunk)
    except OSError:
        return False
    return digest.hexdigest() == source.sha256


def _finite_json(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite_json(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _finite_json(item) for key, item in value.items())
    return False
