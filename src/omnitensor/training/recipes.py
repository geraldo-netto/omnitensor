"""Strict offline intake for portable, accelerator-independent model sources."""

from __future__ import annotations

import hashlib
import json
import math
import os
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
        try:
            response = urllib.request.urlopen(uri, timeout=self._timeout_seconds)  # noqa: S310
        except (OSError, urllib.error.URLError) as error:
            raise ModelRecipeError("source-unavailable", f"cannot fetch source: {error}") from error
        with response:
            _validate_https_uri(response.geturl(), "redirected source URI")
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
        tuple(document["evaluation"]),
        targets,
        document,
        hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    )


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
