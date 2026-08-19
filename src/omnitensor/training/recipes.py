"""Compatibility facade for split offline model-recipe intake modules."""

from __future__ import annotations

# Preserve the original import-visible compatibility surface for one release.
import hashlib as _hashlib
import json as _json
import math as _math
import os
import re as _re
import shutil as _shutil
import tempfile as _tempfile
import urllib as _urllib
from collections.abc import Iterable as _Iterable
from dataclasses import dataclass as _dataclass
from pathlib import Path
from typing import Protocol as _Protocol

import jsonschema as _jsonschema

from ..atomicio import (
    JsonTooLargeError as _JsonTooLargeError,
)
from ..atomicio import (
    read_json_bounded as _read_json_bounded,
)
from ..atomicio import (
    write_json_atomic as _write_json_atomic,
)
from ..registry import load_schema as _load_schema
from ..registry import validate_document as _validate_document
from . import recipe_fetch as _fetch
from . import recipe_model as _model
from . import recipe_registry as _registry
from . import recipe_transport as _transport
from . import recipe_uri_policy as _uri
from . import recipe_validation as _validation

hashlib = _hashlib
json = _json
math = _math
re = _re
shutil = _shutil
tempfile = _tempfile
urllib = _urllib
Iterable = _Iterable
dataclass = _dataclass
Protocol = _Protocol
jsonschema = _jsonschema
JsonTooLargeError = _JsonTooLargeError
read_json_bounded = _read_json_bounded
write_json_atomic = _write_json_atomic
load_schema = _load_schema
validate_document = _validate_document

MODEL_RECIPE_VERSION = _model.MODEL_RECIPE_VERSION
MAX_RECIPE_BYTES = _registry.MAX_RECIPE_BYTES
MAX_TOTAL_SOURCE_BYTES = _validation.MAX_TOTAL_SOURCE_BYTES
DOWNLOAD_CHUNK_BYTES = _transport.DOWNLOAD_CHUNK_BYTES
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = _transport.DEFAULT_DOWNLOAD_TIMEOUT_SECONDS
RECEIPT_FILENAME = _model.RECEIPT_FILENAME
MAX_BUNDLED_RECIPES = _registry.MAX_BUNDLED_RECIPES
REFUSED_BUNDLED_RECIPES = _registry.REFUSED_BUNDLED_RECIPES

_RECIPE_ID = _registry._RECIPE_ID
_GOOGLE_DRIVE_FILE_ID = _uri._GOOGLE_DRIVE_FILE_ID
_PINNED_REVISION = _uri._PINNED_REVISION
_HUGGING_FACE_CDN = _uri._HUGGING_FACE_CDN

ModelRecipeError = _model.ModelRecipeError
ModelSource = _model.ModelSource
ModelLicense = _model.ModelLicense
TargetClaim = _model.TargetClaim
ModelRecipe = _model.ModelRecipe
FetchedModelSource = _model.FetchedModelSource
SourceTransport = _model.SourceTransport
HttpsSourceTransport = _transport.HttpsSourceTransport

load_model_recipe = _registry.load_model_recipe

_validate_recipe_semantics = _validation.validate_recipe_semantics
_validate_source_inventory = _validation.validate_source_inventory
_validate_preprocessing_sources = _validation.validate_preprocessing_sources
_validate_contracts = _validation.validate_contracts
_validate_producer = _validation.validate_producer
_validate_sentence_embedding_producer = _validation._sentence_embedding_producer
_validate_clip_image_producer = _validation._clip_image_producer
_validate_retinexformer_producer = _validation._retinexformer_producer
_validate_timeseries_producer = _validation._timeseries_producer
_finite_json = _validation.finite_json

_validate_https_uri = _uri.validate_https_uri
_source_download_uri = _uri.source_download_uri
_validate_download_response_uri = _uri.validate_download_response_uri
_is_pinned_hugging_face_redirect = _uri.is_pinned_hugging_face_redirect

_fetch_one = _fetch.fetch_one
_receipt_document = _fetch.receipt_document
_installed_source_matches = _fetch.installed_source_matches
_source_file_matches = _fetch.source_file_matches


def _validate_specialized_producer(document: dict, producer: dict, inputs: list, kind: str) -> None:
    _validation.PRODUCER_VALIDATORS[kind].semantics(document, producer, inputs)


def _validate_producer_contract(document: dict, kind: str) -> None:
    _validation.PRODUCER_VALIDATORS[kind].contract(document)


def bundled_model_recipe_root() -> Path:
    """Resolve the catalog through the facade's patchable module location."""
    return _registry.bundled_model_recipe_root(__file__)


def load_bundled_model_recipes() -> tuple[ModelRecipe, ...]:
    """Load through the facade's patchable root, bound, and recipe loader."""
    return _registry.load_bundled_model_recipes(
        root=bundled_model_recipe_root(),
        maximum_recipes=MAX_BUNDLED_RECIPES,
        loader=load_model_recipe,
    )


def resolve_model_recipe_path(reference: str | Path) -> Path:
    """Resolve through the facade's patchable root and refusal registry."""
    return _registry.resolve_model_recipe_path(
        reference,
        root_resolver=bundled_model_recipe_root,
        refused=REFUSED_BUNDLED_RECIPES,
    )


def fetch_model_sources(
    recipe_path: Path | str,
    destination_root: Path | str,
    *,
    accepted_license: str,
    transport: SourceTransport | None = None,
) -> FetchedModelSource:
    """Fetch through the facade's patchable loader, matcher, and rename seam."""
    return _fetch._fetch_model_sources(
        recipe_path,
        destination_root,
        accepted_license=accepted_license,
        transport=transport,
        recipe_loader=load_model_recipe,
        installed_matcher=_installed_source_matches,
        rename=os.rename,
    )


def open_fetched_model_source(
    recipe_path: Path | str,
    source_root: Path | str,
) -> FetchedModelSource:
    """Open through the facade's patchable loader and matcher seam."""
    return _fetch._open_fetched_model_source(
        recipe_path,
        source_root,
        recipe_loader=load_model_recipe,
        installed_matcher=_installed_source_matches,
    )
