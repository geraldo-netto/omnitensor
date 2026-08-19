"""Verified atomic installation and reopening of pinned model sources."""

from __future__ import annotations

import errno
import hashlib
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

from ..atomicio import JsonTooLargeError, read_json_bounded, write_json_atomic
from .recipe_model import (
    MODEL_RECIPE_VERSION,
    RECEIPT_FILENAME,
    FetchedModelSource,
    ModelRecipe,
    ModelRecipeError,
    ModelSource,
    SourceTransport,
)
from .recipe_registry import MAX_RECIPE_BYTES, load_model_recipe
from .recipe_transport import DOWNLOAD_CHUNK_BYTES, HttpsSourceTransport

# A directory rename onto an existing directory reports EEXIST when the
# destination is empty and ENOTEMPTY when it is not; both mean another
# fetch published this version first.
_DESTINATION_EXISTS = frozenset({errno.EEXIST, errno.ENOTEMPTY})


def fetch_model_sources(
    recipe_path: Path | str,
    destination_root: Path | str,
    *,
    accepted_license: str,
    transport: SourceTransport | None = None,
) -> FetchedModelSource:
    """Fetch every pinned source, verify bytes, then publish one atomic version."""
    return _fetch_model_sources(
        recipe_path,
        destination_root,
        accepted_license=accepted_license,
        transport=transport,
        recipe_loader=load_model_recipe,
        installed_matcher=installed_source_matches,
        rename=os.rename,
    )


def _fetch_model_sources(
    recipe_path: Path | str,
    destination_root: Path | str,
    *,
    accepted_license: str,
    transport: SourceTransport | None,
    recipe_loader: Callable[[Path | str], ModelRecipe],
    installed_matcher: Callable[[Path, ModelRecipe], bool],
    rename: Callable[[Path, Path], None],
) -> FetchedModelSource:
    recipe = recipe_loader(recipe_path)
    if accepted_license != recipe.license.spdx:
        raise ModelRecipeError(
            "license-not-accepted",
            f"pass --accept-license {recipe.license.spdx} after reviewing "
            f"{recipe.license.terms_uri}",
        )
    destination = Path(destination_root) / recipe.id / recipe.version
    if destination.exists():
        if installed_matcher(destination, recipe):
            return FetchedModelSource(recipe, destination, destination / RECEIPT_FILENAME)
        raise ModelRecipeError(
            "source-conflict", f"existing source version does not match recipe: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".model-source-", dir=destination.parent))
    adapter = transport or HttpsSourceTransport()
    try:
        for source in recipe.sources:
            fetch_one(source, stage / source.filename, adapter)
        write_json_atomic(
            stage / RECEIPT_FILENAME,
            receipt_document(recipe),
            prefix=".source-receipt-",
        )
        _publish_stage(stage, destination, recipe, rename=rename, matcher=installed_matcher)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return FetchedModelSource(recipe, destination, destination / RECEIPT_FILENAME)


def _publish_stage(
    stage: Path,
    destination: Path,
    recipe: ModelRecipe,
    *,
    rename: Callable[[Path, Path], None],
    matcher: Callable[[Path, ModelRecipe], bool],
) -> None:
    """Publish the staged version, tolerating a concurrent fetch that matches."""
    try:
        rename(stage, destination)
    except OSError as error:
        if not _destination_already_exists(error):
            raise
        if matcher(destination, recipe):
            return
        raise ModelRecipeError(
            "source-conflict", f"source version appeared concurrently: {destination}"
        ) from None


def _destination_already_exists(error: OSError) -> bool:
    """True when a rename failed because the version directory is already there."""
    return isinstance(error, FileExistsError) or error.errno in _DESTINATION_EXISTS


def open_fetched_model_source(
    recipe_path: Path | str,
    source_root: Path | str,
) -> FetchedModelSource:
    """Open an already fetched source only after rechecking every pinned byte."""
    return _open_fetched_model_source(
        recipe_path,
        source_root,
        recipe_loader=load_model_recipe,
        installed_matcher=installed_source_matches,
    )


def _open_fetched_model_source(
    recipe_path: Path | str,
    source_root: Path | str,
    *,
    recipe_loader: Callable[[Path | str], ModelRecipe],
    installed_matcher: Callable[[Path, ModelRecipe], bool],
) -> FetchedModelSource:
    recipe = recipe_loader(recipe_path)
    destination = Path(source_root) / recipe.id / recipe.version
    if not installed_matcher(destination, recipe):
        raise ModelRecipeError(
            "source-invalid",
            f"fetched source does not match its reviewed recipe: {destination}",
        )
    return FetchedModelSource(recipe, destination, destination / RECEIPT_FILENAME)


def fetch_one(source: ModelSource, destination: Path, transport: SourceTransport) -> None:
    digest = hashlib.sha256()
    size = 0
    try:
        with destination.open("xb") as handle:
            for chunk in transport.chunks(source.uri, source.size_bytes):
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


def receipt_document(recipe: ModelRecipe) -> dict:
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


def installed_source_matches(destination: Path, recipe: ModelRecipe) -> bool:
    receipt_path = destination / RECEIPT_FILENAME
    if receipt_path.is_symlink():
        return False
    try:
        receipt = read_json_bounded(receipt_path, MAX_RECIPE_BYTES)
    except (OSError, ValueError, JsonTooLargeError):
        return False
    if receipt != receipt_document(recipe):
        return False
    for source in recipe.sources:
        if not source_file_matches(destination / source.filename, source):
            return False
    return True


def source_file_matches(path: Path, source: ModelSource) -> bool:
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


__all__ = ["fetch_model_sources", "open_fetched_model_source"]
