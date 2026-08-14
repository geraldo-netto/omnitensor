"""Bounded HTTPS transport for explicit offline model-source intake."""

from __future__ import annotations

import math
import urllib.error
import urllib.request
from collections.abc import Iterable

from .recipe_model import ModelRecipeError
from .recipe_uri_policy import source_download_uri, validate_download_response_uri

DOWNLOAD_CHUNK_BYTES = 1024 * 1024
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 60.0
_LEGACY_MODULE = "omnitensor.training.recipes"


class HttpsSourceTransport:
    """Bounded HTTPS adapter used only by the explicit producer command."""

    def __init__(self, timeout_seconds: float = DEFAULT_DOWNLOAD_TIMEOUT_SECONDS):
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("download timeout must be a positive finite number")
        self._timeout_seconds = timeout_seconds

    def chunks(self, uri: str, maximum_bytes: int) -> Iterable[bytes]:
        download_uri = source_download_uri(uri)
        try:
            response = urllib.request.urlopen(  # noqa: S310
                download_uri, timeout=self._timeout_seconds
            )
        except (OSError, urllib.error.URLError) as error:
            raise ModelRecipeError("source-unavailable", f"cannot fetch source: {error}") from error
        with response:
            validate_download_response_uri(uri, response.geturl())
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


HttpsSourceTransport.__module__ = _LEGACY_MODULE

__all__ = ["HttpsSourceTransport"]
