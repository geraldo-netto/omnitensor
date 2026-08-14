"""Immutable source-URI and redirect policy for reviewed model recipes."""

from __future__ import annotations

import re
import urllib.parse

from .recipe_model import ModelRecipeError

_GOOGLE_DRIVE_FILE_ID = re.compile(r"^[A-Za-z0-9_-]{20,100}$")
_PINNED_REVISION = re.compile(r"^[0-9a-f]{40}$")
_HUGGING_FACE_CDN = re.compile(
    r"^(?:(?:[a-z0-9-]+\.)*cdn\.hf\.co|cas-bridge\.xethub\.hf\.co)$"
)


def validate_https_uri(uri: str, label: str) -> None:
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


def revision_is_path_segment(uri: str, revision: str) -> bool:
    path_segments = urllib.parse.unquote(urllib.parse.urlsplit(uri).path).split("/")
    return revision in path_segments


def source_download_uri(uri: str) -> str:
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


def validate_download_response_uri(declared_uri: str, response_uri: str) -> None:
    mapped = source_download_uri(declared_uri)
    if response_uri == mapped:
        return
    if is_pinned_hugging_face_redirect(declared_uri, response_uri):
        return
    validate_https_uri(response_uri, "redirected source URI")


def is_pinned_hugging_face_redirect(declared_uri: str, response_uri: str) -> bool:
    """Accept only signed CDN redirects from an immutable public Hub revision."""
    declared = urllib.parse.urlsplit(declared_uri)
    response = urllib.parse.urlsplit(response_uri)
    source_parts = declared.path.strip("/").split("/")
    response_parts = response.path.strip("/").split("/")
    pinned_source = (
        declared.hostname == "huggingface.co"
        and len(source_parts) >= 5
        and source_parts[2] == "resolve"
        and _PINNED_REVISION.fullmatch(source_parts[3]) is not None
    )
    safe_common = (
        response.scheme == "https"
        and response.hostname is not None
        and response.username is None
        and response.password is None
        and bool(response.path.strip("/"))
        and bool(response.query)
        and not response.fragment
    )
    signed_cdn = (
        response.hostname is not None
        and _HUGGING_FACE_CDN.fullmatch(response.hostname) is not None
    )
    exact_cache = (
        response.hostname == "huggingface.co"
        and response_parts[:3] == ["api", "resolve-cache", "models"]
        and response_parts[3:5] == source_parts[:2]
        and response_parts[5:] == [source_parts[3], *source_parts[4:]]
    )
    return pinned_source and safe_common and (signed_cdn or exact_cache)


__all__ = [
    "is_pinned_hugging_face_redirect",
    "revision_is_path_segment",
    "source_download_uri",
    "validate_download_response_uri",
    "validate_https_uri",
]
