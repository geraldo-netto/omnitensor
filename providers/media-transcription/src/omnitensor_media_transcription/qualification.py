"""Frozen provider qualification receipt validation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.resources
import json
from collections.abc import Mapping

from .errors import QualifiedMediaError

QUALIFICATION_EVIDENCE = "media-transcription-rx6600xt-report.json"
MAX_QUALIFICATION_EVIDENCE_BYTES = 256 * 1024

def load_qualification() -> dict:
    resource = importlib.resources.files("omnitensor_media_transcription").joinpath(
        "qualification.json"
    )
    try:
        document = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QualifiedMediaError("media qualification receipt is unreadable") from error
    required = {
        "version",
        "recordedAt",
        "qualified",
        "devices",
        "providerVersion",
        "runtimes",
        "artifacts",
        "evidenceSha256",
    }
    if not isinstance(document, dict) or set(document) != required or document.get("version") != 1:
        raise QualifiedMediaError("media qualification receipt is invalid")
    if document.get("recordedAt") != "2026-08-13":
        raise QualifiedMediaError("media qualification receipt date is invalid")
    if document.get("qualified") is not True:
        raise QualifiedMediaError("media provider has no accepted qualification")
    devices = document.get("devices")
    if (
        not isinstance(devices, dict)
        or set(devices) != {"speech", "vision"}
        or not all(isinstance(value, str) and value for value in devices.values())
    ):
        raise QualifiedMediaError("media qualification devices are invalid")
    if document.get("providerVersion") != importlib.metadata.version(
        "omnitensor-media-transcription"
    ):
        raise QualifiedMediaError("media provider differs from qualification")
    runtimes = document.get("runtimes")
    expected = {
        "llama-cpp-python": importlib.metadata.version("llama-cpp-python"),
        "pywhispercpp": importlib.metadata.version("pywhispercpp"),
    }
    if runtimes != expected:
        raise QualifiedMediaError("media runtimes differ from qualification")
    _validate_qualification_evidence(document)
    return document


def _validate_qualification_evidence(document: Mapping[str, object]) -> None:
    evidence = importlib.resources.files("omnitensor_media_transcription").joinpath(
        QUALIFICATION_EVIDENCE
    )
    try:
        raw = evidence.read_bytes()
    except OSError as error:
        raise QualifiedMediaError("media qualification evidence is unreadable") from error
    if (
        not 0 < len(raw) <= MAX_QUALIFICATION_EVIDENCE_BYTES
        or hashlib.sha256(raw).hexdigest() != document.get("evidenceSha256")
    ):
        raise QualifiedMediaError("media qualification evidence differs from receipt")



__all__ = ["QUALIFICATION_EVIDENCE", "load_qualification"]
