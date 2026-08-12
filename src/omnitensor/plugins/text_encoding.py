"""Bounded decoding for explicitly selected plain-text documents."""

from __future__ import annotations

import unicodedata
from collections.abc import Callable, Mapping

from chardet import detect

MAX_DETECTION_BYTES = 512 * 1024
_ALLOWED_CONTROLS = frozenset("\n\r\t")
_LEGACY_ENCODINGS = {
    "cp932": ("cp932", 0.9),
    "koi8r": ("koi8-r", 0.9),
    "shiftjis": ("shift_jis", 0.9),
    "windows1252": ("cp1252", 0.7),
}

EncodingDetector = Callable[[bytes], Mapping[str, object]]


class TextEncodingError(UnicodeError):
    """Selected text bytes have no supported, trustworthy decoding."""


def _encoding_key(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return "".join(character for character in value.lower() if character.isalnum())


def _detection_sample(raw: bytes) -> bytes:
    if len(raw) <= MAX_DETECTION_BYTES:
        return raw
    half = MAX_DETECTION_BYTES // 2
    return raw[:half] + raw[-half:]


def _validate_decoded_text(text: str) -> str:
    for character in text:
        if character not in _ALLOWED_CONTROLS and unicodedata.category(character) == "Cc":
            raise TextEncodingError("plain text contains unsupported control bytes")
    return text


def decode_plain_text(raw: bytes, detector: EncodingDetector = detect) -> str:
    """Decode UTF-8 or a high-confidence allowlisted legacy encoding.

    Detection sees at most 512 KiB even when the already size-bounded source is
    larger. Exact round-trip and control-byte checks prevent lossy or binary
    content from entering document extraction.
    """
    if not isinstance(raw, bytes):
        raise TypeError("plain text input must be bytes")
    if not callable(detector):
        raise TypeError("encoding detector must be callable")
    try:
        return _validate_decoded_text(raw.decode("utf-8-sig"))
    except UnicodeDecodeError:
        pass

    detection = detector(_detection_sample(raw))
    if not isinstance(detection, Mapping):
        raise TextEncodingError("plain text encoding could not be detected")
    candidate = _LEGACY_ENCODINGS.get(_encoding_key(detection.get("encoding")))
    confidence = detection.get("confidence")
    if (
        candidate is None
        or isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not candidate[1] <= float(confidence) <= 1.0
    ):
        raise TextEncodingError("plain text encoding is unsupported or uncertain")
    codec = candidate[0]
    try:
        text = raw.decode(codec, errors="strict")
    except UnicodeDecodeError as error:
        raise TextEncodingError("plain text bytes do not match the detected encoding") from error
    if text.encode(codec, errors="strict") != raw:
        raise TextEncodingError("plain text decoding is not lossless")
    return _validate_decoded_text(text)
