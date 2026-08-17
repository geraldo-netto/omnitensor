from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.plugins.text_encoding import (
    MAX_DETECTION_BYTES,
    TextEncodingError,
    _detection_sample,
    _encoding_key,
    decode_plain_text,
)


@pytest.mark.parametrize(
    ("text", "codec", "detected", "confidence"),
    [
        (
            "Résumé — München costs €42; François owns the façade.",
            "cp1252",
            "Windows-1252",
            0.73,
        ),
        ("会議は八月十三日午後三時から東京で開催します。", "shift_jis", "SHIFT_JIS", 0.99),
        (
            "Проверка мультимедиа состоится в лаборатории.",
            "koi8-r",
            "KOI8-R",
            0.96,
        ),
    ],
)
def test_decode_plain_text_supports_allowlisted_legacy_encodings(text, codec, detected, confidence):
    raw = text.encode(codec)
    calls = []

    def detector(sample):
        calls.append(sample)
        return {"encoding": detected, "confidence": confidence}

    assert decode_plain_text(raw, detector) == text
    assert calls == [raw]


@pytest.mark.parametrize(
    ("text", "codec"),
    [
        (
            "Résumé — München release review. The cost is €42; François owns the façade.",
            "cp1252",
        ),
        ("会議は八月十三日午後三時から東京で開催します。担当者は山田です。", "shift_jis"),
        (
            "Проверка мультимедиа состоится 13 августа 2026 года в лаборатории.",
            "koi8-r",
        ),
    ],
)
def test_default_detector_decodes_representative_legacy_text(text, codec):
    assert decode_plain_text(text.encode(codec)) == text


def test_decode_plain_text_uses_bounded_head_and_tail_detection_sample():
    raw = b"\x80" + (b"a" * MAX_DETECTION_BYTES) + b"z"
    samples = []

    def detector(sample):
        samples.append(sample)
        return {"encoding": "Windows-1252", "confidence": 0.73}

    assert decode_plain_text(raw, detector).startswith("€a")
    assert len(samples[0]) == MAX_DETECTION_BYTES
    assert samples[0].startswith(b"\x80")
    assert samples[0].endswith(b"z")


@pytest.mark.parametrize(
    "detection",
    [
        None,
        {},
        {"encoding": None, "confidence": 1.0},
        {"encoding": "UTF-16", "confidence": 1.0},
        {"encoding": "Windows-1252", "confidence": 0.69},
        {"encoding": "KOI8-R", "confidence": True},
        {"encoding": "KOI8-R", "confidence": "high"},
        {"encoding": "KOI8-R", "confidence": 1.1},
    ],
)
def test_decode_plain_text_rejects_unknown_or_uncertain_detection(detection):
    with pytest.raises(TextEncodingError, match="unsupported|detected"):
        decode_plain_text(b"\xff", lambda _sample: detection)


def test_decode_plain_text_rejects_invalid_detector_and_input():
    with pytest.raises(TypeError, match="must be bytes"):
        decode_plain_text("text")
    with pytest.raises(TypeError, match="must be callable"):
        decode_plain_text(b"\xff", None)


def test_decode_plain_text_rejects_detected_codec_mismatch():
    class BrokenBytes(bytes):
        def decode(self, encoding="utf-8", errors="strict"):
            if encoding == "cp1252":
                raise UnicodeDecodeError(encoding, self, 0, 1, "broken")
            return super().decode(encoding, errors)

    raw = BrokenBytes(b"\xff")
    with pytest.raises(TextEncodingError, match="do not match"):
        decode_plain_text(raw, lambda _sample: {"encoding": "Windows-1252", "confidence": 1.0})


def test_decode_plain_text_rejects_non_round_trip_decoding():
    class BrokenText(str):
        def encode(self, encoding="utf-8", errors="strict"):
            return b"different"

    class BrokenBytes(bytes):
        def decode(self, encoding="utf-8", errors="strict"):
            if encoding == "cp1252":
                return BrokenText("ÿ")
            return super().decode(encoding, errors)

    with pytest.raises(TextEncodingError, match="not lossless"):
        decode_plain_text(
            BrokenBytes(b"\xff"),
            lambda _sample: {"encoding": "Windows-1252", "confidence": 1.0},
        )


def test_decode_plain_text_rejects_legacy_control_bytes():
    with pytest.raises(TextEncodingError, match="control"):
        decode_plain_text(
            b"\x80\x00text",
            lambda _sample: {"encoding": "Windows-1252", "confidence": 1.0},
        )


def test_encoding_key_and_short_sample_are_stable():
    raw = b"short"
    assert _encoding_key("Shift-JIS") == "shiftjis"
    assert _encoding_key(42) == ""
    assert _detection_sample(raw) is raw


@given(st.text(st.characters(exclude_categories=("Cc", "Cs"))))
def test_utf8_round_trip_never_calls_detector(text):
    def detector(_sample):
        raise AssertionError("UTF-8 must not use legacy detection")

    assert decode_plain_text(text.encode("utf-8"), detector) == text
