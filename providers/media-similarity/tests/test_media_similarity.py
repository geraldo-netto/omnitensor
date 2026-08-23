from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from omnitensor_media_similarity import fingerprints as fingerprint_module
from omnitensor_media_similarity import similarity as similarity_module
from omnitensor_media_similarity.fingerprints import (
    MediaFingerprint,
    MediaFingerprintError,
    fingerprint_file,
    perceptual_hash,
    spectral_hash,
)
from omnitensor_media_similarity.provider import (
    MediaSimilarityError,
    MediaSimilarityPlugin,
    _safe_relative_path,
    _validated_request,
)
from omnitensor_media_similarity.similarity import compare_media, similarity_groups

from omnitensor.plugins.protocol import PluginContext, PluginRequest
from omnitensor.registry import validate_document
from omnitensor.sdk import CancellationController, PluginResultStatus


class Progress:
    def __init__(self) -> None:
        self.items = []

    async def report(self, item) -> None:
        self.items.append(item)


def cancellation():
    return CancellationController()


def media(
    name: str,
    digest: str,
    *,
    visual: tuple[int, ...] = (),
    audio: tuple[int, ...] = (),
) -> MediaFingerprint:
    modality = "audio-video" if visual and audio else "video" if visual else "audio"
    return MediaFingerprint(name, Path(name).name, digest, modality, 2_000, visual, audio)


def test_perceptual_hash_ignores_source_resolution():
    small = np.zeros((48, 64), dtype=np.uint8)
    small[8:32, 10:44] = 220
    large = np.repeat(np.repeat(small, 3, axis=0), 3, axis=1)

    assert perceptual_hash(small) == perceptual_hash(large) == 0x3009B9B9B9B00


def test_spectral_hash_ignores_gain():
    time = np.arange(16_000) / 16_000
    wave = np.sin(2 * np.pi * 440 * time).astype(np.float32)

    assert spectral_hash(wave) == spectral_hash(wave * 0.2) == 0x2000
    assert spectral_hash(np.sin(2 * np.pi * 880 * time).astype(np.float32)) == 0x20000


def test_hashes_refuse_invalid_shapes_and_short_audio():
    with pytest.raises(ValueError, match="grayscale frame"):
        perceptual_hash(np.zeros((0, 2), dtype=np.uint8))
    with pytest.raises(ValueError, match="too short"):
        spectral_hash(np.zeros(12, dtype=np.float32))
    assert spectral_hash(np.zeros(1_000, dtype=np.float32)) == 0


def test_fingerprint_time_rotation_and_bucket_helpers_have_exact_boundaries():
    assert fingerprint_module._ending_ms(None, 500) is None
    assert fingerprint_module._ending_ms(1_250, 500) == 1_750
    assert fingerprint_module._rotation(SimpleNamespace(metadata={})) == 0
    assert fingerprint_module._rotation(SimpleNamespace(metadata={"rotate": "90"})) == 90
    assert fingerprint_module._rotation(SimpleNamespace(metadata={"rotate": "180"})) == 180
    assert fingerprint_module._rotation(SimpleNamespace(metadata={"rotate": "-90"})) == 270
    assert fingerprint_module._rotation(SimpleNamespace(metadata={"rotate": "360"})) == 0
    assert fingerprint_module._rotation(SimpleNamespace(metadata={"rotate": "bad"})) == 0
    assert fingerprint_module._majority_hash([0b0001, 0b0011, 0b0000], 4) == 0b0001
    assert fingerprint_module._majority_hash([0b0001, 0b0000], 4) == 0
    assert (
        fingerprint_module._frame_timestamp_ms(
            SimpleNamespace(pts=3), SimpleNamespace(time_base=0.25), 9, 30.0
        )
        == 750
    )
    assert (
        fingerprint_module._frame_timestamp_ms(
            SimpleNamespace(pts=None), SimpleNamespace(time_base=None), 9, 30.0
        )
        == 300
    )
    assert (
        fingerprint_module._frame_timestamp_ms(
            SimpleNamespace(pts=None), SimpleNamespace(time_base=None), 9, 0.0
        )
        == 4_500
    )
    assert fingerprint_module._container_duration_ms(SimpleNamespace(duration=None)) is None
    assert fingerprint_module._container_duration_ms(SimpleNamespace(duration=1_250_000)) == 1_250


def test_audio_window_consumption_keeps_only_the_unconsumed_tail():
    pending = np.zeros(32_123, dtype=np.float32)
    windows = []

    tail, consumed = fingerprint_module._consume_audio_windows(pending, windows)

    assert windows == [0, 0]
    assert consumed == 32_000
    assert tail.shape == (123,)
    assert fingerprint_module._resampled(None) == ()
    marker = object()
    assert fingerprint_module._resampled(marker) == (marker,)
    assert fingerprint_module._resampled([marker]) == (marker,)


class _Opened:
    def __init__(self, *, video=(), audio=(), frames=()):
        self.streams = SimpleNamespace(video=video, audio=audio)
        self._frames = frames

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def decode(self, stream):
        assert stream in (*self.streams.video, *self.streams.audio)
        return iter(self._frames)


def test_audio_decode_flushes_the_resampler_and_counts_the_partial_window(monkeypatch):
    stream = object()
    frame = object()
    opened = _Opened(audio=(stream,), frames=(frame,))
    calls = []

    class Converted:
        def __init__(self, size):
            self._values = np.zeros((1, size), dtype=np.float32)

        def to_ndarray(self):
            return self._values

    class Resampler:
        def __init__(self, **options):
            assert options == {"format": "fltp", "layout": "mono", "rate": 16_000}

        def resample(self, value):
            return [Converted(400 if value is None else 16_000)]

    def open_media(source, *, mode):
        assert source == "/selected/audio.wav"
        assert mode == "r"
        return opened

    monkeypatch.setattr(fingerprint_module.av, "open", open_media)
    monkeypatch.setattr(fingerprint_module.av, "AudioResampler", Resampler)
    token = SimpleNamespace(raise_if_cancelled=lambda: calls.append("checked"))

    windows, duration = fingerprint_module._audio_fingerprints(Path("/selected/audio.wav"), token)

    assert windows == (0, 0)
    assert duration == 1_025
    assert calls == ["checked"]


def test_audio_decode_reports_absent_and_invalid_streams(monkeypatch):
    monkeypatch.setattr(
        fingerprint_module.av,
        "open",
        lambda *_args, **_kwargs: _Opened(audio=()),
    )
    token = SimpleNamespace(raise_if_cancelled=lambda: None)
    assert fingerprint_module._audio_fingerprints(Path("empty.wav"), token) == ((), None)

    class Broken(_Opened):
        def decode(self, _stream):
            raise RuntimeError("decoder failed")

    monkeypatch.setattr(
        fingerprint_module.av,
        "open",
        lambda *_args, **_kwargs: Broken(audio=(object(),)),
    )
    with pytest.raises(MediaFingerprintError) as invalid:
        fingerprint_module._audio_fingerprints(Path("bad.wav"), token)
    assert invalid.value.code == "audio-invalid"
    assert str(invalid.value) == "audio stream could not be decoded"


def test_video_decode_rotates_clockwise_buckets_frames_and_counts_checks(monkeypatch):
    pixels = np.arange(6, dtype=np.uint8).reshape(2, 3)
    stream = SimpleNamespace(average_rate=4, time_base=0.25, metadata={"rotate": "90"})

    class Frame:
        def __init__(self, pts):
            self.pts = pts

        def to_ndarray(self, *, format):
            assert format == "gray"
            return pixels

    opened = _Opened(video=(stream,), frames=(Frame(0), Frame(1), Frame(2)))
    seen = []
    hashes = iter((1, 3, 4))

    def hashed(gray):
        seen.append(gray.copy())
        return next(hashes)

    monkeypatch.setattr(fingerprint_module.av, "open", lambda *_args, **_kwargs: opened)
    monkeypatch.setattr(fingerprint_module, "perceptual_hash", hashed)
    checks = []
    token = SimpleNamespace(raise_if_cancelled=lambda: checks.append("checked"))

    visual, duration = fingerprint_module._video_fingerprints(Path("video.mp4"), token)

    assert visual == (1, 4)
    assert duration == 1_000
    assert checks == ["checked", "checked", "checked"]
    assert all(np.array_equal(item, np.rot90(pixels, -1)) for item in seen)


def test_video_decode_reports_absent_and_invalid_streams(monkeypatch):
    token = SimpleNamespace(raise_if_cancelled=lambda: None)
    monkeypatch.setattr(
        fingerprint_module.av,
        "open",
        lambda *_args, **_kwargs: _Opened(video=()),
    )
    assert fingerprint_module._video_fingerprints(Path("empty.mp4"), token) == ((), None)

    class Broken(_Opened):
        def decode(self, _stream):
            raise RuntimeError("decoder failed")

    monkeypatch.setattr(
        fingerprint_module.av,
        "open",
        lambda *_args, **_kwargs: Broken(video=(SimpleNamespace(average_rate=4, metadata={}),)),
    )
    with pytest.raises(MediaFingerprintError) as invalid:
        fingerprint_module._video_fingerprints(Path("bad.mp4"), token)
    assert invalid.value.code == "video-invalid"
    assert str(invalid.value) == "video frames could not be decoded"


def test_exact_and_perceptual_matches_are_grouped_without_unrelated_file():
    first = media("a.mp4", "a" * 64, visual=(0x1234,) * 4, audio=(0x123,) * 2)
    transcoded = media("b.mkv", "b" * 64, visual=(0x1235,) * 4, audio=(0x123,) * 2)
    exact = media("copy.mp4", "a" * 64, visual=(0x1234,) * 4, audio=(0x123,) * 2)
    other = media("other.mp4", "c" * 64, visual=(0xFFFFFFFFFFFFFFFF,) * 4)

    groups = similarity_groups((other, transcoded, exact, first), 0.8)

    assert len(groups) == 1
    assert [item["relativePath"] for item in groups[0]["files"]] == [
        "a.mp4",
        "b.mkv",
        "copy.mp4",
    ]
    assert compare_media(first, exact).exact is True
    assert compare_media(first, transcoded).score > 0.9


def test_similarity_math_and_band_boundaries_are_exact():
    assert similarity_module._bands(0x1234, 16) == ((0, 0x34), (1, 0x12))
    assert similarity_module._hash_similarity(0b1010, 0b1000, 4) == 0.75
    assert similarity_module._majority((0b0001, 0b0011, 0b0000), 4) == 0b0001
    assert similarity_module._majority((0b0001, 0b0000), 4) == 0
    assert similarity_module._aligned_similarities((0, 1, 3), (7, 0, 1, 3), 1, 3) == (
        1.0,
        1.0,
        1.0,
    )
    assert similarity_module._bounded(-0.2) == 0.0
    assert similarity_module._bounded(0.1234567) == 0.123457
    assert similarity_module._bounded(1.2) == 1.0

    left = media("left.mp4", "a" * 64, visual=(0,), audio=(0,))
    right = media("right.mp4", "b" * 64, visual=(1,), audio=(1,))
    score = compare_media(left, right)
    assert score == similarity_module.PairScore(
        score=0.979687,
        visual_score=0.984375,
        audio_score=0.96875,
        left_coverage=1.0,
        right_coverage=1.0,
        offset_ms=0,
        exact=False,
    )
    assert compare_media(
        media("video.mp4", "c" * 64, visual=(0,)),
        media("audio.wav", "d" * 64, audio=(0,)),
    ) == similarity_module.PairScore(0.0, None, None, 0.0, 0.0, 0, False)

    audio_video = media("both.mp4", "e" * 64, visual=(0x1234,) * 4, audio=(0x123,) * 2)
    assert similarity_module._file_bands(audio_video) == {
        ("visual", 0, 0x34),
        ("visual", 1, 0x12),
        ("visual", 2, 0),
        ("visual", 3, 0),
        ("visual", 4, 0),
        ("visual", 5, 0),
        ("visual", 6, 0),
        ("visual", 7, 0),
        ("audio", 0, 0x23),
        ("audio", 1, 0x01),
        ("audio", 2, 0),
        ("audio", 3, 0),
        ("duration", "visual", 10),
        ("duration", "audio", 10),
    }


def test_candidate_index_emits_only_deterministic_representative_pairs():
    inputs = (
        media("c.mp4", "c" * 64, visual=(0x1234,) * 4),
        media("a.mp4", "a" * 64, visual=(0x1234,) * 4),
        media("b.mp4", "b" * 64, visual=(0x1234,) * 4),
        media("other.wav", "d" * 64, audio=(0xFFFF,) * 2),
    )

    assert similarity_module._candidate_pairs(inputs) == (
        ("a.mp4", "b.mp4"),
        ("a.mp4", "c.mp4"),
    )


def test_temporal_alignment_handles_a_long_shift_without_quadratic_position_pairs():
    timeline = tuple((index * 0x9E3779B97F4A7C15) & ((1 << 64) - 1) for index in range(256))
    shifted = (0xAAAAAAAAAAAAAAAA,) * 12 + timeline + (0x5555555555555555,) * 7

    score = compare_media(
        media("original.mp4", "a" * 64, visual=timeline),
        media("shifted.mp4", "b" * 64, visual=shifted),
    )

    assert score.score == score.visual_score == 0.964836
    assert score.left_coverage == 1.0
    assert score.right_coverage == 0.930909
    assert score.offset_ms == 6_000


def test_directory_candidate_comparisons_scale_linearly_for_one_large_collision_bucket(
    monkeypatch,
):
    compared = 0
    original = similarity_module.compare_media

    def counted(left, right):
        nonlocal compared
        compared += 1
        return original(left, right)

    monkeypatch.setattr(similarity_module, "compare_media", counted)
    inputs = tuple(media(f"{index:03}.mp4", "a" * 64, visual=(0x1234,) * 4) for index in range(100))

    groups = similarity_module.similarity_groups(inputs, 0.8)

    assert len(groups) == 1
    assert len(groups[0]["files"]) == 100
    assert compared == 99


@given(
    st.lists(st.integers(min_value=0, max_value=(1 << 64) - 1), min_size=1, max_size=32),
    st.lists(st.integers(min_value=0, max_value=(1 << 64) - 1), min_size=1, max_size=32),
)
def test_comparison_scores_are_bounded_and_symmetric(left_hashes, right_hashes):
    left = media("left.mp4", "a" * 64, visual=tuple(left_hashes))
    right = media("right.mp4", "b" * 64, visual=tuple(right_hashes))

    forward = compare_media(left, right)
    reverse = compare_media(right, left)

    assert 0 <= forward.score <= 1
    assert forward.score == reverse.score
    assert forward.left_coverage == reverse.right_coverage
    assert forward.right_coverage == reverse.left_coverage


def test_provider_scans_every_source_once_and_keeps_per_file_failures():
    calls = []

    def extract(path, relative_path, _cancellation):
        calls.append((path, relative_path))
        if relative_path == "broken.mp4":
            raise MediaFingerprintError("video-invalid", "bad")
        return media(relative_path, "a" * 64, visual=(0x1234,) * 4)

    plugin = MediaSimilarityPlugin(extractor=extract, clock_ms=lambda: 10)
    context = PluginContext(
        "media-similarity",
        1,
        {},
        frozenset({"files:read-selected"}),
    )
    asyncio.run(plugin.start(context))
    request = PluginRequest(
        "job-1",
        "media-similarity",
        "manual",
        {
            "sources": ["/staged/a.mp4", "/staged/b.mp4", "/staged/broken.mp4"],
            "relativePaths": ["a.mp4", "b.mp4", "broken.mp4"],
        },
        1,
        None,
    )
    progress = Progress()

    result = asyncio.run(plugin.execute(request, cancellation(), progress))

    assert result.status is PluginResultStatus.SUCCEEDED
    assert [relative for _path, relative in calls] == ["a.mp4", "b.mp4", "broken.mp4"]
    assert result.output["scannedFileCount"] == 3
    assert result.output["fingerprintedFileCount"] == 2
    assert result.output["similarFileCount"] == 2
    assert result.output["unmatchedFileCount"] == 0
    assert result.output["minimumSimilarity"] == 0.8
    assert result.output["failures"] == [{"relativePath": "broken.mp4", "code": "video-invalid"}]
    assert [(item.stage, item.fraction, item.detail) for item in progress.items] == [
        ("fingerprint", 0.0, "0 of 3 files"),
        ("fingerprint", 0.3, "1 of 3 files"),
        ("fingerprint", 0.6, "2 of 3 files"),
        ("fingerprint", 0.9, "3 of 3 files"),
        ("group", 0.95, "grouping similar media"),
        ("terminal", 1.0, "similarity groups ready"),
    ]
    assert validate_document("media-similarity-result.schema.json", result.output) == []


def test_provider_reuses_one_staged_hard_link_fingerprint_for_every_relative_name():
    calls = []

    def extract(path, relative_path, _cancellation):
        calls.append(path)
        return media(relative_path, "a" * 64, visual=(0x1234,) * 4)

    plugin = MediaSimilarityPlugin(extractor=extract, clock_ms=lambda: 10)
    asyncio.run(
        plugin.start(
            PluginContext(
                "media-similarity",
                1,
                {},
                frozenset({"files:read-selected"}),
            )
        )
    )
    request = PluginRequest(
        "job-hard-links",
        "media-similarity",
        "manual",
        {
            "sources": ["/staged/00/original.mp4", "/staged/00/original.mp4"],
            "relativePaths": ["original.mp4", "links/copy.mp4"],
        },
        1,
        None,
    )

    result = asyncio.run(plugin.execute(request, cancellation(), Progress()))

    assert calls == [Path("/staged/00/original.mp4")]
    assert [item["relativePath"] for item in result.output["groups"][0]["files"]] == [
        "links/copy.mp4",
        "original.mp4",
    ]


def test_provider_reuses_one_staged_hard_link_failure_for_every_relative_name():
    calls = []

    def extract(path, _relative_path, _cancellation):
        calls.append(path)
        raise MediaFingerprintError("media-invalid", "bad")

    plugin = MediaSimilarityPlugin(extractor=extract, clock_ms=lambda: 10)
    asyncio.run(
        plugin.start(
            PluginContext(
                "media-similarity",
                1,
                {},
                frozenset({"files:read-selected"}),
            )
        )
    )
    request = PluginRequest(
        "job-broken-hard-links",
        "media-similarity",
        "manual",
        {
            "sources": ["/staged/00/broken.mp4", "/staged/00/broken.mp4"],
            "relativePaths": ["broken.mp4", "links/broken.mp4"],
        },
        1,
        None,
    )

    result = asyncio.run(plugin.execute(request, cancellation(), Progress()))

    assert calls == [Path("/staged/00/broken.mp4")]
    assert result.output["failures"] == [
        {"relativePath": "broken.mp4", "code": "media-invalid"},
        {"relativePath": "links/broken.mp4", "code": "media-invalid"},
    ]


def test_provider_refuses_misaligned_relative_paths():
    plugin = MediaSimilarityPlugin(clock_ms=lambda: 10)
    context = PluginContext(
        "media-similarity",
        1,
        {},
        frozenset({"files:read-selected"}),
    )
    asyncio.run(plugin.start(context))
    request = PluginRequest(
        "job-1",
        "media-similarity",
        "manual",
        {"sources": ["/staged/a.mp4"], "relativePaths": ["../a.mp4"]},
        1,
        None,
    )

    result = asyncio.run(plugin.execute(request, cancellation(), Progress()))

    assert result.status is PluginResultStatus.FAILED
    assert result.detail == "paths-invalid"


def request(payload, plugin_id="media-similarity"):
    return PluginRequest("job-boundary", plugin_id, "manual", payload, 1, None)


def test_request_validation_returns_the_exact_normalized_contract():
    assert _validated_request(
        request(
            {
                "sources": ["/staged/a.mp4", "/staged/b.wav"],
                "relativePaths": ["a.mp4", "nested/b.wav"],
                "minimumSimilarity": 0.75,
            }
        )
    ) == (
        ("/staged/a.mp4", "/staged/b.wav"),
        ("a.mp4", "nested/b.wav"),
        0.75,
    )
    assert _validated_request(
        request({"sources": ["/staged/a.mp4"], "relativePaths": ["a.mp4"]})
    ) == (("/staged/a.mp4",), ("a.mp4",), 0.8)


@pytest.mark.parametrize(
    ("candidate", "code"),
    [
        (object(), "request-invalid"),
        (request({"sources": ["a"], "relativePaths": ["a"]}, "another"), "request-invalid"),
        (request({"sources": [], "relativePaths": []}), "sources-invalid"),
        (request({"sources": ("a",), "relativePaths": ["a"]}), "sources-invalid"),
        (request({"sources": [1], "relativePaths": ["a"]}), "sources-invalid"),
        (request({"sources": ["a"], "relativePaths": []}), "paths-invalid"),
        (request({"sources": ["a", "b"], "relativePaths": ["same", "same"]}), "paths-invalid"),
        (
            request({"sources": ["a"], "relativePaths": ["a"], "minimumSimilarity": True}),
            "similarity-invalid",
        ),
        (
            request({"sources": ["a"], "relativePaths": ["a"], "minimumSimilarity": -0.1}),
            "similarity-invalid",
        ),
        (
            request({"sources": ["a"], "relativePaths": ["a"], "minimumSimilarity": 1.1}),
            "similarity-invalid",
        ),
    ],
)
def test_request_validation_refuses_each_boundary_with_a_stable_code(candidate, code):
    with pytest.raises(MediaSimilarityError) as invalid:
        _validated_request(candidate)

    assert invalid.value.code == code
    assert str(invalid.value)


@pytest.mark.parametrize(
    "value",
    ["", "/absolute.mp4", "../escape.mp4", "nested/../escape.mp4", "back\\slash.mp4", None],
)
def test_relative_path_boundary_refuses_unsafe_names(value):
    assert _safe_relative_path(value) is False


def test_relative_path_boundary_accepts_nested_relative_names():
    assert _safe_relative_path("nested/media clip.mp4") is True


def test_provider_health_is_ready_without_an_accelerator_lease():
    plugin = MediaSimilarityPlugin(clock_ms=lambda: 123)
    asyncio.run(
        plugin.start(
            PluginContext(
                "media-similarity",
                1,
                {},
                frozenset({"files:read-selected"}),
            )
        )
    )

    health = asyncio.run(plugin.health())

    assert health.status.value == "ready"
    assert health.detail == "ready for directory-scale audio and video similarity"
    assert health.checked_at_ms == 123


def test_real_video_fingerprint_survives_resolution_change(tmp_path):
    first = _video(tmp_path / "small.mp4", 64, 48)
    second = _video(tmp_path / "large.mp4", 192, 144)

    left = fingerprint_file(first, "small.mp4", cancellation())
    right = fingerprint_file(second, "large.mp4", cancellation())

    assert left.visual and right.visual
    assert len(left.visual) == len(right.visual) == 4
    assert left.audio == right.audio == ()
    assert left.duration_ms == right.duration_ms == 2_000
    assert left.modality == right.modality == "video"
    assert left.sha256 == hashlib.sha256(first.read_bytes()).hexdigest()
    assert compare_media(left, right).visual_score > 0.9


def test_real_resolution_variants_flow_through_the_provider_and_result_schema(tmp_path):
    first = _video(tmp_path / "small.mp4", 64, 48)
    second = _video(tmp_path / "large.mp4", 192, 144)
    plugin = MediaSimilarityPlugin(clock_ms=lambda: 10)
    asyncio.run(
        plugin.start(
            PluginContext(
                "media-similarity",
                1,
                {},
                frozenset({"files:read-selected"}),
            )
        )
    )
    request = PluginRequest(
        "job-real",
        "media-similarity",
        "manual",
        {
            "sources": [str(first), str(second)],
            "relativePaths": ["small.mp4", "nested/large.mp4"],
        },
        1,
        None,
    )

    result = asyncio.run(plugin.execute(request, cancellation(), Progress()))

    assert result.status is PluginResultStatus.SUCCEEDED
    assert [item["relativePath"] for item in result.output["groups"][0]["files"]] == [
        "nested/large.mp4",
        "small.mp4",
    ]
    assert validate_document("media-similarity-result.schema.json", result.output) == []


def test_real_audio_fingerprint_survives_gain_change(tmp_path):
    first = fingerprint_file(_wav(tmp_path / "first.wav", 0.8), "first.wav", cancellation())
    second = fingerprint_file(_wav(tmp_path / "second.wav", 0.2), "second.wav", cancellation())

    assert first.audio and second.audio
    assert len(first.audio) == len(second.audio) == 2
    assert first.duration_ms == second.duration_ms == 1_250
    assert first.visual == second.visual == ()
    assert first.modality == second.modality == "audio"
    assert compare_media(first, second).audio_score > 0.9


def test_fingerprint_rejects_unsupported_and_broken_media(tmp_path):
    unsupported = tmp_path / "notes.txt"
    unsupported.write_text("text", encoding="utf-8")
    broken = tmp_path / "broken.mp4"
    broken.write_bytes(b"not a container")

    with pytest.raises(MediaFingerprintError) as wrong_type:
        fingerprint_file(unsupported, "notes.txt", cancellation())
    assert wrong_type.value.code == "media-unsupported"
    with pytest.raises(MediaFingerprintError) as invalid:
        fingerprint_file(broken, "broken.mp4", cancellation())
    assert invalid.value.code == "media-invalid"


def test_fingerprint_refuses_a_declared_stream_with_no_decodable_content(tmp_path, monkeypatch):
    source = _video(tmp_path / "declared.mp4", 64, 48)
    monkeypatch.setattr(fingerprint_module, "_video_fingerprints", lambda *_args: ((), None))

    with pytest.raises(MediaFingerprintError) as invalid:
        fingerprint_file(source, "declared.mp4", cancellation())

    assert invalid.value.code == "media-invalid"


def _video(path: Path, width: int, height: int) -> Path:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for index in range(8):
            pixels = np.zeros((height, width, 3), dtype=np.uint8)
            left = round(width * (0.1 + (index * 0.03)))
            pixels[height // 4 : height * 3 // 4, left : left + width // 3] = (220, 90, 30)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return path


def _wav(path: Path, gain: float) -> Path:
    samples = (np.sin(2 * np.pi * 440 * np.arange(20_000) / 16_000) * 30_000 * gain).astype(
        np.int16
    )
    with av.open(str(path), "w") as container:
        stream = container.add_stream("pcm_s16le", rate=16_000)
        frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = 16_000
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return path
