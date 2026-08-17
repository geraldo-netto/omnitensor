from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.plugins.media_transcription import (
    MAX_DOCUMENT_PAGES,
    MAX_DURATION_MS,
    MAX_IMAGE_PIXELS,
    MAX_LANGUAGE_CHARACTERS,
    MAX_PRESENTATION_SLIDES,
    MAX_SOURCE_BYTES,
    MAX_VIDEO_DURATION_MS,
    FrameSampler,
    MediaInfo,
    MediaModality,
    MediaProviderIdentity,
    MediaTranscriptionError,
    MediaTranscriptionPlugin,
    SpeechSegment,
    SpeechTranscript,
    VisualFrame,
    VisualTranscript,
    _bounded_content,
    _validated_media,
    _validated_speech,
    _validated_visual,
    _validated_visuals,
    media_result,
)
from omnitensor.plugins.protocol import PluginContext, PluginRequest, PluginResultStatus
from omnitensor.registry import validate_document
from omnitensor.sdk import CancellationController


class Progress:
    def __init__(self):
        self.items = []

    async def report(self, item):
        self.items.append(item)


class Probe:
    def __init__(self, media):
        self.media = media
        self.sources = []

    async def inspect(self, source):
        self.sources.append(source)
        if isinstance(self.media, Exception):
            raise self.media
        return self.media


class Speech:
    def __init__(self, transcript=None):
        self.transcript = transcript or SpeechTranscript(
            "en", (SpeechSegment(0, 900, "Hello world"),)
        )
        self.sources = []

    async def transcribe(self, source, cancellation):
        cancellation.raise_if_cancelled()
        self.sources.append(source)
        return self.transcript


class Frames:
    def __init__(self, root, count=1, *, timestamped=False):
        self.frames = tuple(
            VisualFrame(
                root / f"frame-{index}.png",
                index * 1000 if timestamped or count > 1 else None,
            )
            for index in range(count)
        )
        for frame in self.frames:
            frame.path.write_bytes(b"frame")
        self.discards = []

    async def sample(self, source, media, cancellation):
        cancellation.raise_if_cancelled()
        return self.frames

    async def discard(self, frames):
        self.discards.append(tuple(frames))


class Vision:
    def __init__(self, *, crash=False):
        self.crash = crash
        self.frames = []
        self.releases = 0

    async def transcribe(self, frame, cancellation):
        cancellation.raise_if_cancelled()
        self.frames.append(frame)
        if self.crash:
            raise RuntimeError("private decoder failure")
        return VisualTranscript(frame.timestamp_ms, "שלום", "A cat beside a sign.")

    async def release(self):
        self.releases += 1


class Presentations:
    def __init__(self, transcripts=()):
        self.transcripts = tuple(transcripts)
        self.sources = []

    async def transcribe(self, source, cancellation):
        cancellation.raise_if_cancelled()
        self.sources.append(source)
        return self.transcripts


class Documents(Presentations):
    pass


class BadFrames:
    async def sample(self, _source, _media, _cancellation):
        return ()

    async def discard(self, _frames):
        raise RuntimeError("cleanup failed")


def request(source, *, plugin_id="media-transcription", trigger="manual", payload=None):
    return PluginRequest(
        "media-job-1",
        plugin_id,
        trigger,
        payload if payload is not None else {"sources": [str(source)]},
        1,
        None,
    )


async def running(
    tmp_path,
    media,
    *,
    speech=None,
    frames=None,
    vision=None,
    presentations=None,
    documents=None,
    preflight=None,
):
    ports = {
        "probe": Probe(media),
        "speech": speech or Speech(),
        "frames": frames
        or Frames(
            tmp_path,
            timestamped=isinstance(media, MediaInfo) and media.modality is MediaModality.VIDEO,
        ),
        "vision": vision or Vision(),
        "presentations": presentations or Presentations(),
        "documents": documents or Documents(),
    }
    plugin = MediaTranscriptionPlugin(
        identity=MediaProviderIdentity("media-vulkan", "gpu"),
        **ports,
        preflight=preflight,
        clock_ms=lambda: 10,
    )
    await plugin.start(
        PluginContext("media-transcription", 1, {}, frozenset({"files:read-selected"}))
    )
    return plugin, ports


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("media", "speech_count", "visual_count"),
    [
        (MediaInfo(MediaModality.AUDIO, 1000, None, None, True), 1, 0),
        (MediaInfo(MediaModality.IMAGE, None, 640, 480, False), 0, 1),
        (MediaInfo(MediaModality.VIDEO, 2000, 640, 480, True), 1, 1),
    ],
)
async def test_audio_image_and_video_produce_exact_bounded_results(
    tmp_path, media, speech_count, visual_count
):
    source = tmp_path / f"source.{media.modality}"
    source.write_bytes(b"selected media")
    plugin, ports = await running(tmp_path, media)
    progress = Progress()

    result = await plugin.execute(request(source), CancellationController(), progress)

    assert result.status is PluginResultStatus.SUCCEEDED
    assert result.detail == "media transcription ready"
    assert validate_document("media-transcription-result.schema.json", result.output) == []
    assert result.output["providerId"] == "media-vulkan"
    assert result.output["accelerator"] == "gpu"
    assert result.output["source"]["fileName"] == source.name
    assert result.output["source"]["modality"] == media.modality
    assert len(result.output["speech"]["segments"]) == speech_count
    assert len(result.output["visuals"]) == visual_count
    assert ports["speech"].sources == ([source] if speech_count else [])
    assert len(ports["vision"].frames) == visual_count
    assert ports["frames"].discards == [ports["frames"].frames if visual_count else ()]
    assert ports["vision"].releases == 1
    assert progress.items[0].stage == "inspect"
    assert progress.items[-1].stage == "terminal"
    assert progress.items[-1].fraction == 1.0


@pytest.mark.asyncio
async def test_video_transcribes_each_bounded_frame_in_timestamp_order(tmp_path):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video")
    frames = Frames(tmp_path, count=3)
    plugin, _ports = await running(
        tmp_path,
        MediaInfo(MediaModality.VIDEO, 4000, 320, 200, False),
        frames=frames,
    )
    progress = Progress()

    result = await plugin.execute(request(source), CancellationController(), progress)

    assert result.status is PluginResultStatus.SUCCEEDED
    assert [item["timestampMs"] for item in result.output["visuals"]] == [0, 1000, 2000]
    assert result.output["speech"] == {"language": None, "segments": []}
    assert [item.fraction for item in progress.items if item.stage == "vision"] == pytest.approx(
        [0.6 + 0.35 / 3, 0.6 + 0.7 / 3, 0.95]
    )


@pytest.mark.asyncio
async def test_presentation_transcribes_exact_ordered_slides_without_frame_sampling(tmp_path):
    source = tmp_path / "briefing.pptx"
    source.write_bytes(b"presentation")
    presentations = Presentations(
        (
            VisualTranscript(None, "שלום", "Title slide.", 1),
            VisualTranscript(None, "Привет", "Chart slide.", 2),
        )
    )
    plugin, ports = await running(
        tmp_path,
        MediaInfo(MediaModality.PRESENTATION, None, None, None, False, 2),
        presentations=presentations,
    )

    result = await plugin.execute(request(source), CancellationController(), Progress())

    assert result.status is PluginResultStatus.SUCCEEDED
    assert [item["slideNumber"] for item in result.output["visuals"]] == [1, 2]
    assert [item["visibleText"] for item in result.output["visuals"]] == ["שלום", "Привет"]
    assert presentations.sources == [source]
    assert ports["speech"].sources == []
    assert ports["frames"].discards == [()]
    assert ports["vision"].frames == []


@pytest.mark.asyncio
async def test_document_transcribes_exact_ordered_pages_without_frame_sampling(tmp_path):
    source = tmp_path / "report.pdf"
    source.write_bytes(b"document")
    documents = Documents(
        (
            VisualTranscript(None, "שלום", "Cover page.", None, 1),
            VisualTranscript(None, "Привет", "Chart page.", None, 2),
        )
    )
    plugin, ports = await running(
        tmp_path,
        MediaInfo(MediaModality.DOCUMENT, None, None, None, False, None, 2),
        documents=documents,
    )

    result = await plugin.execute(request(source), CancellationController(), Progress())

    assert result.status is PluginResultStatus.SUCCEEDED
    assert [item["pageNumber"] for item in result.output["visuals"]] == [1, 2]
    assert [item["slideNumber"] for item in result.output["visuals"]] == [None, None]
    assert documents.sources == [source]
    assert ports["speech"].sources == []
    assert ports["frames"].discards == [()]


@pytest.mark.asyncio
async def test_start_requires_permission_ports_identity_and_preflight(tmp_path):
    media = MediaInfo(MediaModality.IMAGE, None, 1, 1, False)
    good = dict(
        probe=Probe(media),
        speech=Speech(),
        frames=Frames(tmp_path),
        vision=Vision(),
        presentations=Presentations(),
        documents=Documents(),
    )
    with pytest.raises(MediaTranscriptionError, match="provider-invalid"):
        MediaTranscriptionPlugin(identity=MediaProviderIdentity("Bad", "cpu"), **good)
    for name in good:
        candidate = dict(good)
        candidate[name] = object()
        with pytest.raises(TypeError, match="port is required"):
            MediaTranscriptionPlugin(
                identity=MediaProviderIdentity("media-vulkan", "gpu"), **candidate
            )
    with pytest.raises(TypeError, match="preflight"):
        MediaTranscriptionPlugin(
            identity=MediaProviderIdentity("media-vulkan", "gpu"), **good, preflight=1
        )
    with pytest.raises(TypeError, match="clock"):
        MediaTranscriptionPlugin(
            identity=MediaProviderIdentity("media-vulkan", "gpu"), **good, clock_ms=1
        )
    plugin = MediaTranscriptionPlugin(identity=MediaProviderIdentity("media-vulkan", "gpu"), **good)
    with pytest.raises(Exception, match="permission is not granted"):
        await plugin.start(PluginContext("media-transcription", 1, {}, frozenset()))
    failed = MediaTranscriptionPlugin(
        identity=MediaProviderIdentity("media-vulkan", "gpu"),
        **good,
        preflight=lambda: (_ for _ in ()).throw(RuntimeError("not qualified")),
    )
    with pytest.raises(RuntimeError, match="not qualified"):
        await failed.start(
            PluginContext("media-transcription", 1, {}, frozenset({"files:read-selected"}))
        )

    async def asynchronous_preflight():
        return None

    ready, _ports = await running(tmp_path, media, preflight=asynchronous_preflight)
    health = await ready.health()
    assert health.status.value == "ready"
    assert health.checked_at_ms == 10


@pytest.mark.asyncio
async def test_request_and_source_failures_are_content_free(tmp_path):
    source = tmp_path / "image.png"
    source.write_bytes(b"image")
    plugin, _ports = await running(tmp_path, MediaInfo(MediaModality.IMAGE, None, 1, 1, False))
    candidates = [
        (request(source, plugin_id="other"), "request-invalid"),
        (request(source, trigger="periodic"), "request-invalid"),
        (request(source, payload={}), "source-invalid"),
        (request(source, payload={"sources": []}), "source-invalid"),
        (request(source, payload={"sources": [str(source), str(source)]}), "source-invalid"),
        (request(source, payload={"sources": [1]}), "source-invalid"),
        (request(tmp_path / "missing.wav"), "source-unavailable"),
    ]
    for candidate, code in candidates:
        result = await plugin.execute(candidate, CancellationController(), Progress())
        assert result.status is PluginResultStatus.FAILED
        assert result.detail == code
    empty = tmp_path / "empty.wav"
    empty.touch()
    result = await plugin.execute(request(empty), CancellationController(), Progress())
    assert result.detail == "source-invalid"


@pytest.mark.asyncio
async def test_probe_provider_frame_and_cleanup_failures_are_isolated(tmp_path):
    source = tmp_path / "media.bin"
    source.write_bytes(b"private bytes")
    plugin, _ports = await running(tmp_path, RuntimeError("contains private bytes"))
    result = await plugin.execute(request(source), CancellationController(), Progress())
    assert result.status is PluginResultStatus.FAILED
    assert result.detail == "media-transcription-failed"

    plugin, _ports = await running(
        tmp_path,
        MediaInfo(MediaModality.IMAGE, None, 10, 10, False),
        frames=BadFrames(),
    )
    result = await plugin.execute(request(source), CancellationController(), Progress())
    assert result.detail == "frames-invalid"

    frames = Frames(tmp_path)
    plugin, _ports = await running(
        tmp_path,
        MediaInfo(MediaModality.IMAGE, None, 10, 10, False),
        frames=frames,
        vision=Vision(crash=True),
    )
    result = await plugin.execute(request(source), CancellationController(), Progress())
    assert result.detail == "media-transcription-failed"
    assert frames.discards == [frames.frames]


@pytest.mark.asyncio
async def test_cancelled_request_returns_terminal_cancelled_and_discards_frames(tmp_path):
    source = tmp_path / "image.png"
    source.write_bytes(b"image")
    frames = Frames(tmp_path)
    plugin, _ports = await running(
        tmp_path,
        MediaInfo(MediaModality.IMAGE, None, 10, 10, False),
        frames=frames,
    )
    cancellation = CancellationController()
    cancellation.cancel("test")

    result = await plugin.execute(request(source), cancellation, Progress())

    assert result.status is PluginResultStatus.CANCELLED
    assert result.detail == "media transcription cancelled"
    assert frames.discards == [()]


@pytest.mark.parametrize(
    "media",
    [
        object(),
        MediaInfo(MediaModality.AUDIO, None, None, None, True),
        MediaInfo(MediaModality.AUDIO, 0, None, None, True),
        MediaInfo(MediaModality.AUDIO, 1, None, None, False),
        MediaInfo(MediaModality.IMAGE, 1, 1, 1, False),
        MediaInfo(MediaModality.IMAGE, None, 0, 1, False),
        MediaInfo(MediaModality.IMAGE, None, MAX_IMAGE_PIXELS + 1, 1, False),
        MediaInfo(MediaModality.VIDEO, None, 1, 1, False),
        MediaInfo(MediaModality.VIDEO, MAX_VIDEO_DURATION_MS + 1, 1, 1, False),
        MediaInfo(MediaModality.AUDIO, MAX_DURATION_MS + 1, None, None, True),
        MediaInfo(MediaModality.AUDIO, True, None, None, True),
        MediaInfo(MediaModality.AUDIO, 1, None, None, True, 1),
        MediaInfo(MediaModality.IMAGE, None, 1, 1, False, None, 1),
        MediaInfo(MediaModality.PRESENTATION, None, None, None, False, None),
        MediaInfo(MediaModality.PRESENTATION, None, None, None, False, 0),
        MediaInfo(
            MediaModality.PRESENTATION,
            None,
            None,
            None,
            False,
            MAX_PRESENTATION_SLIDES + 1,
        ),
        MediaInfo(MediaModality.PRESENTATION, None, 1, None, False, 1),
        MediaInfo(MediaModality.PRESENTATION, None, None, None, True, 1),
        MediaInfo(MediaModality.DOCUMENT, None, None, None, False, None, None),
        MediaInfo(MediaModality.DOCUMENT, None, None, None, False, None, 0),
        MediaInfo(
            MediaModality.DOCUMENT,
            None,
            None,
            None,
            False,
            None,
            MAX_DOCUMENT_PAGES + 1,
        ),
        MediaInfo(MediaModality.DOCUMENT, None, 1, None, False, None, 1),
    ],
)
def test_media_validation_refuses_inconsistent_or_unbounded_metadata(media):
    with pytest.raises(MediaTranscriptionError):
        _validated_media(media)


def test_transcript_validation_bounds_text_language_timing_and_visuals():
    media = MediaInfo(MediaModality.AUDIO, 1000, None, None, True)
    assert (
        _validated_speech(SpeechTranscript("he-IL", (SpeechSegment(0, 1000, " שלום "),)), media)
        .segments[0]
        .text
        == "שלום"
    )
    assert (
        len(
            _validated_speech(SpeechTranscript("he", (SpeechSegment(0, 1000, "x" * 4096),)), media)
            .segments[0]
            .text
        )
        == 4096
    )
    bad_speech = [
        object(),
        SpeechTranscript("not_language!", ()),
        SpeechTranscript("en-abcdefgh-abcdefgh-abcdefgh-abcdefgh", ()),
        SpeechTranscript("en", (SpeechSegment(0, 0, "text"),)),
        SpeechTranscript("en", (SpeechSegment(0, 1001, "text"),)),
        SpeechTranscript("en", (SpeechSegment(0, 500, "first"), SpeechSegment(400, 600, "next"))),
        SpeechTranscript("en", (SpeechSegment(0, 500, " "),)),
    ]
    for value in bad_speech:
        with pytest.raises(MediaTranscriptionError):
            _validated_speech(value, media)
    assert _validated_visual(VisualTranscript(None, " שלום ", " cat ")) == VisualTranscript(
        None, "שלום", "cat"
    )
    for value in [
        object(),
        VisualTranscript(-1, "", "cat"),
        VisualTranscript(True, "", "cat"),
        VisualTranscript(None, "", " "),
        VisualTranscript(None, "\x00", "cat"),
        VisualTranscript(None, "", "cat", 0),
        VisualTranscript(None, "", "cat", True),
        VisualTranscript(None, "", "cat", MAX_PRESENTATION_SLIDES + 1),
        VisualTranscript(None, "", "cat", None, 0),
        VisualTranscript(None, "", "cat", None, True),
        VisualTranscript(None, "", "cat", None, MAX_DOCUMENT_PAGES + 1),
    ]:
        with pytest.raises(MediaTranscriptionError):
            _validated_visual(value)
    for value, required in [(1, False), (" ", True), ("\x00", False)]:
        with pytest.raises(MediaTranscriptionError):
            _bounded_content(value, required=required)


def test_visual_sets_enforce_modality_position_and_presentation_slide_order():
    image = MediaInfo(MediaModality.IMAGE, None, 1, 1, False)
    video = MediaInfo(MediaModality.VIDEO, 1000, 1, 1, False)
    presentation = MediaInfo(MediaModality.PRESENTATION, None, None, None, False, 2)
    document = MediaInfo(MediaModality.DOCUMENT, None, None, None, False, None, 2)
    valid_slides = (
        VisualTranscript(None, "שלום", "Title.", 1),
        VisualTranscript(None, "Привет", "Image.", 2),
    )
    assert _validated_visuals(valid_slides, presentation) == valid_slides
    valid_pages = (
        VisualTranscript(None, "שלום", "Cover.", None, 1),
        VisualTranscript(None, "Привет", "Chart.", None, 2),
    )
    assert _validated_visuals(valid_pages, document) == valid_pages
    invalid = (
        ((), image),
        ((VisualTranscript(None, "", "image", 1),), image),
        ((VisualTranscript(None, "", "video"),), video),
        ((VisualTranscript(0, "", "video", 1),), video),
        ((VisualTranscript(500, "", "video"), VisualTranscript(0, "", "video")), video),
        ((valid_slides[1], valid_slides[0]), presentation),
        ((valid_slides[0],), presentation),
        ((VisualTranscript(0, "", "slide", 1), valid_slides[1]), presentation),
        ((valid_pages[1], valid_pages[0]), document),
        ((valid_pages[0],), document),
        ((VisualTranscript(0, "", "page", None, 1), valid_pages[1]), document),
    )
    for visuals, media in invalid:
        with pytest.raises(MediaTranscriptionError, match="visual-invalid"):
            _validated_visuals(visuals, media)


def test_media_result_refuses_unreadable_or_contract_invalid_source(tmp_path):
    source = tmp_path / "image.png"
    source.write_bytes(b"image")
    media = MediaInfo(MediaModality.IMAGE, None, 1, 1, False)
    identity = MediaProviderIdentity("media-vulkan", "gpu")
    valid = media_result(
        "job-1",
        source,
        media,
        identity,
        SpeechTranscript(None, ()),
        (VisualTranscript(None, "", "image"),),
    )
    assert valid["source"]["sourceSha256"]
    source.unlink()
    with pytest.raises(MediaTranscriptionError, match="source-unavailable"):
        media_result(
            "job-1",
            source,
            media,
            identity,
            SpeechTranscript(None, ()),
            (),
        )
    source.write_bytes(b"image")
    with pytest.raises(MediaTranscriptionError, match="result-invalid"):
        media_result(
            "bad id",
            source,
            media,
            identity,
            SpeechTranscript(None, ()),
            (),
        )


@given(
    width=st.integers(min_value=1, max_value=10_000),
    height=st.integers(min_value=1, max_value=10_000),
)
def test_image_pixel_boundary_is_total(width, height):
    media = MediaInfo(MediaModality.IMAGE, None, width, height, False)
    if width * height <= MAX_IMAGE_PIXELS:
        assert _validated_media(media) is media
    else:
        with pytest.raises(MediaTranscriptionError, match="image-invalid"):
            _validated_media(media)


@given(duration=st.integers(min_value=0, max_value=MAX_DURATION_MS))
def test_timed_media_duration_matches_public_schema_minimum(duration):
    media = MediaInfo(MediaModality.AUDIO, duration, None, None, True)
    if duration >= 1:
        assert _validated_media(media) is media
    else:
        with pytest.raises(MediaTranscriptionError):
            _validated_media(media)


@given(
    subtags=st.lists(
        st.text(
            alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
            min_size=2,
            max_size=8,
        ),
        max_size=8,
    )
)
def test_language_length_matches_public_schema_maximum(subtags):
    language = "-".join(("en", *subtags))
    transcript = SpeechTranscript(language, ())
    media = MediaInfo(MediaModality.AUDIO, 1, None, None, True)
    if len(language) <= MAX_LANGUAGE_CHARACTERS:
        assert _validated_speech(transcript, media).language == language
    else:
        with pytest.raises(MediaTranscriptionError):
            _validated_speech(transcript, media)


@given(
    start=st.integers(min_value=0, max_value=999),
    end=st.integers(min_value=1, max_value=1000),
    text=st.text(min_size=0, max_size=32),
)
def test_speech_segment_boundary_is_total(start, end, text):
    value = SpeechTranscript("he", (SpeechSegment(start, end, text),))
    media = MediaInfo(MediaModality.AUDIO, 1000, None, None, True)
    valid = start < end and bool(text.strip()) and "\x00" not in text
    if valid:
        assert _validated_speech(value, media).segments[0].text == text.strip()
    else:
        with pytest.raises(MediaTranscriptionError):
            _validated_speech(value, media)


def test_runtime_protocols_remain_structural(tmp_path):
    assert isinstance(Frames(tmp_path), FrameSampler)
    assert MAX_SOURCE_BYTES == 128 * 1024 * 1024
