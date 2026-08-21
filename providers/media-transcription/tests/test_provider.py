from __future__ import annotations

import asyncio
import base64
import io
import json
import shutil
import subprocess
import sys
import types
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import omnitensor_media_transcription.documents as documents
import omnitensor_media_transcription.presentations as presentations
import omnitensor_media_transcription.provider as provider
import pytest
from PIL import Image, ImageDraw, ImageFont

from omnitensor.plugins.media_transcription import (
    MediaInfo,
    MediaModality,
    MediaTranscriptionError,
    VisualFrame,
    VisualTranscript,
)
from omnitensor.sdk import BootstrapArtifact, CancellationController, PluginBootstrap


def _wav(path: Path, seconds: float = 0.1) -> Path:
    import av

    samples = (np.sin(np.arange(int(16_000 * seconds)) / 8) * 10_000).astype(np.int16)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("pcm_s16le", rate=16_000)
        frame = av.AudioFrame.from_ndarray(samples.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = 16_000
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return path


def _video(path: Path, frame_count: int = 4) -> Path:
    import av

    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"
        for index in range(frame_count):
            pixels = np.full((48, 64, 3), index * 40, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)
    return path


def _png_bytes() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (16, 8), "blue").save(stream, format="PNG")
    return stream.getvalue()


def _complex_raster(path: Path, image_format: str | None = None) -> Path:
    image = Image.new("RGB", (640, 420), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    draw.rectangle((20, 20, 620, 400), outline="navy", width=4)
    draw.text((40, 40), "שלום · Привет · résumé", fill="black", font=font)
    for index, height in enumerate((80, 150, 240, 110)):
        left = 60 + index * 120
        draw.rectangle((left, 360 - height, left + 70, 360), fill=(30, 80 + index * 30, 180))
    draw.line((40, 360, 600, 360), fill="black", width=3)
    image.save(path, format=image_format)
    return path


def _complex_svg(path: Path) -> Path:
    encoded = base64.b64encode(_png_bytes()).decode("ascii")
    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="640" height="420">
<rect width="640" height="420" fill="white"/>
<path d="M40 360 L160 220 L280 300 L420 100 L600 180"
 fill="none" stroke="navy" stroke-width="8"/>
<text x="40" y="60" font-size="28">שלום · Привет · résumé</text>
<image x="450" y="250" width="100" height="50" href="data:image/png;base64,{encoded}"/>
</svg>""",
        encoding="utf-8",
    )
    return path


def _complex_pdf(path: Path) -> Path:
    import pymupdf

    document = pymupdf.open()
    for number in range(2):
        page = document.new_page(width=640, height=420)
        page.insert_text((40, 50), f"Complex report page {number + 1}", fontsize=24)
        page.draw_rect(pymupdf.Rect(40, 90, 600, 360), color=(0, 0, 0.6), width=3)
        page.insert_image(pymupdf.Rect(420, 230, 580, 310), stream=_png_bytes())
    document.save(path)
    document.close()
    return path


def _complex_tiff(path: Path) -> Path:
    pages = []
    for color in ("white", "lightyellow"):
        page = Image.new("RGB", (640, 420), color)
        draw = ImageDraw.Draw(page)
        draw.rectangle((30, 30, 610, 390), outline="navy", width=5)
        draw.ellipse((180, 100, 460, 350), fill="royalblue")
        pages.append(page)
    pages[0].save(path, save_all=True, append_images=pages[1:], compression="tiff_lzw")
    return path


def _pptx(path: Path, *, image: bytes | None = None) -> Path:
    presentation = """<?xml version="1.0" encoding="UTF-8"?>
<p:presentation xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
 <p:sldIdLst><p:sldId id="256" r:id="rId1"/></p:sldIdLst>
</p:presentation>"""
    presentation_rels = """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rId1" Target="slides/slide1.xml" Type="slide"/>
</Relationships>"""
    image_xml = '<a:blip r:embed="rIdImage"/>' if image is not None else ""
    slide = f"""<?xml version="1.0" encoding="UTF-8"?>
<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
 xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
 xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
 <p:cSld><a:t>שלום</a:t><a:t>Привет</a:t>{image_xml}</p:cSld>
</p:sld>"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ppt/presentation.xml", presentation)
        archive.writestr("ppt/_rels/presentation.xml.rels", presentation_rels)
        archive.writestr("ppt/slides/slide1.xml", slide)
        if image is not None:
            archive.writestr(
                "ppt/slides/_rels/slide1.xml.rels",
                """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
 <Relationship Id="rIdImage" Target="../media/image1.png" Type="image"/>
</Relationships>""",
            )
            archive.writestr("ppt/media/image1.png", image)
    return path


def _odp(path: Path, *, image: bytes | None = None) -> Path:
    image_xml = '<draw:image xlink:href="Pictures/image.png"/>' if image is not None else ""
    content = f"""<?xml version="1.0" encoding="UTF-8"?>
<office:document-content
 xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
 xmlns:xlink="http://www.w3.org/1999/xlink">
 <office:body><office:presentation><draw:page>
  <text:p>שלום</text:p><text:p>Привет</text:p>{image_xml}
 </draw:page></office:presentation></office:body>
</office:document-content>"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("content.xml", content)
        if image is not None:
            archive.writestr("Pictures/image.png", image)
    return path


def _ffmpeg(path: Path, arguments: list[str]) -> Path:
    completed = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *arguments, str(path)],
        check=False,
        capture_output=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
    return path


def _encoded_audio(path: Path, codec: str) -> Path:
    return _ffmpeg(
        path,
        ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000", "-t", "0.25", "-c:a", codec],
    )


def _encoded_video(path: Path, video_codec: str, audio_codec: str | None) -> Path:
    arguments = [
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=96x64:rate=4",
    ]
    if audio_codec is not None:
        arguments.extend(["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000"])
    arguments.extend(["-t", "0.5", "-c:v", video_codec])
    if audio_codec is not None:
        arguments.extend(["-c:a", audio_codec, "-shortest"])
    return _ffmpeg(path, arguments)


def test_vulkan_lease_requires_file_and_serializes_access(tmp_path):
    lease_path = tmp_path / "gpu.lock"
    lease_path.touch()
    lease = provider.VulkanLease(lease_path)
    stream = lease.acquire()
    assert not stream.closed
    lease.release(stream)
    assert stream.closed
    with lease.hold():
        assert lease_path.is_file()
    with pytest.raises(provider.MediaGpuError, match="lease is unavailable"):
        provider.VulkanLease(tmp_path / "missing")


@pytest.mark.asyncio
async def test_av_adapter_decodes_real_image_audio_and_video(tmp_path):
    adapter = provider.AvMediaAdapter()
    image = tmp_path / "scan.png"
    Image.new("RGB", (32, 16), "blue").save(image)
    image_info = await adapter.inspect(image)
    assert image_info == MediaInfo(MediaModality.IMAGE, None, 32, 16, False)
    assert await adapter.sample(image, image_info, CancellationController()) == (
        VisualFrame(image, None),
    )

    audio = _wav(tmp_path / "voice.wav")
    audio_info = await adapter.inspect(audio)
    assert audio_info.modality is MediaModality.AUDIO
    decoded = await adapter.decode_audio(audio, CancellationController())
    assert decoded.dtype == np.float32
    assert 1000 <= decoded.size <= 2000

    video = _video(tmp_path / "clip.mp4")
    video_info = await adapter.inspect(video)
    assert video_info.modality is MediaModality.VIDEO
    frames = await adapter.sample(video, video_info, CancellationController())
    assert len(frames) == 1
    assert frames[0].path.is_file()
    root = frames[0].path.parent
    await adapter.discard(frames)
    assert not root.exists()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="format fixture encoder unavailable")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("suffix", "codec"),
    [
        ("wav", "pcm_s16le"),
        ("flac", "flac"),
        ("mp3", "libmp3lame"),
        ("ogg", "libvorbis"),
        ("opus", "libopus"),
        ("m4a", "aac"),
    ],
)
async def test_every_advertised_audio_format_is_really_encoded_and_decoded(tmp_path, suffix, codec):
    source = _encoded_audio(tmp_path / f"voice.{suffix}", codec)
    adapter = provider.AvMediaAdapter()

    info = await adapter.inspect(source)
    decoded = await adapter.decode_audio(source, CancellationController())

    assert info.modality is MediaModality.AUDIO
    assert info.has_audio is True
    assert info.duration_ms > 0
    assert decoded.size > 0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="format fixture encoder unavailable")
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("suffix", "video_codec", "audio_codec"),
    [
        ("mp4", "mpeg4", "aac"),
        ("webm", "libvpx-vp9", "libopus"),
        ("mkv", "mpeg4", "aac"),
        ("mov", "mpeg4", "aac"),
        ("avi", "mpeg4", "libmp3lame"),
        ("m4v", "mpeg4", None),
    ],
)
async def test_every_advertised_video_format_decodes_real_changing_frames_and_audio(
    tmp_path, suffix, video_codec, audio_codec
):
    source = _encoded_video(tmp_path / f"clip.{suffix}", video_codec, audio_codec)
    adapter = provider.AvMediaAdapter()

    info = await adapter.inspect(source)
    frames = await adapter.sample(source, info, CancellationController())

    assert info.modality is MediaModality.VIDEO
    assert info.has_audio is (audio_codec is not None)
    assert info.width == 96
    assert info.height == 64
    assert frames and all(frame.path.is_file() for frame in frames)
    if audio_codec is not None:
        assert (await adapter.decode_audio(source, CancellationController())).size > 0
    await adapter.discard(frames)


@pytest.mark.asyncio
@pytest.mark.parametrize("suffix", ["png", "jpg", "webp"])
async def test_complex_raster_image_formats_decode_with_mixed_script_layouts(tmp_path, suffix):
    source = _complex_raster(tmp_path / f"dense.{suffix}")
    adapter = provider.AvMediaAdapter()

    info = await adapter.inspect(source)
    frames = await adapter.sample(source, info, CancellationController())

    assert info == MediaInfo(MediaModality.IMAGE, None, 640, 420, False)
    assert frames == (VisualFrame(source, None),)


@pytest.mark.asyncio
async def test_complex_svg_is_safely_rasterized_to_one_bounded_visual(tmp_path):
    source = _complex_svg(tmp_path / "dense.svg")
    adapter = provider.AvMediaAdapter()

    info = await adapter.inspect(source)
    frames = await adapter.sample(source, info, CancellationController())

    assert info == MediaInfo(MediaModality.IMAGE, None, 1600, 1050, False)
    assert frames[0].path.suffix == ".png"
    assert frames[0].path.is_file()
    assert frames[0].structured_text == "שלום · Привет · résumé"
    root = frames[0].path.parent
    await adapter.discard(frames)
    assert not root.exists()

    source.write_text(
        "<!DOCTYPE svg [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]>"
        "<svg xmlns='http://www.w3.org/2000/svg'><text>&xxe;</text></svg>",
        encoding="utf-8",
    )
    with pytest.raises(MediaTranscriptionError, match="selected image could not be decoded"):
        await adapter.inspect(source)

    source.write_text(
        "<svg xmlns='http://www.w3.org/2000/svg' width='10' height='10'>"
        "<image href='file:///etc/passwd'/></svg>",
        encoding="utf-8",
    )
    with pytest.raises(MediaTranscriptionError, match="selected image could not be decoded"):
        await adapter.inspect(source)


@pytest.mark.asyncio
@pytest.mark.parametrize(("suffix", "factory"), [("pdf", _complex_pdf), ("tiff", _complex_tiff)])
async def test_complex_pdf_and_multipage_tiff_render_every_page_for_visual_transcription(
    tmp_path, suffix, factory
):
    source = factory(tmp_path / f"complex.{suffix}")
    adapter = provider.AvMediaAdapter()
    info = await adapter.inspect(source)
    paths = []

    class Vision:
        async def transcribe(self, frame, cancellation):
            cancellation.raise_if_cancelled()
            paths.append(frame.path)
            with Image.open(frame.path) as image:
                assert image.width <= provider.MAX_RENDER_DIMENSION
                assert image.height <= provider.MAX_RENDER_DIMENSION
            return VisualTranscript(None, "שלום · Привет", "A chart and embedded image.")

        async def release(self):
            return None

    transcript = await provider.DocumentPageTranscriber(Vision()).transcribe(
        source, CancellationController()
    )

    assert info == MediaInfo(MediaModality.DOCUMENT, None, None, None, False, None, 2)
    assert [item.page_number for item in transcript] == [1, 2]
    assert all("שלום" in item.visible_text for item in transcript)
    assert paths and all(not path.exists() for path in paths)


@pytest.mark.asyncio
async def test_av_adapter_refuses_unsupported_corrupt_missing_and_cancelled_media(tmp_path):
    adapter = provider.AvMediaAdapter()
    # `.txt` used to be the example of a type this workload does not read.
    # It reads one now, so the example is a suffix that really is unsupported.
    unsupported = tmp_path / "item.bin"
    unsupported.write_bytes(b"x")
    with pytest.raises(MediaTranscriptionError, match="media-unsupported"):
        await adapter.inspect(unsupported)
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not-png")
    with pytest.raises(MediaTranscriptionError, match="media-invalid"):
        await adapter.inspect(broken)
    video = _video(tmp_path / "clip.mp4")
    info = await adapter.inspect(video)
    cancellation = CancellationController()
    cancellation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await adapter.sample(video, info, cancellation)
    assert not adapter._owned_roots
    with pytest.raises(MediaTranscriptionError, match="frames-invalid"):
        await adapter.sample(
            video,
            MediaInfo(MediaModality.AUDIO, 100, None, None, True),
            CancellationController(),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(("suffix", "factory"), [(".pptx", _pptx), (".odp", _odp)])
async def test_presentation_archives_preserve_slide_text_and_transcribe_images(
    tmp_path, suffix, factory
):
    source = factory(tmp_path / f"slides{suffix}", image=_png_bytes())
    adapter = provider.AvMediaAdapter()
    info = await adapter.inspect(source)
    assert info == MediaInfo(MediaModality.PRESENTATION, None, None, None, False, 1)

    class Vision:
        def __init__(self):
            self.paths = []

        async def transcribe(self, frame, cancellation):
            cancellation.raise_if_cancelled()
            self.paths.append(frame.path)
            assert frame.path.is_file()
            return VisualTranscript(None, "KOI8-R: Привет", "Blue diagram.")

        async def release(self):
            return None

    vision = Vision()
    transcript = await provider.PresentationArchiveTranscriber(vision).transcribe(
        source, CancellationController()
    )

    assert transcript == (
        VisualTranscript(
            None,
            "שלום\nПривет\nKOI8-R: Привет",
            "Blue diagram.",
            1,
        ),
    )
    assert vision.paths
    assert all(not path.exists() for path in vision.paths)


@pytest.mark.asyncio
async def test_text_only_and_blank_presentation_slides_have_deterministic_descriptions(
    tmp_path,
):
    source = _pptx(tmp_path / "text.pptx")

    class UnusedVision:
        async def transcribe(self, _frame, _cancellation):
            raise AssertionError("text-only slide must not invoke vision")

        async def release(self):
            return None

    result = await provider.PresentationArchiveTranscriber(UnusedVision()).transcribe(
        source, CancellationController()
    )
    assert result[0].description == "Presentation slide containing text."
    assert provider._slide_description("", ()) == "Blank presentation slide."


@pytest.mark.asyncio
async def test_presentation_parser_refuses_unsafe_invalid_and_unbounded_archives(tmp_path):
    adapter = provider.AvMediaAdapter()
    broken = tmp_path / "broken.pptx"
    broken.write_bytes(b"not a zip")
    with pytest.raises(MediaTranscriptionError, match="presentation-invalid"):
        await adapter.inspect(broken)

    unsafe = tmp_path / "unsafe.pptx"
    with zipfile.ZipFile(unsafe, "w") as archive:
        archive.writestr("../escape.xml", "x")
    with pytest.raises(MediaTranscriptionError, match="entry is unsafe"):
        await adapter.inspect(unsafe)

    empty = tmp_path / "empty.pptx"
    with zipfile.ZipFile(empty, "w") as archive:
        archive.writestr("ppt/presentation.xml", "<p:presentation xmlns:p='x'/>")
    with pytest.raises(MediaTranscriptionError, match="entry is unavailable"):
        await adapter.inspect(empty)

    entity = tmp_path / "entity.odp"
    with zipfile.ZipFile(entity, "w") as archive:
        archive.writestr(
            "content.xml",
            "<!DOCTYPE x [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]><x>&xxe;</x>",
        )
    with pytest.raises(MediaTranscriptionError, match="XML is invalid"):
        await adapter.inspect(entity)


@pytest.mark.asyncio
async def test_presentation_image_failure_and_cancellation_always_remove_temp_files(
    monkeypatch, tmp_path
):
    source = _odp(tmp_path / "slides.odp", image=b"not an image")
    fixed_root = tmp_path / "presentation-work"
    monkeypatch.setattr(presentations.tempfile, "mkdtemp", lambda **_kwargs: str(fixed_root))
    fixed_root.mkdir()

    class Vision:
        async def transcribe(self, _frame, _cancellation):
            raise AssertionError("invalid image must be rejected first")

        async def release(self):
            return None

    with pytest.raises(MediaTranscriptionError, match="presentation image is invalid"):
        await provider.PresentationArchiveTranscriber(Vision()).transcribe(
            source, CancellationController()
        )
    assert not fixed_root.exists()

    source = _odp(tmp_path / "cancel.odp", image=_png_bytes())
    fixed_root.mkdir()
    cancellation = CancellationController()
    cancellation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await provider.PresentationArchiveTranscriber(Vision()).transcribe(source, cancellation)
    assert not fixed_root.exists()


def test_duration_sampling_frame_and_audio_helpers_enforce_bounds():
    container = SimpleNamespace(duration=2_500_000)
    assert provider._duration_ms(container, None) == 2500
    stream = SimpleNamespace(duration=4, time_base=0.5)
    assert provider._duration_ms(SimpleNamespace(duration=None), stream) == 2000
    packets = [
        SimpleNamespace(pts=0, dts=0, duration=1),
        SimpleNamespace(pts=None, dts=2, duration=1),
    ]
    demuxed = SimpleNamespace(duration=None, demux=lambda _stream: packets)
    assert provider._duration_ms(demuxed, SimpleNamespace(duration=None, time_base=0.5)) == 1500
    with pytest.raises(MediaTranscriptionError, match="duration is unavailable"):
        provider._demux_duration_ms(SimpleNamespace(), stream)
    without_timestamps = SimpleNamespace(
        demux=lambda _stream: [SimpleNamespace(pts=None, dts=None, duration=0)]
    )
    with pytest.raises(MediaTranscriptionError, match="duration is unavailable"):
        provider._demux_duration_ms(without_timestamps, stream)
    # No ceiling: an eleven-minute recording is measured like any other.
    long_recording = SimpleNamespace(
        demux=lambda _stream: [SimpleNamespace(pts=0, dts=0, duration=660_000 * 2)]
    )
    assert provider._demux_duration_ms(long_recording, stream) == 660_000_000
    for duration in (None, 0):
        with pytest.raises(MediaTranscriptionError):
            provider._duration_ms(SimpleNamespace(duration=duration), None)

    assert provider._sample_timestamps(1) == (0,)
    assert provider._sample_timestamps(30_000) == (0, 29_999)
    assert len(provider._sample_timestamps(300_000)) == 20
    # The cadence carries on past five minutes rather than refusing.
    assert len(provider._sample_timestamps(600_000)) == 40
    with pytest.raises(MediaTranscriptionError, match="video duration is invalid"):
        provider._sample_timestamps(0)

    frames = [SimpleNamespace(pts=1), SimpleNamespace(pts=3)]
    stream = SimpleNamespace(time_base=0.5)
    assert provider._frame_at_or_after(iter(frames), stream, 1000) is frames[1]
    assert provider._frame_at_or_after(iter([SimpleNamespace(pts=None)]), stream, 1).pts is None
    with pytest.raises(MediaTranscriptionError, match="no frame"):
        provider._frame_at_or_after(iter(()), stream, 1)

    class AudioFrame:
        def __init__(self, size):
            self.size = size

        def to_ndarray(self):
            return np.ones((1, self.size), dtype=np.float64)

    (chunk,) = provider._resampled_chunks(AudioFrame(2), np)
    assert chunk.tolist() == [1.0, 1.0]
    assert chunk.dtype == np.float32
    assert provider._resampled_chunks(None, np) == ()
    assert len(provider._resampled_chunks([AudioFrame(2), AudioFrame(3)], np)) == 2
    # No ceiling on a recording: more than one window's worth is split, and
    # every sample comes out with the offset it belongs to.
    buffer = provider._WindowBuffer(provider.AUDIO_WINDOW_SAMPLES, np)
    windows = list(buffer.add(np.ones(provider.AUDIO_WINDOW_SAMPLES * 2 + 5, dtype=np.float32)))
    windows.extend(buffer.flush())
    assert [start for start, _ in windows] == [
        0,
        provider.AUDIO_WINDOW_SAMPLES,
        provider.AUDIO_WINDOW_SAMPLES * 2,
    ]
    assert [int(samples.size) for _, samples in windows] == [
        provider.AUDIO_WINDOW_SAMPLES,
        provider.AUDIO_WINDOW_SAMPLES,
        5,
    ]


@pytest.mark.asyncio
async def test_adapter_failure_branches_remain_typed_and_clean(monkeypatch, tmp_path):
    adapter = provider.AvMediaAdapter()
    missing = tmp_path / "missing.mp4"
    with pytest.raises(MediaTranscriptionError, match="could not be decoded"):
        await adapter.inspect(missing)
    with pytest.raises(MediaTranscriptionError, match="audio stream could not be decoded"):
        await adapter.decode_audio(tmp_path / "missing.wav", CancellationController())

    video_only = _video(tmp_path / "video-only.mp4")
    with pytest.raises(MediaTranscriptionError, match="media has no audio stream"):
        await adapter.decode_audio(video_only, CancellationController())

    invalid_svg = tmp_path / "broken.svg"
    invalid_svg.write_text("<svg>", encoding="utf-8")
    with pytest.raises(MediaTranscriptionError):
        adapter._sample_svg_sync(invalid_svg)
    assert not adapter._owned_roots

    media = MediaInfo(MediaModality.VIDEO, 1000, 10, 10, False)
    audio_only = _wav(tmp_path / "voice.wav")
    with pytest.raises(MediaTranscriptionError, match="video stream is missing"):
        adapter._sample_video_sync(audio_only, media, CancellationController())
    with pytest.raises(MediaTranscriptionError, match="could not be decoded"):
        adapter._sample_video_sync(missing, media, CancellationController())
    assert not adapter._owned_roots

    class EmptyContainer:
        duration = 1_000_000
        streams = SimpleNamespace(audio=(), video=())

        def __enter__(self):
            return self

        def __exit__(self, *_arguments):
            return False

    import av

    monkeypatch.setattr(av, "open", lambda *_arguments, **_kwargs: EmptyContainer())
    with pytest.raises(MediaTranscriptionError, match="no supported stream"):
        adapter._inspect_sync(tmp_path / "empty.mp4")


def test_svg_document_and_slide_failure_branches_are_bounded(monkeypatch, tmp_path):
    assert provider._svg_output_size(
        b"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 320 160'/>"
    ) == (1600, 800)
    with pytest.raises(ValueError, match="viewBox is invalid"):
        provider._svg_output_size(b"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 10'/>")
    with pytest.raises(ValueError, match="dimensions are invalid"):
        provider._svg_output_size(
            b"<svg xmlns='http://www.w3.org/2000/svg' width='0' height='10'/>"
        )
    # No ceiling on what a page says: a slide of 16,385 characters is a slide
    # somebody wrote, and it used to lose the whole job.
    assert provider._joined_slide_text(("x" * 16_385,)) == "x" * 16_385

    unsupported = tmp_path / "document.bin"
    unsupported.write_bytes(b"x")
    with pytest.raises(MediaTranscriptionError, match="could not be decoded"):
        provider._open_document(unsupported)

    # A sixty-five page document is read, not refused for its length — and it
    # is opened once for all of it rather than once per page.
    many = tmp_path / "pages.tiff"
    pages = [Image.new("1", (1, 1)) for _index in range(65)]
    pages[0].save(many, save_all=True, append_images=pages[1:])
    opened = provider._open_document(many)
    try:
        assert opened.page_count == 65
    finally:
        opened.close()

    root = tmp_path / "rendered"
    root.mkdir()
    refusal = MediaTranscriptionError("document-invalid", "refused")

    class Refusing(documents._OpenDocument):
        def __init__(self, error):
            super().__init__(1)
            self._error = error

        def page(self, _root, _page_number):
            raise self._error

    with pytest.raises(MediaTranscriptionError, match="refused"):
        provider._read_page(Refusing(refusal), root, 1)
    with pytest.raises(MediaTranscriptionError, match="could not be rendered"):
        provider._read_page(Refusing(RuntimeError("decoder crashed")), root, 1)


def _install_fake_whisper(monkeypatch, *, good_device=True):
    native = types.ModuleType("_pywhispercpp")
    native.callback = None
    native.freed = []
    native.show_device = True

    def log_set(callback):
        native.callback = callback

    native.whisper_log_set = log_set
    native.whisper_free = native.freed.append
    monkeypatch.setitem(sys.modules, "_pywhispercpp", native)

    model_module = types.ModuleType("pywhispercpp.model")

    class Segment:
        t0 = 0
        t1 = 10
        text = " שלום "

    class Model:
        def __init__(self, *_args, **kwargs):
            self.kwargs = kwargs
            self._ctx = object()
            name = "AMD Radeon RX 6600 XT"
            if native.show_device and good_device:
                native.callback(1, f"ggml_vulkan: 0 = {name} (driver)\n")
            native.callback(1, "whisper_backend_init_gpu: using Vulkan0 backend\n")

        def auto_detect_language(self, _audio):
            return (("he", np.float32(0.9)), {"he": np.float32(0.9)})

        def transcribe(self, _audio, **_kwargs):
            return [Segment()]

    model_module.Model = Model
    monkeypatch.setitem(sys.modules, "pywhispercpp.model", model_module)
    return native


def test_whisper_vulkan_proves_device_transcribes_and_releases(monkeypatch, tmp_path):
    native = _install_fake_whisper(monkeypatch)
    lease_path = tmp_path / "lease"
    lease_path.touch()
    decoder = SimpleNamespace(
        decode_audio_windows=lambda _source, _cancellation: iter(
            ((0, np.ones(160, dtype=np.float32)),)
        )
    )
    transcriber = provider.WhisperVulkanTranscriber(
        tmp_path / "model.bin",
        decoder,
        provider.VulkanLease(lease_path),
    )
    transcript = transcriber._transcribe_sync(tmp_path / "voice.wav", CancellationController())
    assert transcript.language == "he"
    assert transcript.segments[0].text == " שלום "
    assert transcript.segments[0].end_ms == 10
    assert len(native.freed) == 1
    native.show_device = False
    transcriber._load_and_release()
    assert len(native.freed) == 2


def test_whisper_vulkan_refuses_unproved_device_and_closes_context(monkeypatch, tmp_path):
    native = _install_fake_whisper(monkeypatch, good_device=False)
    lease_path = tmp_path / "lease"
    lease_path.touch()
    transcriber = provider.WhisperVulkanTranscriber(
        tmp_path / "model.bin",
        SimpleNamespace(),
        provider.VulkanLease(lease_path),
    )
    with pytest.raises(provider.MediaGpuError, match="did not prove"):
        transcriber._load()
    assert len(native.freed) == 1
    provider._close_whisper(None)
    provider._close_whisper(SimpleNamespace(_ctx=None))


@pytest.mark.asyncio
async def test_whisper_async_preflight_and_empty_or_spoken_audio(monkeypatch, tmp_path):
    _install_fake_whisper(monkeypatch)
    lease_path = tmp_path / "lease"
    lease_path.touch()

    class Decoder:
        audio = np.zeros(0, dtype=np.float32)

        async def decode_audio(self, _source, cancellation):
            cancellation.raise_if_cancelled()
            return self.audio

        def decode_audio_windows(self, _source, cancellation):
            cancellation.raise_if_cancelled()
            if self.audio.size:
                yield 0, self.audio

    decoder = Decoder()
    transcriber = provider.WhisperVulkanTranscriber(
        tmp_path / "model.bin",
        decoder,
        provider.VulkanLease(lease_path),
    )
    await transcriber.preflight()
    assert await transcriber.transcribe(tmp_path / "empty.wav", CancellationController()) == (
        provider.SpeechTranscript(None, ())
    )
    decoder.audio = np.ones(160, dtype=np.float32)
    spoken = await transcriber.transcribe(tmp_path / "spoken.wav", CancellationController())
    assert spoken.language == "he"


def _install_fake_llama(monkeypatch, *, good_device=True, content=None):  # noqa: C901
    package = types.ModuleType("llama_cpp")
    native = SimpleNamespace(callback=None, completion_kwargs=None, abort_callback=None)
    # `good_device` now means the load ended up wholly on a Vulkan device.
    # Which card it is is no longer asserted; a partial offload is, because
    # the remainder of it runs on the CPU.
    device_name = "AMD Radeon RX 6600 XT"
    # `good_device` now means the load reached the GPU at all: a declared
    # partial offload is accepted since OMNI-0586, so the refusal case is a
    # load that put no layer on any Vulkan device.
    offloaded = "33/33" if good_device else "0/33"
    response_content = _fake_visual_content(content)

    def callback_type(callback):
        return callback

    def log_set(callback, _data):
        native.callback = callback

    native.llama_log_callback = callback_type
    native.llama_log_set = log_set
    native.ggml_abort_callback = callback_type

    def set_abort_callback(_context, callback, _data):
        native.abort_callback = callback

    native.llama_set_abort_callback = set_abort_callback

    class Stack:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class Handler:
        def __init__(self, *_args, **_kwargs):
            self._exit_stack = Stack()

    class Llama:
        def __init__(self, **_kwargs):
            self.closed = False
            self.ctx = object()
            native.callback(
                1,
                f"using device Vulkan0 ({device_name}) (0000:00:00.0)\n".encode(),
                None,
            )
            native.callback(1, f"offloaded {offloaded} layers to GPU\n".encode(), None)

        def create_chat_completion(self, **kwargs):
            native.completion_kwargs = kwargs
            if native.abort_callback is not None and native.abort_callback(None):
                raise RuntimeError("native decode aborted")
            return iter([{"choices": [{"delta": {"content": response_content}}]}])

        def close(self):
            self.closed = True

    package.Llama = Llama
    package.llama_cpp = native
    chat = types.ModuleType("llama_cpp.llama_chat_format")
    chat.Qwen25VLChatHandler = Handler
    monkeypatch.setitem(sys.modules, "llama_cpp", package)
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_chat_format", chat)


def _fake_visual_content(content):
    return content or '{"visibleText":"שלום","description":"Blue sign."}'


def test_qwen_vulkan_proves_full_offload_transcribes_and_releases(monkeypatch, tmp_path):
    _install_fake_llama(monkeypatch)
    lease_path = tmp_path / "lease"
    lease_path.touch()
    image = tmp_path / "frame.png"
    Image.new("RGB", (900, 600), "blue").save(image)
    transcriber = provider.QwenVulkanVisualTranscriber(
        tmp_path / "model.gguf",
        tmp_path / "mmproj.gguf",
        provider.VulkanLease(lease_path),
    )
    result = transcriber._transcribe_sync(VisualFrame(image, 10), CancellationController())
    assert result.visible_text == "שלום"
    assert result.description == "Blue sign."
    assert callable(sys.modules["llama_cpp"].llama_cpp.abort_callback)
    schema = sys.modules["llama_cpp"].llama_cpp.completion_kwargs["response_format"]["schema"]
    assert "maxLength" not in json.dumps(schema)
    payload = sys.modules["llama_cpp"].llama_cpp.completion_kwargs["messages"][1]["content"][0]
    encoded = payload["image_url"]["url"].split(",", 1)[1]
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as normalized:
        assert normalized.size == (768, 512)
    assert transcriber._ensure_loaded() is transcriber._llama
    transcriber._release_sync()
    assert transcriber._llama is None
    transcriber._release_sync()

    structured = transcriber._transcribe_sync(
        VisualFrame(image, None, "שלום עולם"), CancellationController()
    )
    assert structured.visible_text == "שלום עולם"
    transcriber._release_sync()

    cancellation = CancellationController()
    cancellation.cancel()
    with pytest.raises(asyncio.CancelledError):
        transcriber._transcribe_sync(VisualFrame(image, None), cancellation)
    transcriber._release_sync()


@pytest.mark.parametrize(
    ("good_device", "content"),
    [(False, None), (True, "not-json"), (True, '{"visibleText":"x"}')],
)
def test_qwen_vulkan_refuses_unproved_or_invalid_output(
    monkeypatch, tmp_path, good_device, content
):
    _install_fake_llama(monkeypatch, good_device=good_device, content=content)
    lease_path = tmp_path / "lease"
    lease_path.touch()
    image = tmp_path / "frame.png"
    Image.new("RGB", (8, 8), "blue").save(image)
    transcriber = provider.QwenVulkanVisualTranscriber(
        tmp_path / "model.gguf",
        tmp_path / "mmproj.gguf",
        provider.VulkanLease(lease_path),
    )
    expected = provider.MediaGpuError if not good_device else MediaTranscriptionError
    with pytest.raises(expected):
        transcriber._transcribe_sync(VisualFrame(image, None), CancellationController())
    transcriber._release_sync()


@pytest.mark.asyncio
async def test_qwen_async_lifecycle_and_missing_runtime(monkeypatch, tmp_path):
    _install_fake_llama(monkeypatch)
    lease_path = tmp_path / "lease"
    lease_path.touch()
    image = tmp_path / "frame.png"
    Image.new("RGB", (8, 8), "blue").save(image)
    transcriber = provider.QwenVulkanVisualTranscriber(
        tmp_path / "model.gguf",
        tmp_path / "mmproj.gguf",
        provider.VulkanLease(lease_path),
    )
    await transcriber.preflight()
    result = await transcriber.transcribe(VisualFrame(image, None), CancellationController())
    assert result.visible_text == "שלום"
    await transcriber.release()

    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_chat_format", None)
    with pytest.raises(provider.MediaGpuError, match="runtime is unavailable"):
        provider.QwenVulkanVisualTranscriber._runtime()


def test_create_binds_exact_bootstrap_artifacts_and_preflight(monkeypatch, tmp_path):
    vision_path = tmp_path / "model.gguf"
    projector_path = tmp_path / provider.VISION_PROJECTOR
    speech_path = tmp_path / "model.bin"
    lease_path = tmp_path / "lease"
    for path in (vision_path, projector_path, speech_path, lease_path):
        path.touch()
    bootstrap = PluginBootstrap(
        provider.PLUGIN_ID,
        (
            BootstrapArtifact(
                provider.VISION_ARTIFACT_ID,
                "1.0.0",
                "gguf",
                "vision",
                vision_path,
                ((provider.VISION_PROJECTOR, "projector"),),
            ),
            BootstrapArtifact(
                provider.SPEECH_ARTIFACT_ID,
                "1.0.0",
                "ggml-whisper",
                "speech",
                speech_path,
            ),
        ),
        None,
        lease_path,
    )
    requested_plugin_ids = []

    def current_bootstrap(plugin_id):
        requested_plugin_ids.append(plugin_id)
        return bootstrap

    monkeypatch.setattr(provider, "current_plugin_bootstrap", current_bootstrap)
    preflight_order = []

    def vision_runtime():
        preflight_order.append("vision-runtime")

    async def vision_preflight(_self):
        preflight_order.append("vision")

    async def speech_preflight(_self):
        preflight_order.append("speech")

    monkeypatch.setattr(provider.QwenVulkanVisualTranscriber, "preflight", vision_preflight)
    monkeypatch.setattr(
        provider.QwenVulkanVisualTranscriber, "_runtime", staticmethod(vision_runtime)
    )
    monkeypatch.setattr(provider.WhisperVulkanTranscriber, "preflight", speech_preflight)
    plugin = provider.create()
    assert requested_plugin_ids == [provider.PLUGIN_ID]
    assert plugin.plugin_id == provider.PLUGIN_ID
    assert plugin._identity.provider_id == provider.PROVIDER_ID
    asyncio.run(plugin._preflight())
    assert preflight_order == ["vision-runtime", "speech", "vision"]

    missing_lease = PluginBootstrap(provider.PLUGIN_ID, bootstrap.artifacts, None, None)
    monkeypatch.setattr(provider, "current_plugin_bootstrap", lambda _plugin_id: missing_lease)
    with pytest.raises(provider.MediaGpuError) as excinfo:
        provider.create()
    assert str(excinfo.value) == "GPU accelerator grant is unavailable"

    projector_path.unlink()
    monkeypatch.setattr(provider, "current_plugin_bootstrap", lambda _plugin_id: bootstrap)
    with pytest.raises(provider.MediaGpuError) as excinfo:
        provider.create()
    assert str(excinfo.value) == "the visual projector is unavailable"


def test_public_provider_exports_are_bounded():
    assert provider.__all__ == [
        "AvMediaAdapter",
        "DocumentPageTranscriber",
        "PROVIDER_ID",
        "PresentationArchiveTranscriber",
        "QwenVulkanVisualTranscriber",
        "MediaGpuError",
        "VulkanLease",
        "WhisperVulkanTranscriber",
        "create",
    ]
    assert provider.AvMediaAdapter.__module__.endswith(".formats")
    assert provider.WhisperVulkanTranscriber.__module__.endswith(".models")
    assert provider.QwenVulkanVisualTranscriber.__module__.endswith(".models")
    assert provider.DocumentPageTranscriber.__module__.endswith(".documents")
    assert provider.PresentationArchiveTranscriber.__module__.endswith(".presentations")


def test_a_chosen_vision_model_is_the_one_that_loads(monkeypatch, tmp_path):
    """OMNI-0587/0588: the 9B is the default now; choosing the legacy 7B must
    actually mount it, and no choice at all serves the default."""
    vision_path = tmp_path / "vl-model.gguf"
    nine_b_path = tmp_path / "9b-model.gguf"
    speech_path = tmp_path / "speech.bin"
    lease_path = tmp_path / "lease"
    for path in (vision_path, nine_b_path, speech_path, lease_path):
        path.touch()
    (tmp_path / provider.VISION_PROJECTOR).touch()
    artifacts = (
        BootstrapArtifact(
            provider.LEGACY_VISION_ARTIFACT_ID, "1.0.0", "gguf", "vl", vision_path,
            ((provider.VISION_PROJECTOR, "vl-proj"),),
        ),
        BootstrapArtifact(
            provider.VISION_ARTIFACT_ID, "1.0.0", "gguf", "9b", nine_b_path,
            ((provider.VISION_PROJECTOR, "9b-proj"),),
        ),
        BootstrapArtifact(
            provider.SPEECH_ARTIFACT_ID, "1.0.0", "ggml-whisper", "speech", speech_path,
        ),
    )
    chosen = PluginBootstrap(
        provider.PLUGIN_ID, artifacts, None, lease_path,
        model_choice=provider.LEGACY_VISION_ARTIFACT_ID,
    )
    monkeypatch.setattr(provider, "current_plugin_bootstrap", lambda _plugin_id: chosen)

    plugin = provider.create()

    assert plugin._vision._model_path == vision_path

    unchosen = PluginBootstrap(provider.PLUGIN_ID, artifacts, None, lease_path)
    monkeypatch.setattr(provider, "current_plugin_bootstrap", lambda _plugin_id: unchosen)
    assert provider.create()._vision._model_path == nine_b_path
