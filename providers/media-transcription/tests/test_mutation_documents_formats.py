"""Mutation-focused boundaries for document and media format adapters."""

from __future__ import annotations

import asyncio
import io
import sys
from types import SimpleNamespace

import numpy as np
import omnitensor_media_transcription.documents as documents
import omnitensor_media_transcription.formats as formats
import pytest
from hypothesis import given
from hypothesis import strategies as st
from PIL import Image

from omnitensor.plugins.media_transcription import (
    MediaInfo,
    MediaModality,
    MediaTranscriptionError,
    VisualFrame,
    VisualTranscript,
)
from omnitensor.sdk import CancellationController


class _Context:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, *_arguments):
        return False


def _assert_media_error(error, code: str, detail: str) -> None:
    assert error.value.code == code
    assert error.value.detail == detail
    assert str(error.value) == f"{code}: {detail}"


@pytest.mark.asyncio
async def test_document_transcriber_preserves_every_field_and_cleans_root(monkeypatch, tmp_path):
    root = tmp_path / "omnitensor-document-pages-fixed"
    root.mkdir()
    calls = []

    monkeypatch.setattr(documents, "_document_page_count", lambda _source: 2)
    monkeypatch.setattr(
        documents.tempfile,
        "mkdtemp",
        lambda *, prefix: calls.append(("prefix", prefix)) or str(root),
    )

    def render(_source, render_root, page_number):
        assert render_root == root
        path = render_root / f"page-{page_number:02d}.png"
        path.write_bytes(b"frame")
        return f"printed {page_number}", path

    monkeypatch.setattr(documents, "_render_document_page", render)

    class Vision:
        async def transcribe(self, frame, cancellation):
            cancellation.raise_if_cancelled()
            calls.append((frame.path.name, frame.timestamp_ms))
            return VisualTranscript(
                None,
                f"visual {frame.path.stem}",
                f"description {frame.path.stem}",
            )

    result = await documents.DocumentPageTranscriber(Vision()).transcribe(
        tmp_path / "source.pdf", CancellationController()
    )

    assert result == (
        VisualTranscript(
            None,
            "printed 1\nvisual page-01",
            "description page-01",
            None,
            1,
        ),
        VisualTranscript(
            None,
            "printed 2\nvisual page-02",
            "description page-02",
            None,
            2,
        ),
    )
    assert calls == [
        ("prefix", "omnitensor-document-pages-"),
        ("page-01.png", None),
        ("page-02.png", None),
    ]
    assert not root.exists()


@pytest.mark.parametrize("count", [1, 64, 2_500])
def test_document_page_count_accepts_any_positive_count(monkeypatch, tmp_path, count):
    source = tmp_path / "document.PDF"
    source.write_bytes(b"stub")
    document = SimpleNamespace(page_count=count, needs_pass=False)
    monkeypatch.setitem(
        sys.modules,
        "pymupdf",
        SimpleNamespace(open=lambda selected: _Context(document) if selected == source else None),
    )

    assert documents._document_page_count(source) == count


@pytest.mark.parametrize("count", [0, -1])
def test_document_page_count_rejects_a_document_with_no_pages(monkeypatch, tmp_path, count):
    source = tmp_path / "document.pdf"
    source.write_bytes(b"stub")
    document = SimpleNamespace(page_count=count, needs_pass=False)
    monkeypatch.setitem(
        sys.modules,
        "pymupdf",
        SimpleNamespace(open=lambda _path: _Context(document)),
    )

    with pytest.raises(MediaTranscriptionError) as error:
        documents._document_page_count(source)

    _assert_media_error(error, "document-invalid", "document page count is invalid")


def test_document_page_count_accepts_tif_and_rejects_encrypted_pdf(monkeypatch, tmp_path):
    tif = tmp_path / "scan.TIF"
    Image.new("RGB", (2, 3), "blue").save(tif, format="TIFF")
    assert documents._document_page_count(tif) == 1

    encrypted = tmp_path / "private.pdf"
    encrypted.write_bytes(b"stub")
    document = SimpleNamespace(page_count=1, needs_pass=True)
    monkeypatch.setitem(
        sys.modules,
        "pymupdf",
        SimpleNamespace(open=lambda _path: _Context(document)),
    )
    with pytest.raises(MediaTranscriptionError) as error:
        documents._document_page_count(encrypted)
    _assert_media_error(error, "document-invalid", "selected document could not be decoded")
    assert isinstance(error.value.__cause__, ValueError)
    assert str(error.value.__cause__) == "encrypted PDF"


def test_render_pdf_page_uses_exact_page_scale_and_opaque_png(monkeypatch, tmp_path):
    source = tmp_path / "source.pdf"
    target = tmp_path / "page.png"
    calls = []

    class Pixmap:
        width = 1200
        height = 800

        def save(self, path):
            calls.append(("save", path))
            path.write_bytes(b"png")

    class Page:
        rect = SimpleNamespace(width=800, height=400)

        def get_text(self, mode):
            calls.append(("text", mode))
            return " first \n\n second "

        def get_pixmap(self, *, matrix, alpha):
            calls.append(("pixmap", matrix, alpha))
            return Pixmap()

    class Document:
        def load_page(self, index):
            calls.append(("page", index))
            return Page()

    class Matrix:
        def __init__(self, x_scale, y_scale):
            self.scales = (x_scale, y_scale)

        def __eq__(self, other):
            return isinstance(other, Matrix) and self.scales == other.scales

    monkeypatch.setitem(
        sys.modules,
        "pymupdf",
        SimpleNamespace(
            open=lambda selected: _Context(Document()) if selected == source else None,
            Matrix=Matrix,
        ),
    )

    assert documents._render_pdf_page(source, target, 2) == "first\nsecond"
    assert calls == [
        ("page", 1),
        ("text", "text"),
        ("pixmap", Matrix(2.0, 2.0), False),
        ("save", target),
    ]
    assert target.read_bytes() == b"png"


@pytest.mark.parametrize("pixels", [50_000_000, 50_000_001])
def test_render_pdf_page_enforces_exact_pixel_limit(monkeypatch, tmp_path, pixels):
    saved = []
    pixmap = SimpleNamespace(width=pixels, height=1, save=saved.append)
    page = SimpleNamespace(
        rect=SimpleNamespace(width=1600, height=1),
        get_text=lambda mode: "" if mode == "text" else None,
        get_pixmap=lambda *, matrix, alpha: pixmap,
    )
    document = SimpleNamespace(load_page=lambda index: page if index == 0 else None)
    monkeypatch.setitem(
        sys.modules,
        "pymupdf",
        SimpleNamespace(open=lambda _path: _Context(document), Matrix=lambda x, y: (x, y)),
    )
    target = tmp_path / "page.png"

    if pixels == 50_000_000:
        assert documents._render_pdf_page(tmp_path / "source.pdf", target, 1) == ""
        assert saved == [target]
    else:
        with pytest.raises(MediaTranscriptionError) as error:
            documents._render_pdf_page(tmp_path / "source.pdf", target, 1)
        _assert_media_error(error, "document-invalid", "document page dimensions are invalid")
        assert saved == []


def test_render_tiff_page_uses_exact_page_rgb_thumbnail_and_png(monkeypatch, tmp_path):
    calls = []

    class Rendered:
        def thumbnail(self, size):
            calls.append(("thumbnail", size))

        def save(self, path, *, format, optimize):
            calls.append(("save", path, format, optimize))

    class SourceImage:
        width = 3200
        height = 800

        def seek(self, index):
            calls.append(("seek", index))

        def convert(self, mode):
            calls.append(("convert", mode))
            return Rendered()

    import PIL.Image

    monkeypatch.setattr(PIL.Image, "open", lambda _source: _Context(SourceImage()))
    target = tmp_path / "page.png"

    assert documents._render_tiff_page(tmp_path / "source.tiff", target, 3) == ""
    assert calls == [
        ("seek", 2),
        ("convert", "RGB"),
        ("thumbnail", (1600, 1600)),
        ("save", target, "PNG", False),
    ]


def test_render_tiff_page_rejects_over_pixel_limit_with_stable_error(monkeypatch, tmp_path):
    image = SimpleNamespace(width=50_000_001, height=1, seek=lambda _index: None)
    import PIL.Image

    monkeypatch.setattr(PIL.Image, "open", lambda _source: _Context(image))
    with pytest.raises(MediaTranscriptionError) as error:
        documents._render_tiff_page(tmp_path / "source.tiff", tmp_path / "page.png", 1)
    _assert_media_error(error, "document-invalid", "document page dimensions are invalid")


def test_svg_png_enforces_byte_bounds_and_exact_rasterizer_contract(monkeypatch, tmp_path):
    source = tmp_path / "drawing.svg"
    content = b"<svg width='4' height='2'/>"
    source.write_bytes(content)
    calls = []

    class Surface:
        @staticmethod
        def convert(**kwargs):
            calls.append(kwargs)
            return b"rendered"

    monkeypatch.setattr(
        formats,
        "_svg_output_size",
        lambda _selected: (40, 20),
    )
    import cairosvg.surface

    monkeypatch.setattr(cairosvg.surface, "PNGSurface", Surface)
    stream = formats._svg_png(source)

    assert isinstance(stream, io.BytesIO)
    assert stream.read() == b"rendered"
    assert calls == [
        {
            "bytestring": content,
            "output_width": 40,
            "output_height": 20,
            "url_fetcher": formats._svg_url_fetcher,
        }
    ]

    for boundary in (b"x", b"x" * formats.MAX_SVG_BYTES):
        source.write_bytes(boundary)
        calls.clear()
        stream = formats._svg_png(source)
        assert stream.read() == b"rendered"
        assert calls[0]["bytestring"] == boundary

    for invalid in (b"", b"x" * (formats.MAX_SVG_BYTES + 1)):
        source.write_bytes(invalid)
        with pytest.raises(MediaTranscriptionError) as error:
            formats._svg_png(source)
        _assert_media_error(error, "media-invalid", "selected SVG could not be safely rasterized")


@given(
    width=st.integers(min_value=1, max_value=10_000),
    height=st.integers(min_value=1, max_value=10_000),
)
def test_svg_output_size_preserves_aspect_ratio_at_bounded_dimension(width, height):
    content = f"<svg width='{width}' height='{height}'/>".encode()
    scale = min(formats.MAX_RENDER_DIMENSION / width, formats.MAX_RENDER_DIMENSION / height)

    assert formats._svg_output_size(content) == (
        max(1, round(width * scale)),
        max(1, round(height * scale)),
    )


@pytest.mark.parametrize(
    ("content", "detail"),
    [
        (b"<svg viewBox='0 0 1 2 3'/>", "SVG viewBox is invalid"),
        (b"<svg width='nan' height='2'/>", "SVG dimensions are invalid"),
        (b"<svg width='1' height='0'/>", "SVG dimensions are invalid"),
    ],
)
def test_svg_output_size_reports_exact_invalid_boundary(content, detail):
    with pytest.raises(ValueError) as error:
        formats._svg_output_size(content)
    assert str(error.value) == detail


def test_svg_output_size_honors_dpi_and_font_size():
    assert formats._svg_output_size(b"<svg width='1in' height='48px'/>") == (1600, 800)
    assert formats._svg_output_size(b"<svg width='2em' height='12px'/>") == (1600, 800)


def test_svg_output_size_passes_exact_secure_parser_and_size_contract(monkeypatch):
    content = b"<svg width='4' height='2'/>"
    calls = []

    class Tree:
        def __init__(self, *, bytestring, unsafe, url_fetcher):
            calls.append(("tree", bytestring, unsafe, url_fetcher))

        def get(self, key, default=None):
            calls.append(("get", key, default))
            return {"width": "4", "height": "2", "viewBox": None}[key]

    def size(surface, value, axis):
        assert vars(surface) == {
            "dpi": 96,
            "context_width": 0,
            "context_height": 0,
            "font_size": 12,
        }
        calls.append(("size", value, axis))
        return float(value)

    import cairosvg.helpers
    import cairosvg.parser

    monkeypatch.setattr(cairosvg.parser, "Tree", Tree)
    monkeypatch.setattr(cairosvg.helpers, "size", size)

    assert formats._svg_output_size(content) == (1600, 800)
    assert calls == [
        ("tree", content, False, formats._svg_url_fetcher),
        ("get", "width", ""),
        ("size", "4", "x"),
        ("get", "height", ""),
        ("size", "2", "y"),
        ("get", "viewBox", None),
    ]


def test_svg_text_reads_only_text_nodes_and_normalizes_whitespace(tmp_path):
    source = tmp_path / "drawing.svg"
    source.write_text(
        "<svg xmlns='http://www.w3.org/2000/svg'>"
        "<title>ignored</title><text> שלום <tspan>עולם</tspan></text>"
        "<g><text>Привет   мир</text></g></svg>",
        encoding="utf-8",
    )
    assert formats._svg_text(source) == "שלום עולם\nПривет   мир"


def test_svg_text_accepts_an_unnamespaced_text_element(tmp_path):
    source = tmp_path / "plain.svg"
    source.write_text("<svg><text>plain text</text></svg>", encoding="utf-8")

    assert formats._svg_text(source) == "plain text"

    source.write_text(
        "<svg><title>ignored</title><text>עברית plain</text></svg>",
        encoding="utf-8",
    )
    assert formats._svg_text(source) == "עברית plain"


def test_svg_url_fetcher_only_delegates_embedded_images(monkeypatch):
    calls = []
    import cairosvg.url

    monkeypatch.setattr(cairosvg.url, "fetch", lambda url, kind: calls.append((url, kind)) or b"ok")
    assert formats._svg_url_fetcher("data:image/png;base64,AA==", "image/png") == b"ok"
    assert calls == [("data:image/png;base64,AA==", "image/png")]

    for forbidden in (None, 7, "https://example.test/a.png", "data:text/plain,hello"):
        with pytest.raises(ValueError, match="external SVG resources are forbidden"):
            formats._svg_url_fetcher(forbidden, "image/png")
    assert len(calls) == 1


def test_visual_payload_is_exact_rgb_png_with_bounded_dimensions(tmp_path):
    source = tmp_path / "alpha.png"
    Image.new("RGBA", (1536, 768), (10, 20, 30, 40)).save(source)
    payload = formats._visual_payload(source)

    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        assert image.format == "PNG"
        assert image.mode == "RGB"
        assert image.size == (768, 384)
        assert np.allclose(image.getpixel((0, 0)), (10, 20, 30), atol=2)

    with pytest.raises(MediaTranscriptionError) as error:
        formats._visual_payload(tmp_path / "missing.png")
    _assert_media_error(error, "visual-invalid", "visual frame could not be normalized")


def test_visual_payload_passes_exact_normalization_options(monkeypatch, tmp_path):
    calls = []
    output = b"normalized-png"

    class Converted:
        def save(self, stream, *, format, optimize):
            calls.append(("save", format, optimize))
            stream.write(output)

    class SourceImage:
        def load(self):
            calls.append("load")

        def thumbnail(self, size, resample, *, reducing_gap):
            calls.append(("thumbnail", size, resample, reducing_gap))

        def convert(self, mode):
            calls.append(("convert", mode))
            return Converted()

    import PIL.Image

    monkeypatch.setattr(PIL.Image, "open", lambda source: _Context(SourceImage()))

    assert formats._visual_payload(tmp_path / "source.png") == output
    assert calls == [
        "load",
        (
            "thumbnail",
            (formats.MAX_INFERENCE_DIMENSION, formats.MAX_INFERENCE_DIMENSION),
            PIL.Image.Resampling.LANCZOS,
            2,
        ),
        ("convert", "RGB"),
        ("save", "PNG", False),
    ]


class _AudioFrame:
    def __init__(self, array):
        self.array = array

    def to_ndarray(self):
        return self.array


def test_resampled_chunks_flattens_every_result_shape_without_copying():
    original = np.array([[1.5, 2.5]], dtype=np.float32)

    (single,) = formats._resampled_chunks(_AudioFrame(original), np)
    assert single.shape == (2,)
    assert single.dtype == np.float32
    assert np.shares_memory(single, original)
    assert single.tolist() == [1.5, 2.5]

    # A list of frames, and a flush that produced nothing: the three shapes
    # `AudioResampler.resample` returns.
    assert len(formats._resampled_chunks([_AudioFrame(original)] * 3, np)) == 3
    assert formats._resampled_chunks(None, np) == ()


def test_windows_are_exact_and_carry_their_offset_however_chunks_arrive():
    """A recording is split, never refused: no length reaches a ceiling."""

    buffer = formats._WindowBuffer(4, np)
    produced = []
    for size in (3, 3, 1, 6):
        produced.extend(buffer.add(np.arange(size, dtype=np.float32)))
    produced.extend(buffer.flush())

    assert [start for start, _ in produced] == [0, 4, 8, 12]
    assert [int(samples.size) for _, samples in produced] == [4, 4, 4, 1]
    # Every decoded sample comes out exactly once and in order.
    assert np.concatenate([samples for _, samples in produced]).tolist() == [
        0.0,
        1.0,
        2.0,
        0.0,
        1.0,
        2.0,
        0.0,
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    ]
    assert list(buffer.flush()) == []


def test_a_window_must_hold_a_sample():
    with pytest.raises(ValueError, match="at least one sample"):
        formats._WindowBuffer(0, np)


@pytest.mark.parametrize("milliseconds", [1, 600_000, 3 * 60 * 60 * 1000])
def test_duration_ms_accepts_any_positive_container_duration(milliseconds):
    assert formats._duration_ms(SimpleNamespace(duration=milliseconds * 1000), None) == milliseconds


def test_duration_ms_uses_stream_and_has_stable_refusals():
    stream = SimpleNamespace(duration=3, time_base=0.25)
    assert formats._duration_ms(SimpleNamespace(duration=None), stream) == 750

    with pytest.raises(MediaTranscriptionError) as unavailable:
        formats._duration_ms(SimpleNamespace(duration=None), None)
    _assert_media_error(unavailable, "media-invalid", "media duration is unavailable")

    with pytest.raises(MediaTranscriptionError) as empty:
        formats._duration_ms(SimpleNamespace(duration=0), None)
    _assert_media_error(empty, "media-invalid", "media duration is unavailable")


def test_demux_duration_uses_requested_stream_skips_unknown_and_accepts_limit():
    stream = SimpleNamespace(time_base=0.001)
    packets = [
        SimpleNamespace(pts=None, dts=None, duration=999),
        SimpleNamespace(pts=1, dts=None, duration=None),
        SimpleNamespace(pts=None, dts=3 * 60 * 60 * 1000 - 1, duration=1),
    ]
    requested = []
    container = SimpleNamespace(demux=lambda selected: requested.append(selected) or packets)

    assert formats._demux_duration_ms(container, stream) == 3 * 60 * 60 * 1000
    assert requested == [stream]


def test_demux_duration_exact_errors_and_overlong_boundary():
    stream = SimpleNamespace(time_base=0.001)
    with pytest.raises(MediaTranscriptionError) as unavailable:
        formats._demux_duration_ms(SimpleNamespace(), stream)
    _assert_media_error(unavailable, "media-invalid", "media duration is unavailable")

    empty = SimpleNamespace(demux=lambda _stream: [SimpleNamespace(pts=0, dts=None, duration=0)])
    with pytest.raises(MediaTranscriptionError) as zero:
        formats._demux_duration_ms(empty, stream)
    _assert_media_error(zero, "media-invalid", "media duration is unavailable")

    # A three-hour recording is demuxed, not refused: there is no ceiling on
    # how long a person's own recording may be.
    long_recording = SimpleNamespace(
        demux=lambda _stream: [
            SimpleNamespace(pts=3 * 60 * 60 * 1000, dts=None, duration=1),
        ]
    )
    assert formats._demux_duration_ms(long_recording, stream) == 3 * 60 * 60 * 1000 + 1


def test_frame_at_or_after_observes_units_boundary_and_last_fallback():
    stream = SimpleNamespace(time_base=0.001)
    before = SimpleNamespace(pts=999)
    exact = SimpleNamespace(pts=1000)
    after = SimpleNamespace(pts=1001)
    assert formats._frame_at_or_after(iter((before, exact, after)), stream, 1000) is exact
    assert formats._frame_at_or_after(iter((before,)), stream, 1000) is before
    unknown = SimpleNamespace(pts=None)
    assert formats._frame_at_or_after(iter((unknown, after)), stream, 1000) is unknown

    with pytest.raises(MediaTranscriptionError) as error:
        formats._frame_at_or_after(iter(()), stream, 1000)
    _assert_media_error(error, "frames-invalid", "video produced no frame at sample time")


def test_remove_root_deletes_exact_owned_root(monkeypatch, tmp_path):
    adapter = formats.AvMediaAdapter()
    root = tmp_path / "owned"
    adapter._owned_roots.add(root)
    calls = []
    monkeypatch.setattr(
        formats.shutil,
        "rmtree",
        lambda path, *, ignore_errors: calls.append((path, ignore_errors)),
    )

    adapter._remove_root(root)

    assert calls == [(root, True)]
    assert adapter._owned_roots == set()


def test_decode_audio_sync_uses_exact_resampler_flush_and_concatenation(monkeypatch, tmp_path):
    cancellation = SimpleNamespace(raise_if_cancelled=lambda: calls.append("cancel"))
    decoded = [object(), object()]
    stream = object()
    calls = []

    class Container:
        streams = SimpleNamespace(audio=(stream,))

        def decode(self, selected):
            calls.append(("decode", selected))
            return decoded

    class Resampler:
        def __init__(self, *, format, layout, rate):
            calls.append(("resampler", format, layout, rate))

        def resample(self, frame):
            calls.append(("resample", frame))
            value = {decoded[0]: 1.0, decoded[1]: 2.0, None: 3.0}[frame]
            return _AudioFrame(np.array([[value]], dtype=np.float32))

    fake_av = SimpleNamespace(
        open=lambda selected, mode: (
            _Context(Container())
            if (selected, mode) == (str(tmp_path / "voice.wav"), "r")
            else None
        ),
        AudioResampler=Resampler,
    )
    monkeypatch.setitem(sys.modules, "av", fake_av)

    result = formats.AvMediaAdapter._decode_audio_sync(tmp_path / "voice.wav", cancellation)

    assert result.dtype == np.float32
    assert result.tolist() == [1.0, 2.0, 3.0]
    assert calls == [
        ("resampler", "fltp", "mono", 16_000),
        ("decode", stream),
        "cancel",
        ("resample", decoded[0]),
        "cancel",
        ("resample", decoded[1]),
        ("resample", None),
    ]

    windows = list(
        formats.AvMediaAdapter.decode_audio_windows(
            tmp_path / "voice.wav", cancellation, window_samples=2
        )
    )
    assert [(start, samples.tolist()) for start, samples in windows] == [
        (0, [1.0, 2.0]),
        (2, [3.0]),
    ]


def test_decode_audio_sync_rejects_missing_stream_with_exact_error(monkeypatch, tmp_path):
    container = SimpleNamespace(streams=SimpleNamespace(audio=()))
    fake_av = SimpleNamespace(open=lambda *_args, **_kwargs: _Context(container))
    monkeypatch.setitem(sys.modules, "av", fake_av)
    with pytest.raises(MediaTranscriptionError) as error:
        formats.AvMediaAdapter._decode_audio_sync(tmp_path / "silent.wav", CancellationController())
    _assert_media_error(error, "audio-missing", "media has no audio stream")


def test_sample_video_sync_seeks_and_writes_each_exact_frame(monkeypatch, tmp_path):
    root = tmp_path / "omnitensor-media-frames-fixed"
    root.mkdir()
    calls = []
    stream = SimpleNamespace(time_base=0.001)

    class Frame:
        def __init__(self, timestamp):
            self.timestamp = timestamp

        def to_image(self):
            calls.append(("image", self.timestamp))
            return self

        def save(self, path, *, format, optimize):
            calls.append(("save", path, format, optimize))
            path.write_bytes(b"png")

    class Container:
        streams = SimpleNamespace(video=(stream,))

        def seek(self, offset, *, stream, any_frame, backward):
            calls.append(("seek", offset, stream, any_frame, backward))

        def decode(self, selected):
            calls.append(("decode", selected))
            return iter(())

    monkeypatch.setattr(
        formats.tempfile,
        "mkdtemp",
        lambda *, prefix: calls.append(("prefix", prefix)) or str(root),
    )
    monkeypatch.setattr(
        formats,
        "_sample_timestamps",
        lambda duration: (0, 999) if duration == 1000 else (),
    )
    monkeypatch.setattr(
        formats,
        "_frame_at_or_after",
        lambda decoded, selected_stream, timestamp: (
            Frame(timestamp) if list(decoded) == [] and selected_stream is stream else None
        ),
    )

    def open_av(selected, *, mode):
        calls.append(("open", selected, mode))
        return _Context(Container())

    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=open_av))
    cancellation = SimpleNamespace(raise_if_cancelled=lambda: calls.append("cancel"))
    adapter = formats.AvMediaAdapter()
    media = MediaInfo(MediaModality.VIDEO, 1000, 16, 9, False)

    result = adapter._sample_video_sync(tmp_path / "clip.mp4", media, cancellation)

    assert result == (
        VisualFrame(root / "frame-00.png", 0),
        VisualFrame(root / "frame-01.png", 999),
    )
    assert root in adapter._owned_roots
    assert calls == [
        ("prefix", "omnitensor-media-frames-"),
        ("open", str(tmp_path / "clip.mp4"), "r"),
        "cancel",
        ("seek", 0, stream, False, True),
        ("decode", stream),
        ("image", 0),
        ("save", root / "frame-00.png", "PNG", False),
        "cancel",
        ("seek", 999, stream, False, True),
        ("decode", stream),
        ("image", 999),
        ("save", root / "frame-01.png", "PNG", False),
    ]


@pytest.mark.parametrize(
    "media",
    [
        MediaInfo(MediaModality.AUDIO, 1000, None, None, True),
        MediaInfo(MediaModality.VIDEO, None, 16, 9, False),
    ],
)
def test_sample_video_sync_rejects_unavailable_metadata_with_exact_error(tmp_path, media):
    with pytest.raises(MediaTranscriptionError) as error:
        formats.AvMediaAdapter()._sample_video_sync(
            tmp_path / "clip.mp4",
            media,
            CancellationController(),
        )
    _assert_media_error(error, "frames-invalid", "video metadata is unavailable")


@pytest.mark.asyncio
async def test_sample_video_sync_cancellation_removes_owned_root(monkeypatch, tmp_path):
    root = tmp_path / "frames"
    root.mkdir()
    monkeypatch.setattr(formats.tempfile, "mkdtemp", lambda **_kwargs: str(root))
    stream = SimpleNamespace(time_base=0.001)
    container = SimpleNamespace(streams=SimpleNamespace(video=(stream,)))
    monkeypatch.setitem(
        sys.modules,
        "av",
        SimpleNamespace(open=lambda *_args, **_kwargs: _Context(container)),
    )
    adapter = formats.AvMediaAdapter()
    cancellation = CancellationController()
    cancellation.cancel()

    with pytest.raises(asyncio.CancelledError):
        adapter._sample_video_sync(
            tmp_path / "clip.mp4",
            MediaInfo(MediaModality.VIDEO, 1000, 16, 9, False),
            cancellation,
        )

    assert not root.exists()
    assert root not in adapter._owned_roots
