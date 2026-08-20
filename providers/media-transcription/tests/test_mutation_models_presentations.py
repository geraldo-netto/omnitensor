from __future__ import annotations

import hashlib
import io
import json
import stat
import sys
import types
import zipfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import omnitensor_media_transcription.models as models
import omnitensor_media_transcription.presentations as presentations
import omnitensor_media_transcription.provider as provider
import omnitensor_media_transcription.qualification as qualification
import omnitensor_media_transcription.text as visible_text
import pytest
from hypothesis import given
from hypothesis import strategies as st
from PIL import Image

from omnitensor.plugins.media_transcription import (
    MediaTranscriptionError,
    SpeechSegment,
    SpeechTranscript,
    VisualFrame,
    VisualTranscript,
)
from omnitensor.sdk import BootstrapArtifact, CancellationController, PluginBootstrap


def _assert_media_error(error: pytest.ExceptionInfo, code: str, detail: str) -> None:
    assert error.value.code == code
    assert error.value.detail == detail
    assert str(error.value) == f"{code}: {detail}"


class _Lease:
    def __init__(self) -> None:
        self.stream = object()
        self.acquired = 0
        self.released = []

    @contextmanager
    def hold(self):
        self.acquired += 1
        yield

    def acquire(self):
        self.acquired += 1
        return self.stream

    def release(self, stream) -> None:
        self.released.append(stream)


def test_model_constructors_preserve_every_qualified_dependency(tmp_path):
    lease_path = tmp_path / "lease"
    lease_path.touch()
    lease = provider.VulkanLease(lease_path)
    assert lease._path == lease_path

    with pytest.raises(provider.QualifiedMediaError) as error:
        provider.VulkanLease(tmp_path / "missing")
    assert str(error.value) == "GPU accelerator lease is unavailable"

    decoder = object()
    speech_path = tmp_path / "speech.bin"
    whisper = provider.WhisperVulkanTranscriber(
        speech_path, decoder, lease, "Qualified speech device"
    )
    assert whisper._model_path == speech_path
    assert whisper._decoder is decoder
    assert whisper._lease is lease
    assert whisper._expected_device == "Qualified speech device"
    assert whisper._device_proven is False

    vision_path = tmp_path / "vision.gguf"
    projector_path = tmp_path / "mmproj.gguf"
    qwen = provider.QwenVulkanVisualTranscriber(
        vision_path, projector_path, lease, "Qualified vision device"
    )
    assert qwen._model_path == vision_path
    assert qwen._projector_path == projector_path
    assert qwen._lease is lease
    assert qwen._expected_device == "Qualified vision device"
    assert qwen._lease_stream is None
    assert qwen._handler is None
    assert qwen._llama is None
    assert qwen._log_callback is None
    assert qwen._abort_callback is None


def test_whisper_load_uses_exact_vulkan_configuration(monkeypatch, tmp_path):
    native = types.ModuleType("_pywhispercpp")
    native.callback = None
    native.freed = []

    def log_set(callback):
        native.callback = callback

    native.whisper_log_set = log_set
    native.whisper_free = native.freed.append
    monkeypatch.setitem(sys.modules, "_pywhispercpp", native)

    model_module = types.ModuleType("pywhispercpp.model")
    observed = {}

    class Model:
        def __init__(self, *args, **kwargs):
            observed["args"] = args
            observed["kwargs"] = kwargs
            self._ctx = object()
            native.callback(1, "ggml_vulkan: 0 = Qualified GPU ")
            native.callback(1, "(driver)\nwhisper_backend_init_gpu: using Vulkan0 backend\n")

    model_module.Model = Model
    monkeypatch.setitem(sys.modules, "pywhispercpp.model", model_module)

    model_path = tmp_path / "whisper.bin"
    transcriber = provider.WhisperVulkanTranscriber(model_path, object(), _Lease(), "Qualified GPU")
    model, logs = transcriber._load()

    assert observed == {
        "args": (str(model_path),),
        "kwargs": {
            "redirect_whispercpp_logs_to": False,
            "context_params": {"use_gpu": True, "flash_attn": True, "gpu_device": 0},
            "n_threads": 4,
            "print_progress": False,
            "print_realtime": False,
        },
    }
    assert logs == (
        "ggml_vulkan: 0 = Qualified GPU ",
        "(driver)\nwhisper_backend_init_gpu: using Vulkan0 backend\n",
    )
    assert native.callback is None
    assert transcriber._device_proven is True

    context = model._ctx
    provider._close_whisper(model)
    assert model._ctx is None
    assert native.freed == [context]


def test_whisper_load_reports_exact_runtime_and_device_failures(monkeypatch, tmp_path):
    transcriber = provider.WhisperVulkanTranscriber(
        tmp_path / "whisper.bin", object(), _Lease(), "Qualified GPU"
    )
    monkeypatch.setitem(sys.modules, "_pywhispercpp", None)
    monkeypatch.setitem(sys.modules, "pywhispercpp.model", None)
    with pytest.raises(provider.QualifiedMediaError) as error:
        transcriber._load()
    assert str(error.value) == "install pywhispercpp 1.5.0 from source with GGML_VULKAN=1"

    native = types.ModuleType("_pywhispercpp")
    native.callback = None
    native.freed = []
    native.whisper_log_set = lambda callback: setattr(native, "callback", callback)
    native.whisper_free = native.freed.append
    model_module = types.ModuleType("pywhispercpp.model")

    class Model:
        def __init__(self, *_args, **_kwargs):
            self._ctx = object()
            native.callback(1, "ggml_vulkan: 0 = Qualified GPU (driver)\n")

    model_module.Model = Model
    monkeypatch.setitem(sys.modules, "_pywhispercpp", native)
    monkeypatch.setitem(sys.modules, "pywhispercpp.model", model_module)
    with pytest.raises(provider.QualifiedMediaError) as error:
        transcriber._load()
    assert str(error.value) == "Whisper did not prove its qualified Vulkan device"
    assert len(native.freed) == 1


def test_whisper_transcription_preserves_exact_inference_and_segment_bounds(tmp_path):
    audio = SimpleNamespace(size=16_000)
    observed = {}

    class Model:
        def auto_detect_language(self, candidate):
            observed["detected_audio"] = candidate
            return (("he", 0.99), {"he": 0.99})

        def transcribe(self, candidate, **kwargs):
            observed["transcribed_audio"] = candidate
            observed["kwargs"] = kwargs
            return [
                SimpleNamespace(t0=0, t1=1, text="   "),
                SimpleNamespace(t0=0, t1=1, text=7),
                SimpleNamespace(t0=-2, t1=2, text=" first "),
                SimpleNamespace(t0=100, t1=100, text="zero"),
                SimpleNamespace(t0=90, t1=200, text="bounded"),
            ]

    model = Model()
    transcriber = provider.WhisperVulkanTranscriber(
        tmp_path / "whisper.bin", object(), _Lease(), "Qualified GPU"
    )
    transcriber._load = lambda: (model, ())
    cancellation = CancellationController()
    transcript = transcriber._transcribe_sync(audio, cancellation)

    assert observed["detected_audio"] is audio
    assert observed["transcribed_audio"] is audio
    assert set(observed["kwargs"]) == {
        "abort_callback",
        "language",
        "no_context",
        "print_progress",
        "print_realtime",
        "suppress_blank",
        "suppress_non_speech_tokens",
    }
    abort = observed["kwargs"].pop("abort_callback")
    assert callable(abort)
    assert abort() is False
    assert observed["kwargs"] == {
        "language": "he",
        "no_context": True,
        "print_progress": False,
        "print_realtime": False,
        "suppress_blank": True,
        "suppress_non_speech_tokens": True,
    }
    assert transcript == SpeechTranscript(
        "he",
        (
            SpeechSegment(0, 20, " first "),
            SpeechSegment(900, 1_000, "bounded"),
        ),
    )


def _qwen_transcriber(tmp_path) -> provider.QwenVulkanVisualTranscriber:
    return provider.QwenVulkanVisualTranscriber(
        tmp_path / "vision.gguf",
        tmp_path / "mmproj.gguf",
        _Lease(),
        "Qualified GPU",
    )


def test_qwen_transcription_preserves_exact_prompt_schema_and_decode_arguments(
    monkeypatch, tmp_path
):
    transcriber = _qwen_transcriber(tmp_path)
    observed = {}
    llama = SimpleNamespace(ctx=object())

    def completion(**kwargs):
        observed["completion"] = kwargs
        return iter(
            [
                None,
                {"choices": []},
                {"choices": [{"delta": {}}]},
                {"choices": [{"delta": {"content": '{"visibleText":"שלום",'}}]},
                {"choices": [{"delta": {"content": '"description":"Blue sign."}'}}]},
            ]
        )

    llama.create_chat_completion = completion
    native = SimpleNamespace()
    native.ggml_abort_callback = lambda callback: callback

    def set_abort_callback(context, callback, data):
        observed["abort"] = (context, callback, data)

    native.llama_set_abort_callback = set_abort_callback
    transcriber._ensure_loaded = lambda: llama
    transcriber._runtime = lambda: (object, native, object)
    monkeypatch.setattr(models, "_visual_payload", lambda path: b"\x00\xff")
    frame = VisualFrame(tmp_path / "frame.any", 37, "structured text")
    result = transcriber._transcribe_sync(frame, CancellationController())

    assert observed["abort"][0] is llama.ctx
    assert observed["abort"][1] is transcriber._abort_callback
    assert observed["abort"][2] is None
    assert observed["abort"][1](None) is False
    assert result == VisualTranscript(37, "structured text", "Blue sign.")
    assert observed["completion"] == {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Transcribe visual content faithfully. Preserve every legible script, "
                    "including Hebrew and mixed-direction text. Describe only visible facts."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AP8="},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Return JSON with visibleText containing exact legible text "
                            "(empty when none) and description containing a concise scene "
                            "transcription."
                        ),
                    },
                ],
            },
        ],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 0,
        "max_tokens": 768,
        "response_format": {
            "type": "json_object",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["visibleText", "description"],
                "properties": {
                    "visibleText": {"type": "string"},
                    "description": {"type": "string", "minLength": 1},
                },
            },
        },
        "stream": True,
    }


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ("not-json", "visual provider returned invalid JSON"),
        ('{"visibleText":"x"}', "visual provider returned invalid fields"),
        (
            '{"visibleText":"x","description":"d","extra":1}',
            "visual provider returned invalid fields",
        ),
    ],
)
def test_qwen_transcription_reports_exact_invalid_output(monkeypatch, tmp_path, payload, detail):
    transcriber = _qwen_transcriber(tmp_path)
    llama = SimpleNamespace(
        ctx=object(),
        create_chat_completion=lambda **_kwargs: iter(
            [{"choices": [{"delta": {"content": payload}}]}]
        ),
    )
    native = SimpleNamespace(
        ggml_abort_callback=lambda callback: callback,
        llama_set_abort_callback=lambda *_arguments: None,
    )
    transcriber._ensure_loaded = lambda: llama
    transcriber._runtime = lambda: (object, native, object)
    monkeypatch.setattr(models, "_visual_payload", lambda _path: b"image")

    with pytest.raises(MediaTranscriptionError) as error:
        transcriber._transcribe_sync(
            VisualFrame(tmp_path / "frame.png", None), CancellationController()
        )
    _assert_media_error(error, "visual-invalid", detail)


def test_qwen_load_and_release_preserve_every_vulkan_resource(tmp_path):
    lease = _Lease()
    transcriber = provider.QwenVulkanVisualTranscriber(
        tmp_path / "vision.gguf",
        tmp_path / "mmproj.gguf",
        lease,
        "Qualified GPU",
    )
    observed = {}
    native = SimpleNamespace(callback=None)
    native.llama_log_callback = lambda callback: callback
    native.llama_log_set = lambda callback, data: observed.update(log=(callback, data))

    class Stack:
        closed = False

        def close(self):
            self.closed = True

    class Handler:
        def __init__(self, *args, **kwargs):
            observed["handler"] = (args, kwargs)
            self._exit_stack = Stack()

    class Llama:
        def __init__(self, **kwargs):
            observed["llama"] = kwargs
            self.closed = False
            callback = observed["log"][0]
            callback(1, b"using device Vulkan0 (Qualified GPU) ", None)
            callback(1, b"(0000:00:00.0)\noffloaded 33/33 layers to GPU\n", None)

        def close(self):
            self.closed = True

    transcriber._runtime = lambda: (Llama, native, Handler)
    llama = transcriber._ensure_loaded()

    assert lease.acquired == 1
    assert transcriber._lease_stream is lease.stream
    assert transcriber._log_callback is observed["log"][0]
    assert observed["log"][1] is None
    assert observed["handler"] == ((str(tmp_path / "mmproj.gguf"),), {"verbose": False})
    assert observed["llama"] == {
        "model_path": str(tmp_path / "vision.gguf"),
        "chat_handler": transcriber._handler,
        "n_ctx": 8192,
        "n_batch": 1024,
        "n_gpu_layers": -1,
        "main_gpu": 0,
        "offload_kqv": True,
        "op_offload": True,
        "flash_attn": True,
        "verbose": False,
    }
    assert transcriber._ensure_loaded() is llama

    handler = transcriber._handler
    transcriber._abort_callback = object()
    transcriber._release_sync()
    assert handler._exit_stack.closed is True
    assert llama.closed is True
    assert lease.released == [lease.stream]
    assert transcriber._handler is None
    assert transcriber._llama is None
    assert transcriber._abort_callback is None
    assert transcriber._lease_stream is None


def test_qwen_load_releases_resources_and_reports_exact_unproved_offload(tmp_path):
    lease = _Lease()
    transcriber = provider.QwenVulkanVisualTranscriber(
        tmp_path / "vision.gguf",
        tmp_path / "mmproj.gguf",
        lease,
        "Qualified GPU",
    )
    observed = {}
    native = SimpleNamespace()
    native.llama_log_callback = lambda callback: callback
    native.llama_log_set = lambda callback, _data: observed.update(callback=callback)

    class Stack:
        closed = False

        def close(self):
            self.closed = True

    class Handler:
        def __init__(self, *_args, **_kwargs):
            self._exit_stack = Stack()
            observed["handler"] = self

    class Llama:
        def __init__(self, **_kwargs):
            self.closed = False
            observed["llama"] = self
            observed["callback"](
                1,
                b"using device Vulkan0 (Qualified GPU) (0000:00:00.0)\n"
                b"offloaded 32/33 layers to GPU\n",
                None,
            )

        def close(self):
            self.closed = True

    transcriber._runtime = lambda: (Llama, native, Handler)
    with pytest.raises(provider.QualifiedMediaError) as error:
        transcriber._ensure_loaded()
    assert str(error.value) == "Qwen VL did not prove full qualified Vulkan offload"
    assert observed["handler"]._exit_stack.closed is True
    assert observed["llama"].closed is True
    assert lease.released == [lease.stream]


def test_qwen_runtime_and_whisper_context_failures_are_exact(monkeypatch):
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_chat_format", None)
    with pytest.raises(provider.QualifiedMediaError) as error:
        provider.QwenVulkanVisualTranscriber._runtime()
    assert str(error.value) == "Vulkan llama-cpp-python runtime is unavailable"

    native = types.ModuleType("_pywhispercpp")
    native.freed = []
    native.whisper_free = native.freed.append
    monkeypatch.setitem(sys.modules, "_pywhispercpp", native)
    context = object()
    model = SimpleNamespace(_ctx=context)
    provider._close_whisper(model)
    assert model._ctx is None
    assert native.freed == [context]


@given(
    st.lists(st.text(alphabet=st.characters(exclude_characters="\x00"), max_size=24), max_size=8)
)
def test_visible_text_normalization_matches_bounded_reference(parts):
    expected = "\n".join(part.strip() for part in parts if part.strip())
    assert visible_text.joined_visible_text(parts) == expected


def test_visible_text_and_slide_descriptions_enforce_exact_boundaries():
    assert visible_text.joined_visible_text([" alpha ", "", 0, None]) == "alpha\n0\nNone"
    assert visible_text.joined_visible_text(["x" * 16_384]) == "x" * 16_384
    for candidate in ("x" * 16_385, "safe\x00unsafe"):
        with pytest.raises(MediaTranscriptionError) as error:
            visible_text.joined_visible_text([candidate])
        _assert_media_error(
            error, "presentation-invalid", "presentation slide text exceeds its limit"
        )

    assert presentations._slide_description("", ("one", "two", "one")) == "one; two"
    assert presentations._slide_description("text", ()) == "Presentation slide containing text."
    assert presentations._slide_description("", ()) == "Blank presentation slide."
    assert presentations._slide_description("", ("x" * 16_384,)) == "x" * 16_384
    for descriptions in (("",), ("x" * 16_385,)):
        with pytest.raises(MediaTranscriptionError) as error:
            presentations._slide_description("", descriptions)
        _assert_media_error(
            error,
            "presentation-invalid",
            "presentation image descriptions exceed their limit",
        )


class _ArchiveInfo:
    def __init__(
        self,
        filename: str,
        size: int = 1,
        *,
        external_attr: int = 0,
        flag_bits: int = 0,
        directory: bool = False,
    ) -> None:
        self.filename = filename
        self.file_size = size
        self.external_attr = external_attr
        self.flag_bits = flag_bits
        self._directory = directory

    def is_dir(self) -> bool:
        return self._directory


class _Archive:
    def __init__(self, infos, content=b"x") -> None:
        self._infos = infos
        self._content = content

    def infolist(self):
        return self._infos

    def read(self, _info):
        return self._content


def test_archive_entries_accept_exact_limits_and_reject_every_unsafe_shape(monkeypatch):
    monkeypatch.setattr(presentations, "MAX_ARCHIVE_ENTRIES", 2)
    monkeypatch.setattr(presentations, "MAX_ARCHIVE_EXPANDED_BYTES", 3)
    first = _ArchiveInfo("one.xml", 1)
    second = _ArchiveInfo("two.xml", 2)
    assert presentations._archive_entries(_Archive([first, second])) == {
        "one.xml": first,
        "two.xml": second,
    }

    too_many = [first, second, _ArchiveInfo("three.xml", 0)]
    with pytest.raises(MediaTranscriptionError) as error:
        presentations._archive_entries(_Archive(too_many))
    _assert_media_error(error, "presentation-invalid", "presentation archive has too many entries")

    monkeypatch.setattr(presentations, "MAX_ARCHIVE_ENTRIES", 10)
    for infos in (
        [_ArchiveInfo("/absolute.xml")],
        [_ArchiveInfo("../escape.xml")],
        [_ArchiveInfo("folder\\escape.xml")],
        [_ArchiveInfo("encrypted.xml", flag_bits=1)],
        [_ArchiveInfo("link.xml", external_attr=stat.S_IFLNK << 16)],
        [_ArchiveInfo("same.xml"), _ArchiveInfo("same.xml")],
    ):
        with pytest.raises(MediaTranscriptionError) as error:
            presentations._archive_entries(_Archive(infos))
        _assert_media_error(error, "presentation-invalid", "presentation archive entry is unsafe")

    with pytest.raises(MediaTranscriptionError) as error:
        presentations._archive_entries(_Archive([_ArchiveInfo("one", 2), _ArchiveInfo("two", 2)]))
    _assert_media_error(
        error, "presentation-invalid", "presentation archive expands beyond its limit"
    )


def test_archive_reads_allow_zero_and_maximum_bytes_but_reject_invalid_entries():
    zero = _ArchiveInfo("zero", 0)
    maximum = _ArchiveInfo("maximum", 3)
    assert (
        presentations._read_archive_entry(_Archive([zero], b""), {"zero": zero}, "zero", 3) == b""
    )
    assert (
        presentations._read_archive_entry(
            _Archive([maximum], b"abc"), {"maximum": maximum}, "maximum", 3
        )
        == b"abc"
    )

    invalid = (
        ({}, "missing"),
        ({"directory": _ArchiveInfo("directory", directory=True)}, "directory"),
        ({"large": _ArchiveInfo("large", 4)}, "large"),
    )
    for entries, name in invalid:
        with pytest.raises(MediaTranscriptionError) as error:
            presentations._read_archive_entry(_Archive([]), entries, name, 3)
        _assert_media_error(
            error, "presentation-invalid", "presentation archive entry is unavailable"
        )

    truncated = _ArchiveInfo("truncated", 3)
    with pytest.raises(MediaTranscriptionError) as error:
        presentations._read_archive_entry(
            _Archive([truncated], b"ab"), {"truncated": truncated}, "truncated", 3
        )
    _assert_media_error(error, "presentation-invalid", "presentation archive entry was truncated")


def test_xml_and_relationship_parsing_preserve_exact_safety_contract():
    with pytest.raises(MediaTranscriptionError) as error:
        presentations._xml_root(b"<broken")
    _assert_media_error(error, "presentation-invalid", "presentation XML is invalid")

    root = presentations._xml_root(
        b"""<Relationships>
        <Relationship/>
        <Relationship Id="missing-target"/>
        <Relationship Target="missing-id"/>
        <Relationship Id="external" Target="https://example.test/a" TargetMode="External"/>
        <Relationship Id="valid" Target="../media/image.png"/>
        </Relationships>"""
    )
    assert presentations._relationship_map(root, "ppt/slides/slide1.xml") == {
        "valid": "ppt/media/image.png"
    }

    for target in ("/absolute.png", "../../../escape.png"):
        unsafe = presentations._xml_root(
            f'<Relationships><Relationship Id="unsafe" Target="{target}"/></Relationships>'.encode()
        )
        with pytest.raises(MediaTranscriptionError) as error:
            presentations._relationship_map(unsafe, "ppt/slides/slide1.xml")
        _assert_media_error(error, "presentation-invalid", "presentation relationship is unsafe")


def test_valid_image_selection_requires_membership_suffix_and_deduplication():
    names = ["one.png", "two.jpg", "three.webp", "four.jpeg", "five.png"]
    entries = {name: object() for name in names}
    entries["unsupported.gif"] = object()
    assert presentations._valid_image_entries(
        ["missing.png", "unsupported.gif", *names], entries
    ) == tuple(names)
    assert presentations._valid_image_entries(["one.png", "one.png"], entries) == ("one.png",)


def _pptx(path: Path, *, image: bytes | None = None) -> Path:
    presentation_xml = """<p:presentation
      xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
      xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
      <p:sldIdLst><p:sldId id="256" r:id="rId1"/></p:sldIdLst>
    </p:presentation>"""
    rels_xml = """<Relationships>
      <Relationship Id="rId1" Target="slides/slide1.xml"/>
    </Relationships>"""
    image_xml = '<a:blip r:embed="image"/>' if image is not None else ""
    slide_xml = f"""<p:sld
      xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
      xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
      xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
      <a:t>Hello</a:t><a:t></a:t>{image_xml}
    </p:sld>"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ppt/presentation.xml", presentation_xml)
        archive.writestr("ppt/_rels/presentation.xml.rels", rels_xml)
        archive.writestr("ppt/slides/slide1.xml", slide_xml)
        if image is not None:
            archive.writestr(
                "ppt/slides/_rels/slide1.xml.rels",
                '<Relationships><Relationship Id="image" '
                'Target="../media/IMAGE.PNG"/></Relationships>',
            )
            archive.writestr("ppt/media/IMAGE.PNG", image)
    return path


def _odp(path: Path) -> Path:
    content = """<office:document-content
      xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
      xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"
      xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
      xmlns:xlink="http://www.w3.org/1999/xlink">
      <office:body><office:presentation><draw:page>
        <text:p>Hello <text:span>world</text:span></text:p>
        <draw:image/><draw:image xlink:href="Pictures/diagram.png"/>
      </draw:page></office:presentation></office:body>
    </office:document-content>"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("content.xml", content)
        archive.writestr("Pictures/diagram.png", b"image")
    return path


def _png_bytes() -> bytes:
    stream = io.BytesIO()
    Image.new("RGB", (2, 2), "blue").save(stream, format="PNG")
    return stream.getvalue()


def test_pptx_and_odp_blueprints_preserve_relationship_text_and_image_defaults(tmp_path):
    pptx = _pptx(tmp_path / "slides.pptx", image=_png_bytes())
    pptx_blueprints = presentations._presentation_blueprints(pptx)
    assert pptx_blueprints == (presentations._SlideBlueprint(1, "Hello", ("ppt/media/IMAGE.PNG",)),)

    odp = _odp(tmp_path / "slides.odp")
    odp_blueprints = presentations._presentation_blueprints(odp)
    assert odp_blueprints == (
        presentations._SlideBlueprint(1, "Hello world", ("Pictures/diagram.png",)),
    )

    missing = tmp_path / "missing.pptx"
    with zipfile.ZipFile(missing, "w") as archive:
        archive.writestr(
            "ppt/presentation.xml",
            """<p:presentation
              xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
              <p:sldIdLst><p:sldId/></p:sldIdLst>
            </p:presentation>""",
        )
        archive.writestr("ppt/_rels/presentation.xml.rels", "<Relationships/>")
    with pytest.raises(MediaTranscriptionError) as error:
        presentations._presentation_blueprints(missing)
    _assert_media_error(
        error, "presentation-invalid", "presentation slide relationship is unavailable"
    )


def test_presentation_blueprint_archive_and_slide_count_failures_are_exact(monkeypatch, tmp_path):
    invalid = tmp_path / "invalid.pptx"
    invalid.write_bytes(b"not a zip")
    with pytest.raises(MediaTranscriptionError) as error:
        presentations._presentation_blueprints(invalid)
    _assert_media_error(error, "presentation-invalid", "presentation archive could not be decoded")

    unsupported = tmp_path / "slides.zip"
    with zipfile.ZipFile(unsupported, "w"):
        pass
    with pytest.raises(MediaTranscriptionError) as error:
        presentations._presentation_blueprints(unsupported)
    _assert_media_error(error, "presentation-unsupported", "presentation type is unsupported")

    bounded = tmp_path / "bounded.pptx"
    with zipfile.ZipFile(bounded, "w"):
        pass
    monkeypatch.setattr(
        presentations,
        "_pptx_blueprints",
        lambda _archive, _entries: (
            presentations._SlideBlueprint(1, "one", ()),
            presentations._SlideBlueprint(2, "two", ()),
        ),
    )
    assert len(presentations._presentation_blueprints(bounded)) == 2
    monkeypatch.setattr(presentations, "_pptx_blueprints", lambda *_arguments: ())
    with pytest.raises(MediaTranscriptionError) as error:
        presentations._presentation_blueprints(bounded)
    _assert_media_error(error, "presentation-invalid", "presentation slide count is invalid")


def test_presentation_extraction_numbers_images_from_one(tmp_path):
    source = _pptx(tmp_path / "slides.pptx", image=_png_bytes())
    root = tmp_path / "images"
    root.mkdir()
    slides = presentations._extract_presentation(source, root)
    assert slides[0].number == 1
    assert slides[0].text == "Hello"
    assert slides[0].images == (root / "slide-01-image-01.png",)
    assert slides[0].images[0].read_bytes() == _png_bytes()


def test_presentation_image_writer_preserves_lowercase_suffix_and_pixel_boundary(
    monkeypatch, tmp_path
):
    info = _ArchiveInfo("PHOTO.PNG", 3)
    archive = _Archive([info], b"png")

    class OpenedImage:
        def __init__(self, width, height):
            self.width = width
            self.height = height

        def __enter__(self):
            return self

        def __exit__(self, *_arguments):
            return False

        def verify(self):
            return None

    dimensions = iter([(50_000_000, 1), (50_000_000, 1)])
    monkeypatch.setattr(Image, "open", lambda _path: OpenedImage(*next(dimensions)))
    path = presentations._write_presentation_image(
        archive, {"PHOTO.PNG": info}, "PHOTO.PNG", tmp_path, 2, 3
    )
    assert path == tmp_path / "slide-02-image-03.png"
    assert path.read_bytes() == b"png"

    dimensions = iter([(50_000_001, 1), (50_000_001, 1)])
    monkeypatch.setattr(Image, "open", lambda _path: OpenedImage(*next(dimensions)))
    with pytest.raises(MediaTranscriptionError) as error:
        presentations._write_presentation_image(
            archive, {"PHOTO.PNG": info}, "PHOTO.PNG", tmp_path, 1, 1
        )
    _assert_media_error(error, "presentation-invalid", "presentation image is invalid")


@pytest.mark.asyncio
async def test_presentation_transcriber_uses_exact_workspace_and_always_removes_it(
    monkeypatch, tmp_path
):
    root = tmp_path / "workspace"
    root.mkdir()
    observed = {}

    def mkdtemp(*, prefix):
        observed["prefix"] = prefix
        return str(root)

    def rmtree(path, ignore_errors):
        observed["cleanup"] = (path, ignore_errors)

    monkeypatch.setattr(presentations.tempfile, "mkdtemp", mkdtemp)
    monkeypatch.setattr(presentations.shutil, "rmtree", rmtree)
    monkeypatch.setattr(
        presentations,
        "_extract_presentation",
        lambda _source, candidate_root: (
            presentations.PresentationSlide(4, "slide text", (candidate_root / "image.png",)),
        ),
    )

    class Vision:
        async def transcribe(self, frame, cancellation):
            cancellation.raise_if_cancelled()
            assert frame == VisualFrame(root / "image.png", None)
            return VisualTranscript(None, "image text", "diagram")

        async def release(self):
            return None

    result = await presentations.PresentationArchiveTranscriber(Vision()).transcribe(
        tmp_path / "slides.pptx", CancellationController()
    )
    assert result == (VisualTranscript(None, "slide text\nimage text", "diagram", 4),)
    assert observed == {
        "prefix": "omnitensor-presentation-",
        "cleanup": (root, True),
    }


def _qualification_document(evidence_digest: str) -> dict:
    return {
        "version": 1,
        "recordedAt": "2026-08-13",
        "qualified": True,
        "devices": {"speech": "Speech GPU", "vision": "Vision GPU"},
        "providerVersion": "0.2.0",
        "runtimes": {"llama-cpp-python": "0.3.34", "pywhispercpp": "1.5.0"},
        "artifacts": {
            provider.VISION_ARTIFACT_ID: "vision",
            f"{provider.VISION_ARTIFACT_ID}/mmproj": "projector",
            provider.SPEECH_ARTIFACT_ID: "speech",
        },
        "evidenceSha256": evidence_digest,
    }


def test_create_wires_one_shared_adapter_lease_and_vision_into_every_port(monkeypatch, tmp_path):
    vision_path = tmp_path / "vision.gguf"
    projector_path = tmp_path / provider.VISION_PROJECTOR
    speech_path = tmp_path / "speech.bin"
    lease_path = tmp_path / "lease"
    for path in (vision_path, projector_path, speech_path, lease_path):
        path.touch()
    bootstrap = PluginBootstrap(
        provider.PLUGIN_ID,
        (
            BootstrapArtifact(
                provider.VISION_ARTIFACT_ID,
                "1",
                "gguf",
                "vision",
                vision_path,
                ((provider.VISION_PROJECTOR, "projector"),),
            ),
            BootstrapArtifact(
                provider.SPEECH_ARTIFACT_ID,
                "1",
                "ggml-whisper",
                "speech",
                speech_path,
            ),
        ),
        None,
        lease_path,
    )
    monkeypatch.setattr(provider, "_qualification", lambda: _qualification_document("unused"))
    monkeypatch.setattr(provider, "current_plugin_bootstrap", lambda plugin_id: bootstrap)
    plugin = provider.create()

    assert plugin._probe is plugin._frames
    assert plugin._speech._model_path == speech_path
    assert plugin._speech._decoder is plugin._probe
    assert plugin._speech._lease is plugin._vision._lease
    assert plugin._speech._expected_device == "Speech GPU"
    assert plugin._vision._model_path == vision_path
    assert plugin._vision._projector_path == projector_path
    assert plugin._vision._expected_device == "Vision GPU"
    assert plugin._presentations._vision is plugin._vision
    assert plugin._documents._vision is plugin._vision

    projector_path.unlink()
    with pytest.raises(provider.QualifiedMediaError) as error:
        provider.create()
    assert str(error.value) == "qualified visual projector is unavailable"


def test_qualification_evidence_uses_exact_package_digest_and_size_bounds(monkeypatch):
    observed = []

    class Resource:
        def __init__(self, raw):
            self.raw = raw

        def joinpath(self, name):
            observed.append(name)
            return self

        def read_bytes(self):
            return self.raw

    resource = Resource(b"x")

    def files(package):
        observed.append(package)
        return resource

    monkeypatch.setattr(qualification.importlib.resources, "files", files)
    monkeypatch.setattr(qualification, "MAX_QUALIFICATION_EVIDENCE_BYTES", 2)
    document = {"evidenceSha256": hashlib.sha256(b"x").hexdigest()}
    qualification._validate_qualification_evidence(document)
    assert observed == [
        "omnitensor_media_transcription",
        qualification.QUALIFICATION_EVIDENCE,
    ]

    resource.raw = b"xy"
    document["evidenceSha256"] = hashlib.sha256(b"xy").hexdigest()
    qualification._validate_qualification_evidence(document)
    for raw in (b"", b"xyz", b"wrong"):
        resource.raw = raw
        with pytest.raises(provider.QualifiedMediaError) as error:
            qualification._validate_qualification_evidence(document)
        assert str(error.value) == "media qualification evidence differs from receipt"

    def unreadable():
        raise OSError("unavailable")

    resource.read_bytes = unreadable
    with pytest.raises(provider.QualifiedMediaError) as error:
        qualification._validate_qualification_evidence(document)
    assert str(error.value) == "media qualification evidence is unreadable"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("version", 2, "media qualification receipt is invalid"),
        ("recordedAt", "2026-08-12", "media qualification receipt date is invalid"),
        ("qualified", False, "media provider has no accepted qualification"),
        (
            "devices",
            {"speech": "", "vision": "Vision GPU"},
            "media qualification devices are invalid",
        ),
        ("providerVersion", "9.9.9", "media provider differs from qualification"),
        ("runtimes", {}, "media runtimes differ from qualification"),
    ],
)
def test_qualification_receipt_rejects_every_exact_frozen_field(
    monkeypatch, tmp_path, field, value, message
):
    evidence = b"accepted"
    document = _qualification_document(hashlib.sha256(evidence).hexdigest())
    document[field] = value
    package = tmp_path / "package"
    package.mkdir()
    (package / "qualification.json").write_text(json.dumps(document), encoding="utf-8")
    (package / qualification.QUALIFICATION_EVIDENCE).write_bytes(evidence)
    monkeypatch.setattr(qualification.importlib.resources, "files", lambda package_name: package)
    versions = {
        "omnitensor-media-transcription": "0.2.0",
        "llama-cpp-python": "0.3.34",
        "pywhispercpp": "1.5.0",
    }
    monkeypatch.setattr(qualification.importlib.metadata, "version", versions.__getitem__)

    with pytest.raises(provider.QualifiedMediaError) as error:
        qualification.load_qualification()
    assert str(error.value) == message
