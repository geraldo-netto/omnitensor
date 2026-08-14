from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]


def _fixture_generator():
    path = ROOT / "scripts/generate-media-acceptance-fixtures.py"
    spec = importlib.util.spec_from_file_location("media_acceptance_fixtures", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_imagemagick_7_is_preferred_before_legacy_convert(monkeypatch):
    generator = _fixture_generator()
    looked_up = []

    def which(name):
        looked_up.append(name)
        return f"/tools/{name}"

    monkeypatch.setattr(generator.shutil, "which", which)

    assert generator._imagemagick_command() == "/tools/magick"
    assert looked_up == ["magick"]


def test_legacy_convert_is_the_only_imagemagick_fallback(monkeypatch):
    generator = _fixture_generator()
    looked_up = []

    def which(name):
        looked_up.append(name)
        return None if name == "magick" else "/tools/convert"

    monkeypatch.setattr(generator.shutil, "which", which)

    assert generator._imagemagick_command() == "/tools/convert"
    assert looked_up == ["magick", "convert"]

    monkeypatch.setattr(generator.shutil, "which", lambda _name: None)
    with pytest.raises(SystemExit, match="required fixture tool is unavailable: magick or convert"):
        generator._imagemagick_command()


def test_generation_dispatches_each_media_family_in_dependency_order(tmp_path, monkeypatch):
    generator = _fixture_generator()
    target = tmp_path / "fixtures"
    calls = []
    lookups = []

    def command(name):
        lookups.append(name)
        return f"/tools/{name}"

    def imagemagick_command():
        lookups.append("magick|convert")
        return "/tools/magick"

    monkeypatch.setattr(generator, "_command", command)
    monkeypatch.setattr(generator, "_imagemagick_command", imagemagick_command)
    monkeypatch.setattr(
        generator,
        "_generate_text_fixture",
        lambda path: calls.append(("text", path)) or "Привет мир",
    )
    monkeypatch.setattr(
        generator,
        "_generate_visual_fixtures",
        lambda path, tool, text: calls.append(("visual", path, tool, text)),
    )
    monkeypatch.setattr(
        generator,
        "_generate_presentation_fixtures",
        lambda path, tool, text: calls.append(("presentation", path, tool, text)),
    )
    monkeypatch.setattr(
        generator,
        "_generate_audio_fixtures",
        lambda path, tool: calls.append(("audio", path, tool)),
    )
    monkeypatch.setattr(
        generator,
        "_generate_video_fixtures",
        lambda path, tool: calls.append(("video", path, tool)),
    )
    monkeypatch.setattr(
        generator,
        "_write_fixture_manifest",
        lambda path: calls.append(("manifest", path)),
    )

    generator.generate(target)

    assert target.is_dir()
    assert lookups == ["ffmpeg", "magick|convert", "libreoffice"]
    assert calls == [
        ("text", target),
        ("visual", target, "/tools/magick", "Привет мир"),
        ("presentation", target, "/tools/libreoffice", "Привет мир"),
        ("audio", target, "/tools/ffmpeg"),
        ("video", target, "/tools/ffmpeg"),
        ("manifest", target),
    ]


def test_invalid_target_refuses_before_tool_discovery(tmp_path, monkeypatch):
    generator = _fixture_generator()
    monkeypatch.setattr(
        generator,
        "_command",
        lambda _name: pytest.fail("invalid targets must not discover tools"),
    )

    for target in (Path("relative"), tmp_path):
        with pytest.raises(SystemExit, match="target must be a new absolute directory"):
            generator.generate(target)


def test_text_fixture_preserves_koi8_r_source_bytes(tmp_path):
    generator = _fixture_generator()

    assert generator._generate_text_fixture(tmp_path) == "Привет мир"
    assert (tmp_path / "legacy-koi8-r.txt").read_bytes() == "Привет мир".encode("koi8-r")


def test_visual_generation_keeps_exact_format_commands(tmp_path, monkeypatch):
    generator = _fixture_generator()
    calls = []
    saved = []
    image = SimpleNamespace()
    image.convert = lambda mode: image
    image.transpose = lambda operation: ("transposed", operation)
    image.save = lambda path, **options: saved.append((path, options))
    monkeypatch.setattr(generator, "_run", calls.append)
    monkeypatch.setattr(generator.Image, "open", lambda _path: image)

    generator._generate_visual_fixtures(tmp_path, "/tools/magick", "Привет мир")

    assert calls == [
        ["/tools/magick", str(tmp_path / "complex.svg"), str(tmp_path / "complex.png")],
        [
            "/tools/magick",
            str(tmp_path / "complex.png"),
            "-quality",
            "88",
            str(tmp_path / "complex.jpg"),
        ],
        [
            "/tools/magick",
            str(tmp_path / "complex.png"),
            "-quality",
            "88",
            str(tmp_path / "complex.webp"),
        ],
    ]
    assert saved == [
        (
            tmp_path / "complex.tiff",
            {
                "save_all": True,
                "append_images": [("transposed", generator.Image.Transpose.FLIP_LEFT_RIGHT)],
                "compression": "tiff_lzw",
            },
        )
    ]


def test_presentation_generation_keeps_exact_format_commands(tmp_path, monkeypatch):
    generator = _fixture_generator()
    calls = []
    (tmp_path / "complex.png").write_bytes(b"png")
    monkeypatch.setattr(generator, "_run", calls.append)

    generator._generate_presentation_fixtures(tmp_path, "/tools/libreoffice", "Привет мир")

    assert calls == [
        [
            "/tools/libreoffice",
            "--headless",
            "--convert-to",
            "pptx",
            "--outdir",
            str(tmp_path),
            str(tmp_path / "complex.odp"),
        ],
        [
            "/tools/libreoffice",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(tmp_path),
            str(tmp_path / "complex.odp"),
        ],
    ]
    assert (tmp_path / "complex.odp").is_file()


def test_audio_and_video_generation_keep_all_format_commands(tmp_path, monkeypatch):
    generator = _fixture_generator()
    calls = []
    monkeypatch.setattr(generator, "_run", calls.append)

    generator._generate_audio_fixtures(tmp_path, "/tools/ffmpeg")
    generator._generate_video_fixtures(tmp_path, "/tools/ffmpeg")

    prefix = ["/tools/ffmpeg", "-hide_banner", "-loglevel", "error"]
    assert calls == [
        [
            *prefix,
            "-f",
            "lavfi",
            "-i",
            "flite=text='The blue chart has four bars and one circle':voice=slt",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(tmp_path / "speech.wav"),
        ],
        *[
            [
                *prefix,
                "-i",
                str(tmp_path / "speech.wav"),
                "-c:a",
                codec,
                str(tmp_path / f"speech.{suffix}"),
            ]
            for suffix, codec in (
                ("flac", "flac"),
                ("mp3", "libmp3lame"),
                ("ogg", "libvorbis"),
                ("opus", "libopus"),
                ("m4a", "aac"),
            )
        ],
        [
            *prefix,
            "-loop",
            "1",
            "-i",
            str(tmp_path / "complex.png"),
            "-i",
            str(tmp_path / "speech.wav"),
            "-t",
            "4",
            "-vf",
            "scale=960:640,format=yuv420p",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            str(tmp_path / "complex.mp4"),
        ],
        *[
            [
                *prefix,
                "-i",
                str(tmp_path / "complex.mp4"),
                "-c:v",
                video_codec,
                "-c:a",
                audio_codec,
                str(tmp_path / f"complex.{suffix}"),
            ]
            for suffix, video_codec, audio_codec in (
                ("webm", "libvpx-vp9", "libopus"),
                ("mkv", "mpeg4", "aac"),
                ("mov", "mpeg4", "aac"),
                ("avi", "mpeg4", "libmp3lame"),
            )
        ],
        [
            *prefix,
            "-i",
            str(tmp_path / "complex.mp4"),
            "-an",
            "-c:v",
            "mpeg4",
            "-q:v",
            "4",
            "-f",
            "m4v",
            str(tmp_path / "complex.m4v"),
        ],
    ]


def test_fixture_manifest_is_sorted_and_excludes_itself(tmp_path):
    generator = _fixture_generator()
    (tmp_path / "z.bin").write_bytes(b"z")
    (tmp_path / "a.bin").write_bytes(b"aa")

    generator._write_fixture_manifest(tmp_path)

    assert (tmp_path / "fixtures.json").read_text(encoding="utf-8") == (
        "{\n"
        '  "a.bin": {\n'
        '    "bytes": 2,\n'
        '    "sha256": "961b6dd3ede3cb8ecbaacbd68de040cd78eb2ed5889130cceb4c49268ea4d506"\n'
        "  },\n"
        '  "z.bin": {\n'
        '    "bytes": 1,\n'
        '    "sha256": "594e519ae499312b29433b7dd8a97ff068defcba9755b6d5d00e84c524d67b06"\n'
        "  }\n"
        "}\n"
    )
