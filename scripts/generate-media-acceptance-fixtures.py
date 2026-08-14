#!/usr/bin/env python3
"""Generate independent, mixed-format media acceptance fixtures."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

from PIL import Image


def _command(name: str) -> str:
    resolved = shutil.which(name)
    if resolved is None:
        raise SystemExit(f"required fixture tool is unavailable: {name}")
    return resolved


def _imagemagick_command() -> str:
    for name in ("magick", "convert"):
        resolved = shutil.which(name)
        if resolved is not None:
            return resolved
    raise SystemExit("required fixture tool is unavailable: magick or convert")


def _run(arguments: list[str]) -> None:
    completed = subprocess.run(arguments, check=False, capture_output=True, timeout=120)
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise SystemExit(f"fixture command failed: {detail}")


def _svg(cyrillic: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="960" height="640">
<rect width="960" height="640" fill="#f8fafc"/>
<rect x="24" y="24" width="912" height="592" rx="24" fill="none"
 stroke="#183b66" stroke-width="8"/>
<text x="70" y="100" font-family="Noto Sans Hebrew" font-size="52"
 fill="#101828">שלום עולם</text>
<text x="70" y="170" font-family="DejaVu Sans" font-size="44"
 fill="#101828">{cyrillic} · café résumé</text>
<text x="70" y="225" font-family="DejaVu Sans" font-size="30"
 fill="#344054">Quarterly blue chart — 2026</text>
<line x1="90" y1="540" x2="850" y2="540" stroke="#101828" stroke-width="5"/>
<rect x="100" y="390" width="80" height="150" fill="#175cd3"/>
<rect x="245" y="310" width="80" height="230" fill="#2e90fa"/>
<rect x="390" y="240" width="80" height="300" fill="#53b1fd"/>
<rect x="535" y="350" width="80" height="190" fill="#1570ef"/>
<circle cx="780" cy="385" r="105" fill="#84caff" stroke="#175cd3" stroke-width="7"/>
</svg>"""


def _odp(path: Path, png: bytes, cyrillic: str) -> None:
    content = f"""<?xml version="1.0" encoding="UTF-8"?>
<office:document-content
 xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"
 xmlns:presentation="urn:oasis:names:tc:opendocument:xmlns:presentation:1.0"
 xmlns:svg="urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
 xmlns:xlink="http://www.w3.org/1999/xlink" office:version="1.3">
 <office:body><office:presentation>
  <draw:page draw:name="slide1">
   <draw:frame svg:x="1cm" svg:y="1cm" svg:width="24cm" svg:height="3cm">
    <draw:text-box><text:p>שלום עולם · {cyrillic} · KOI8-R decoded</text:p></draw:text-box>
   </draw:frame>
   <draw:frame svg:x="4cm" svg:y="5cm" svg:width="16cm" svg:height="10cm">
    <draw:image xlink:href="Pictures/chart.png" xlink:type="simple"/>
   </draw:frame>
  </draw:page>
  <draw:page draw:name="slide2">
   <draw:frame svg:x="1cm" svg:y="1cm" svg:width="24cm" svg:height="4cm">
    <draw:text-box><text:p>Second slide: blue circle and four chart bars</text:p></draw:text-box>
   </draw:frame>
  </draw:page>
 </office:presentation></office:body>
</office:document-content>"""
    manifest = """<?xml version="1.0" encoding="UTF-8"?>
<manifest:manifest
 xmlns:manifest="urn:oasis:names:tc:opendocument:xmlns:manifest:1.0"
 manifest:version="1.3">
 <manifest:file-entry manifest:full-path="/"
  manifest:media-type="application/vnd.oasis.opendocument.presentation"/>
 <manifest:file-entry manifest:full-path="content.xml" manifest:media-type="text/xml"/>
 <manifest:file-entry manifest:full-path="Pictures/chart.png" manifest:media-type="image/png"/>
</manifest:manifest>"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "mimetype",
            "application/vnd.oasis.opendocument.presentation",
            compress_type=zipfile.ZIP_STORED,
        )
        archive.writestr("content.xml", content)
        archive.writestr("META-INF/manifest.xml", manifest)
        archive.writestr("Pictures/chart.png", png)


def _generate_text_fixture(target: Path) -> str:
    koi8 = "Привет мир".encode("koi8-r")
    (target / "legacy-koi8-r.txt").write_bytes(koi8)
    return koi8.decode("koi8-r")


def _generate_visual_fixtures(target: Path, imagemagick: str, cyrillic: str) -> None:
    (target / "complex.svg").write_text(_svg(cyrillic), encoding="utf-8")
    _run([imagemagick, str(target / "complex.svg"), str(target / "complex.png")])
    _run(
        [
            imagemagick,
            str(target / "complex.png"),
            "-quality",
            "88",
            str(target / "complex.jpg"),
        ]
    )
    _run(
        [
            imagemagick,
            str(target / "complex.png"),
            "-quality",
            "88",
            str(target / "complex.webp"),
        ]
    )

    first = Image.open(target / "complex.png").convert("RGB")
    second = first.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    first.save(
        target / "complex.tiff",
        save_all=True,
        append_images=[second],
        compression="tiff_lzw",
    )


def _generate_presentation_fixtures(target: Path, libreoffice: str, cyrillic: str) -> None:
    _odp(target / "complex.odp", (target / "complex.png").read_bytes(), cyrillic)
    _run(
        [
            libreoffice,
            "--headless",
            "--convert-to",
            "pptx",
            "--outdir",
            str(target),
            str(target / "complex.odp"),
        ]
    )
    _run(
        [
            libreoffice,
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(target),
            str(target / "complex.odp"),
        ]
    )


def _generate_audio_fixtures(target: Path, ffmpeg: str) -> None:
    speech = "The blue chart has four bars and one circle"
    _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"flite=text='{speech}':voice=slt",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(target / "speech.wav"),
        ]
    )
    audio_codecs = {
        "flac": "flac",
        "mp3": "libmp3lame",
        "ogg": "libvorbis",
        "opus": "libopus",
        "m4a": "aac",
    }
    for suffix, codec in audio_codecs.items():
        _run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(target / "speech.wav"),
                "-c:a",
                codec,
                str(target / f"speech.{suffix}"),
            ]
        )


def _generate_video_fixtures(target: Path, ffmpeg: str) -> None:
    _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-loop",
            "1",
            "-i",
            str(target / "complex.png"),
            "-i",
            str(target / "speech.wav"),
            "-t",
            "4",
            "-vf",
            "scale=960:640,format=yuv420p",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            str(target / "complex.mp4"),
        ]
    )
    video_codecs = {
        "webm": ("libvpx-vp9", "libopus"),
        "mkv": ("mpeg4", "aac"),
        "mov": ("mpeg4", "aac"),
        "avi": ("mpeg4", "libmp3lame"),
    }
    for suffix, (video_codec, audio_codec) in video_codecs.items():
        _run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(target / "complex.mp4"),
                "-c:v",
                video_codec,
                "-c:a",
                audio_codec,
                str(target / f"complex.{suffix}"),
            ]
        )
    _run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(target / "complex.mp4"),
            "-an",
            "-c:v",
            "mpeg4",
            "-q:v",
            "4",
            "-f",
            "m4v",
            str(target / "complex.m4v"),
        ]
    )


def _write_fixture_manifest(target: Path) -> None:
    summary = {
        path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(target.iterdir())
        if path.is_file()
    }
    (target / "fixtures.json").write_text(
        __import__("json").dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def generate(target: Path) -> None:
    if not target.is_absolute() or target.exists():
        raise SystemExit("target must be a new absolute directory")
    target.mkdir(parents=True, mode=0o700)
    ffmpeg = _command("ffmpeg")
    imagemagick = _imagemagick_command()
    libreoffice = _command("libreoffice")

    cyrillic = _generate_text_fixture(target)
    _generate_visual_fixtures(target, imagemagick, cyrillic)
    _generate_presentation_fixtures(target, libreoffice, cyrillic)
    _generate_audio_fixtures(target, ffmpeg)
    _generate_video_fixtures(target, ffmpeg)
    _write_fixture_manifest(target)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target", type=Path)
    generate(parser.parse_args().target)


if __name__ == "__main__":
    main()
