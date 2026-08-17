"""Bounded PPTX and ODP archive extraction adapters."""

from __future__ import annotations

import asyncio
import posixpath
import shutil
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from omnitensor.plugins.media_transcription import (
    MAX_PRESENTATION_SLIDES,
    MediaTranscriptionError,
    PresentationTranscriber,
    VisualFrame,
    VisualTranscriber,
    VisualTranscript,
)
from omnitensor.plugins.protocol import CancellationToken

from .text import joined_visible_text as _joined_slide_text

_PRESENTATION_IMAGE_SUFFIXES = frozenset({".jpeg", ".jpg", ".png", ".webp"})
MAX_ARCHIVE_ENTRIES = 4096
MAX_ARCHIVE_EXPANDED_BYTES = 256 * 1024 * 1024
MAX_SLIDE_XML_BYTES = 4 * 1024 * 1024
MAX_PRESENTATION_IMAGE_BYTES = 16 * 1024 * 1024
MAX_IMAGES_PER_SLIDE = 4


@dataclass(frozen=True, slots=True)
class PresentationSlide:
    number: int
    text: str
    images: tuple[Path, ...]


class PresentationArchiveTranscriber(PresentationTranscriber):
    """Transcribe bounded PPTX/ODP slide text and qualified embedded images."""

    def __init__(self, vision: VisualTranscriber) -> None:
        self._vision = vision

    async def transcribe(
        self, source: Path, cancellation: CancellationToken
    ) -> tuple[VisualTranscript, ...]:
        root = Path(tempfile.mkdtemp(prefix="omnitensor-presentation-"))
        try:
            slides = await asyncio.to_thread(_extract_presentation, source, root)
            results = []
            for slide in slides:
                cancellation.raise_if_cancelled()
                visible = [slide.text] if slide.text else []
                descriptions = []
                for image_path in slide.images:
                    visual = await self._vision.transcribe(
                        VisualFrame(image_path, None), cancellation
                    )
                    if visual.visible_text:
                        visible.append(visual.visible_text)
                    descriptions.append(visual.description)
                description = _slide_description(slide.text, descriptions)
                results.append(
                    VisualTranscript(
                        None,
                        _joined_slide_text(visible),
                        description,
                        slide.number,
                    )
                )
            return tuple(results)
        finally:
            await asyncio.to_thread(shutil.rmtree, root, True)


@dataclass(frozen=True, slots=True)
class _SlideBlueprint:
    number: int
    text: str
    image_entries: tuple[str, ...]


def _presentation_slide_count(source: Path) -> int:
    return len(_presentation_blueprints(source))


def _presentation_blueprints(source: Path) -> tuple[_SlideBlueprint, ...]:
    try:
        with zipfile.ZipFile(source) as archive:
            entries = _archive_entries(archive)
            suffix = source.suffix.lower()
            if suffix == ".pptx":
                slides = _pptx_blueprints(archive, entries)
            elif suffix == ".odp":
                slides = _odp_blueprints(archive, entries)
            else:  # pragma: no cover - caller suffix gate
                raise MediaTranscriptionError(
                    "presentation-unsupported", "presentation type is unsupported"
                )
    except MediaTranscriptionError:
        raise
    except (OSError, zipfile.BadZipFile, RuntimeError, ValueError) as error:
        raise MediaTranscriptionError(
            "presentation-invalid", "presentation archive could not be decoded"
        ) from error
    if not 1 <= len(slides) <= MAX_PRESENTATION_SLIDES:
        raise MediaTranscriptionError("presentation-invalid", "presentation slide count is invalid")
    return slides


def _archive_entries(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    if len(infos) > MAX_ARCHIVE_ENTRIES:
        raise MediaTranscriptionError(
            "presentation-invalid", "presentation archive has too many entries"
        )
    entries = {}
    expanded = 0
    for info in infos:
        path = PurePosixPath(info.filename)
        mode = info.external_attr >> 16
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in info.filename
            or info.flag_bits & 1
            or mode & 0o170000 == 0o120000
            or info.filename in entries
        ):
            raise MediaTranscriptionError(
                "presentation-invalid", "presentation archive entry is unsafe"
            )
        expanded += info.file_size
        if expanded > MAX_ARCHIVE_EXPANDED_BYTES:
            raise MediaTranscriptionError(
                "presentation-invalid", "presentation archive expands beyond its limit"
            )
        entries[info.filename] = info
    return entries


def _read_archive_entry(
    archive: zipfile.ZipFile,
    entries: Mapping[str, zipfile.ZipInfo],
    name: str,
    maximum: int,
) -> bytes:
    info = entries.get(name)
    if info is None or info.is_dir() or not 0 <= info.file_size <= maximum:
        raise MediaTranscriptionError(
            "presentation-invalid", "presentation archive entry is unavailable"
        )
    content = archive.read(info)
    if len(content) != info.file_size:
        raise MediaTranscriptionError(
            "presentation-invalid", "presentation archive entry was truncated"
        )
    return content


def _xml_root(content: bytes):
    """Parse presentation XML with the hardened parser, or say which failed.

    The import used to sit inside the same `except Exception` as the parse, so
    an installation missing `defusedxml` told the person their presentation was
    invalid. It is a declared dependency of this distribution, so its absence
    is a broken install rather than a fact about their file — and reporting it
    as one sends them off to re-export a document that was fine.
    """
    try:
        from defusedxml import ElementTree
    except ImportError as error:
        raise MediaTranscriptionError(
            "presentation-runtime-unavailable",
            "install the defusedxml presentation parser",
        ) from error
    try:
        return ElementTree.fromstring(content)
    except Exception as error:
        raise MediaTranscriptionError(
            "presentation-invalid", "presentation XML is invalid"
        ) from error


def _relationship_map(root, base_file: str) -> dict[str, str]:
    relationships = {}
    for item in root:
        identifier = item.attrib.get("Id")
        target = item.attrib.get("Target")
        if not identifier or not target or item.attrib.get("TargetMode") == "External":
            continue
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(base_file), target))
        if PurePosixPath(resolved).is_absolute() or ".." in PurePosixPath(resolved).parts:
            raise MediaTranscriptionError(
                "presentation-invalid", "presentation relationship is unsafe"
            )
        relationships[identifier] = resolved
    return relationships


def _pptx_blueprints(
    archive: zipfile.ZipFile, entries: Mapping[str, zipfile.ZipInfo]
) -> tuple[_SlideBlueprint, ...]:
    presentation_name = "ppt/presentation.xml"
    rels_name = "ppt/_rels/presentation.xml.rels"
    presentation = _xml_root(
        _read_archive_entry(archive, entries, presentation_name, MAX_SLIDE_XML_BYTES)
    )
    relationships = _relationship_map(
        _xml_root(_read_archive_entry(archive, entries, rels_name, MAX_SLIDE_XML_BYTES)),
        presentation_name,
    )
    relation_attribute = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    slide_tag = "{http://schemas.openxmlformats.org/presentationml/2006/main}sldId"
    slide_names = [
        relationships.get(item.attrib.get(relation_attribute, ""), "")
        for item in presentation.iter(slide_tag)
    ]
    if any(name not in entries for name in slide_names):
        raise MediaTranscriptionError(
            "presentation-invalid", "presentation slide relationship is unavailable"
        )
    return tuple(
        _pptx_slide_blueprint(archive, entries, name, index)
        for index, name in enumerate(slide_names, 1)
    )


def _pptx_slide_blueprint(
    archive: zipfile.ZipFile,
    entries: Mapping[str, zipfile.ZipInfo],
    slide_name: str,
    number: int,
) -> _SlideBlueprint:
    slide = _xml_root(_read_archive_entry(archive, entries, slide_name, MAX_SLIDE_XML_BYTES))
    text_tag = "{http://schemas.openxmlformats.org/drawingml/2006/main}t"
    text = _joined_slide_text(item.text or "" for item in slide.iter(text_tag))
    relation_name = posixpath.join(
        posixpath.dirname(slide_name),
        "_rels",
        f"{posixpath.basename(slide_name)}.rels",
    )
    relationships = {}
    if relation_name in entries:
        relationships = _relationship_map(
            _xml_root(_read_archive_entry(archive, entries, relation_name, MAX_SLIDE_XML_BYTES)),
            slide_name,
        )
    embed_attribute = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
    image_tag = "{http://schemas.openxmlformats.org/drawingml/2006/main}blip"
    image_entries = tuple(
        relationships.get(item.attrib.get(embed_attribute, ""), "")
        for item in slide.iter(image_tag)
    )
    return _SlideBlueprint(number, text, _valid_image_entries(image_entries, entries))


def _odp_blueprints(
    archive: zipfile.ZipFile, entries: Mapping[str, zipfile.ZipInfo]
) -> tuple[_SlideBlueprint, ...]:
    content = _xml_root(_read_archive_entry(archive, entries, "content.xml", MAX_SLIDE_XML_BYTES))
    page_tag = "{urn:oasis:names:tc:opendocument:xmlns:drawing:1.0}page"
    paragraph_tag = "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}p"
    image_tag = "{urn:oasis:names:tc:opendocument:xmlns:drawing:1.0}image"
    href_attribute = "{http://www.w3.org/1999/xlink}href"
    slides = []
    for number, page in enumerate(content.iter(page_tag), 1):
        text = _joined_slide_text("".join(item.itertext()) for item in page.iter(paragraph_tag))
        image_entries = tuple(item.attrib.get(href_attribute, "") for item in page.iter(image_tag))
        slides.append(_SlideBlueprint(number, text, _valid_image_entries(image_entries, entries)))
    return tuple(slides)


def _valid_image_entries(
    candidates: Sequence[str], entries: Mapping[str, zipfile.ZipInfo]
) -> tuple[str, ...]:
    selected = []
    for candidate in candidates:
        normalized = posixpath.normpath(candidate)
        suffix = PurePosixPath(normalized).suffix.lower()
        if normalized in entries and suffix in _PRESENTATION_IMAGE_SUFFIXES:
            selected.append(normalized)
        if len(selected) == MAX_IMAGES_PER_SLIDE:
            break
    return tuple(dict.fromkeys(selected))


def _extract_presentation(source: Path, root: Path) -> tuple[PresentationSlide, ...]:
    blueprints = _presentation_blueprints(source)
    slides = []
    with zipfile.ZipFile(source) as archive:
        entries = _archive_entries(archive)
        for blueprint in blueprints:
            paths = tuple(
                _write_presentation_image(archive, entries, entry, root, blueprint.number, index)
                for index, entry in enumerate(blueprint.image_entries, 1)
            )
            slides.append(PresentationSlide(blueprint.number, blueprint.text, paths))
    return tuple(slides)


def _write_presentation_image(
    archive: zipfile.ZipFile,
    entries: Mapping[str, zipfile.ZipInfo],
    entry: str,
    root: Path,
    slide_number: int,
    image_number: int,
) -> Path:
    content = _read_archive_entry(archive, entries, entry, MAX_PRESENTATION_IMAGE_BYTES)
    suffix = PurePosixPath(entry).suffix.lower()
    path = root / f"slide-{slide_number:02d}-image-{image_number:02d}{suffix}"
    path.write_bytes(content)
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            if image.width * image.height > 50_000_000:
                raise ValueError("image is too large")
    except Exception as error:
        raise MediaTranscriptionError(
            "presentation-invalid", "presentation image is invalid"
        ) from error
    return path


def _slide_description(text: str, descriptions: Sequence[str]) -> str:
    if descriptions:
        description = "; ".join(dict.fromkeys(descriptions))
        if 0 < len(description) <= 16_384:
            return description
        raise MediaTranscriptionError(
            "presentation-invalid", "presentation image descriptions exceed their limit"
        )
    return "Presentation slide containing text." if text else "Blank presentation slide."


__all__ = ["PresentationArchiveTranscriber"]
