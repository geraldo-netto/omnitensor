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
    MediaTranscriptionError,
    PresentationTranscriber,
    VisualFrame,
    VisualTranscriber,
    VisualTranscript,
)
from omnitensor.plugins.protocol import CancellationToken

from .archives import (
    MAX_PART_XML_BYTES,
    archive_entries,
    read_archive_entry,
    xml_root,
)
from .text import joined_visible_text as _joined_slide_text

# What this module calls itself in a refusal, which is the only thing that
# differs between reading a slide deck and reading any other OOXML container.
_KIND = "presentation"

_PRESENTATION_IMAGE_SUFFIXES = frozenset({".jpeg", ".jpg", ".png", ".webp"})
MAX_SLIDE_XML_BYTES = MAX_PART_XML_BYTES
MAX_PRESENTATION_IMAGE_BYTES = 16 * 1024 * 1024


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
            entries = archive_entries(archive, _KIND)
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
    if not slides:
        raise MediaTranscriptionError("presentation-invalid", "presentation slide count is invalid")
    return slides


def _require_image():
    """Pillow, or a refusal that names the install rather than the slide."""
    from importlib import import_module  # noqa: PLC0415 - deferred: an optional or heavy dependency

    try:
        return import_module("PIL.Image")
    except ImportError as error:
        raise MediaTranscriptionError(
            "presentation-runtime-unavailable", "install the Pillow image decoder"
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
    presentation = xml_root(
        read_archive_entry(archive, entries, presentation_name, MAX_SLIDE_XML_BYTES, _KIND), _KIND
    )
    relationships = _relationship_map(
        xml_root(
            read_archive_entry(archive, entries, rels_name, MAX_SLIDE_XML_BYTES, _KIND), _KIND
        ),
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
    slide = xml_root(
        read_archive_entry(archive, entries, slide_name, MAX_SLIDE_XML_BYTES, _KIND), _KIND
    )
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
            xml_root(
                read_archive_entry(archive, entries, relation_name, MAX_SLIDE_XML_BYTES, _KIND),
                _KIND,
            ),
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
    content = xml_root(
        read_archive_entry(archive, entries, "content.xml", MAX_SLIDE_XML_BYTES, _KIND), _KIND
    )
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
    # Every image on the slide is read. Stopping after four left the diagrams
    # at the bottom of a dense slide out of its description, with no sign that
    # anything had been skipped.
    return tuple(dict.fromkeys(selected))


def _extract_presentation(source: Path, root: Path) -> tuple[PresentationSlide, ...]:
    blueprints = _presentation_blueprints(source)
    slides = []
    with zipfile.ZipFile(source) as archive:
        entries = archive_entries(archive, _KIND)
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
    content = read_archive_entry(archive, entries, entry, MAX_PRESENTATION_IMAGE_BYTES, _KIND)
    suffix = PurePosixPath(entry).suffix.lower()
    path = root / f"slide-{slide_number:02d}-image-{image_number:02d}{suffix}"
    path.write_bytes(content)
    images = _require_image()
    try:
        with images.open(path) as image:
            image.verify()
        with images.open(path) as image:
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
