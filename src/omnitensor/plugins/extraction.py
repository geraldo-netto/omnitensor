"""Bounded extraction from documents that were built to break parsers.

Every parser, renderer, and OCR engine worth using is a large C library that
has had memory-safety bugs and will have more, and it is being pointed at a
file the user downloaded.  So none of them are called from here: an adapter is
an injected port that the profile runs inside its own worker, and this layer is
what stands between that adapter and the rest of the service.

What that means concretely.  A document that never finishes parsing is bounded
by a deadline, not by hope.  An adapter that raises, or dies, does not take the
job with it — whatever pages it already produced are kept and the result is
marked partial, because a half-read contract is still worth showing as long as
it is *labelled* half-read.  Output is bounded in pages, in characters, and in
regions, so a document that decodes to a hundred million characters costs a
truncation flag rather than the process.

Text is normalised before anyone sees it: control characters go, and so do
bidirectional overrides, which can make a line of text render as something
entirely different from what it says.  Images never travel as bytes — a region
is a bounded descriptor, and the pixels stay wherever the renderer put them.
"""

from __future__ import annotations

import asyncio
import re
import unicodedata
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Protocol, runtime_checkable

from .document_ingestion import DEFAULT_MAX_EXPANSION_RATIO, DocumentIngestor
from .ingestion import IngestedFile

DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_PAGES = 2_000
DEFAULT_MAX_CHARACTERS = 5_000_000
DEFAULT_MAX_REGIONS_PER_PAGE = 512
MAX_COORDINATE = 1_000_000
MAX_IMAGE_PIXELS = 512_000_000
_KEPT_CONTROLS = frozenset({"\n", "\t"})
# Overrides and embeddings that make rendered text differ from its content.
_BIDI_CONTROLS = frozenset("‪‫‬‭‮⁦⁧⁨⁩‎‏")
_RUNS_OF_SPACES = re.compile(r" {2,}")


class AdapterKind(StrEnum):
    """What an adapter does, so a grant can name it."""

    PARSER = "parser"
    RENDERER = "renderer"
    OCR = "ocr"


class ExtractionOutcome(StrEnum):
    """How an extraction ended — never inferred from an empty result."""

    SUCCEEDED = "succeeded"
    TRUNCATED = "truncated"
    TIMED_OUT = "timed-out"
    CRASHED = "crashed"
    REFUSED = "refused"


class BlockKind(StrEnum):
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    TABLE = "table"
    CAPTION = "caption"
    OTHER = "other"


class ExtractionError(ValueError):
    """Stable extraction configuration failure."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class LayoutBlock:
    """One positioned run of text on a page."""

    kind: BlockKind
    x: int
    y: int
    width: int
    height: int
    text: str


@dataclass(frozen=True, slots=True)
class ImageRegion:
    """One image on a page, described rather than carried."""

    image_id: str
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class PageContent:
    """What an adapter reports for one page, before normalisation."""

    page_number: int
    text: str = ""
    blocks: Sequence[LayoutBlock] = ()
    images: Sequence[ImageRegion] = ()


@dataclass(frozen=True, slots=True)
class NormalisedPage:
    """One page after bounding and normalisation."""

    page_number: int
    text: str
    blocks: tuple[LayoutBlock, ...]
    images: tuple[ImageRegion, ...]


@dataclass(frozen=True, slots=True)
class ExtractionLimits:
    """Every ceiling an untrusted document is held to."""

    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_pages: int = DEFAULT_MAX_PAGES
    max_characters: int = DEFAULT_MAX_CHARACTERS
    max_regions_per_page: int = DEFAULT_MAX_REGIONS_PER_PAGE
    max_image_pixels: int = MAX_IMAGE_PIXELS

    def for_file(self, size_bytes: object) -> ExtractionLimits:
        """The ceilings this particular document deserves.

        A fixed 5,000,000 characters is a policy number: it refuses the second
        half of a long report for no reason the report knows about. What a file
        can honestly hold is arithmetic — a page costs at least a byte, and text
        expands by at most the ingestor's declared ratio — so both ceilings
        follow the file's own size. A caller that set either one explicitly
        keeps it; only the defaults are derived.
        """
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 1:
            return self
        characters = self.max_characters
        if characters == DEFAULT_MAX_CHARACTERS:
            characters = max(characters, size_bytes * DEFAULT_MAX_EXPANSION_RATIO)
        pages = self.max_pages
        if pages == DEFAULT_MAX_PAGES:
            pages = max(pages, size_bytes)
        if characters == self.max_characters and pages == self.max_pages:
            return self
        return replace(self, max_characters=characters, max_pages=pages)

    def validate(self) -> None:
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not 0 < self.timeout_seconds <= 3600
        ):
            raise ExtractionError("limits-invalid", "timeout_seconds must be in (0, 3600]")
        for name in ("max_pages", "max_characters", "max_regions_per_page", "max_image_pixels"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ExtractionError("limits-invalid", f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Everything extraction produced, and how far it actually got."""

    outcome: ExtractionOutcome
    pages: tuple[NormalisedPage, ...] = ()
    text: str = ""
    detail: str = ""
    dropped_pages: int = 0
    dropped_characters: int = 0
    dropped_regions: int = 0

    @property
    def partial(self) -> bool:
        """True when there is output but it does not describe the whole document."""
        return self.outcome is not ExtractionOutcome.SUCCEEDED and bool(self.pages)

    @property
    def complete(self) -> bool:
        return self.outcome is ExtractionOutcome.SUCCEEDED

    def document(self) -> dict:
        return {
            "outcome": str(self.outcome),
            "pageCount": len(self.pages),
            "characters": len(self.text),
            "partial": self.partial,
            "detail": self.detail,
            "droppedPages": self.dropped_pages,
            "droppedCharacters": self.dropped_characters,
            "droppedRegions": self.dropped_regions,
        }


@runtime_checkable
class ExtractionAdapter(Protocol):
    """A parser, renderer, or OCR engine, running in its own worker.

    It yields pages as it produces them so that a crash halfway through still
    leaves the pages that came before it.
    """

    kind: AdapterKind

    def pages(self, item: IngestedFile) -> AsyncIterator[PageContent]: ...


def normalise_text(text: object, limit: int) -> tuple[str, int]:
    """Return safe, bounded text and how many characters were dropped."""
    if not isinstance(text, str):
        return "", 0
    normalised = unicodedata.normalize("NFC", text)
    kept = [
        character
        for character in normalised
        if character in _KEPT_CONTROLS
        or (
            character not in _BIDI_CONTROLS
            and unicodedata.category(character) not in {"Cc", "Cf", "Co", "Cs"}
        )
    ]
    # Runs of spaces are collapsed but tabs are left alone: a parser emits
    # them to separate columns, and squashing those merges table cells.
    collapsed = _RUNS_OF_SPACES.sub(" ", "".join(kept))
    stripped = "\n".join(line.strip() for line in collapsed.splitlines()).strip()
    if len(stripped) <= limit:
        return stripped, 0
    return stripped[:limit], len(stripped) - limit


class DocumentExtractor:
    """Run one isolated adapter under hard bounds and contain what it does."""

    def __init__(
        self,
        adapter: ExtractionAdapter,
        *,
        limits: ExtractionLimits | None = None,
        ingestor: DocumentIngestor | None = None,
    ) -> None:
        if not hasattr(adapter, "pages") or not callable(adapter.pages):
            raise ExtractionError("adapter-invalid", "adapter must expose a pages() iterator")
        if not isinstance(getattr(adapter, "kind", None), AdapterKind):
            raise ExtractionError("adapter-invalid", "adapter must declare an AdapterKind")
        if ingestor is not None and not isinstance(ingestor, DocumentIngestor):
            raise ExtractionError("adapter-invalid", "ingestor must be a DocumentIngestor")
        limits = limits or ExtractionLimits()
        limits.validate()
        self._adapter = adapter
        self._limits = limits
        self._ingestor = ingestor

    @property
    def kind(self) -> AdapterKind:
        return self._adapter.kind

    @property
    def limits(self) -> ExtractionLimits:
        return self._limits

    async def extract(
        self, item: IngestedFile, *, declared_uncompressed_bytes: int | None = None
    ) -> ExtractionResult:
        """Extract one document, containing every way the adapter can fail."""
        if not isinstance(item, IngestedFile):
            raise ExtractionError("item-invalid", "item must be an IngestedFile")
        refusal = self._expansion_refusal(item, declared_uncompressed_bytes)
        if refusal is not None:
            # The bomb is refused from what it claims, before a parser is
            # started at all — expanding to find out is the attack.
            return ExtractionResult(ExtractionOutcome.REFUSED, detail=refusal)
        collector = _Collector(self._limits.for_file(item.size_bytes))
        try:
            async with asyncio.timeout(self._limits.timeout_seconds):
                await self._drain(item, collector)
        except TimeoutError:
            return collector.result(ExtractionOutcome.TIMED_OUT, "extraction exceeded its deadline")
        except asyncio.CancelledError:
            # A cancelled job is the caller's decision, not an adapter failure.
            raise
        except Exception as error:  # noqa: BLE001 - containment is the point
            return collector.result(ExtractionOutcome.CRASHED, f"{type(error).__name__}: {error}")
        if collector.dropped_pages or collector.dropped_characters or collector.dropped_regions:
            return collector.result(ExtractionOutcome.TRUNCATED, "output exceeded its bounds")
        return collector.result(ExtractionOutcome.SUCCEEDED, "")

    async def _drain(self, item: IngestedFile, collector: _Collector) -> None:
        iterator = self._adapter.pages(item)
        async for page in iterator:
            if not collector.accept(page):
                # Stop pulling once a ceiling is reached: continuing would let
                # the document decide how long extraction runs.
                aclose = getattr(iterator, "aclose", None)
                if aclose is not None:
                    await aclose()
                return

    def _expansion_refusal(self, item: IngestedFile, declared: int | None) -> str | None:
        if declared is None or self._ingestor is None:
            return None
        refusal = self._ingestor.expansion_refusal(item, declared_uncompressed_bytes=declared)
        return None if refusal is None else f"{refusal.reason}: {refusal.detail}"


@dataclass
class _Collector:
    """Accumulate normalised pages while every ceiling is enforced."""

    limits: ExtractionLimits
    pages: list[NormalisedPage] = field(default_factory=list)
    characters: int = 0
    dropped_pages: int = 0
    dropped_characters: int = 0
    dropped_regions: int = 0

    def accept(self, page: object) -> bool:
        """Add one page; ``False`` means no further page will be taken."""
        if not isinstance(page, PageContent):
            self.dropped_pages += 1
            return True
        if len(self.pages) >= self.limits.max_pages:
            self.dropped_pages += 1
            return False
        remaining = self.limits.max_characters - self.characters
        text, dropped = normalise_text(page.text, max(remaining, 0))
        self.dropped_characters += dropped
        blocks, images, dropped_regions = self._regions(page)
        self.dropped_regions += dropped_regions
        self.characters += len(text)
        self.pages.append(
            NormalisedPage(_page_number(page.page_number, len(self.pages)), text, blocks, images)
        )
        return self.characters < self.limits.max_characters

    def _regions(
        self, page: PageContent
    ) -> tuple[tuple[LayoutBlock, ...], tuple[ImageRegion, ...], int]:
        blocks: list[LayoutBlock] = []
        images: list[ImageRegion] = []
        dropped = 0
        for block in _bounded(page.blocks, self.limits.max_regions_per_page):
            normalised = self._block(block)
            if normalised is None:
                dropped += 1
            else:
                blocks.append(normalised)
        for image in _bounded(page.images, self.limits.max_regions_per_page):
            if self._image_ok(image):
                images.append(image)
            else:
                dropped += 1
        dropped += _overflow(page.blocks, self.limits.max_regions_per_page)
        dropped += _overflow(page.images, self.limits.max_regions_per_page)
        return tuple(blocks), tuple(images), dropped

    def _block(self, block: object) -> LayoutBlock | None:
        if not isinstance(block, LayoutBlock) or not isinstance(block.kind, BlockKind):
            return None
        if not _geometry_ok(block.x, block.y, block.width, block.height):
            return None
        text, dropped = normalise_text(block.text, self.limits.max_characters)
        self.dropped_characters += dropped
        return LayoutBlock(block.kind, block.x, block.y, block.width, block.height, text)

    def _image_ok(self, image: object) -> bool:
        if not isinstance(image, ImageRegion):
            return False
        if not isinstance(image.image_id, str) or not 1 <= len(image.image_id) <= 128:
            return False
        if not _geometry_ok(image.x, image.y, image.width, image.height):
            return False
        return image.width * image.height <= self.limits.max_image_pixels

    def result(self, outcome: ExtractionOutcome, detail: str) -> ExtractionResult:
        return ExtractionResult(
            outcome,
            tuple(self.pages),
            "\n".join(page.text for page in self.pages if page.text),
            detail,
            self.dropped_pages,
            self.dropped_characters,
            self.dropped_regions,
        )


TRUNCATED_SOURCE_CODE = "source-truncated"


def measured_failure_detail(code: object, detail: object) -> str:
    """A refusal whose explanation was measured, not looked up from a table.

    Every other workload failure is a code with a fixed sentence beside it.
    A truncated source is different: what went unread is counted at the moment
    of refusal, so the counts have to travel with the code or they are lost.

    Returns ``""`` when the code's explanation is a fixed sentence elsewhere.
    """
    if str(code) != TRUNCATED_SOURCE_CODE or not detail:
        return ""
    return f"{code}: {detail}"


def truncation_summary(result: ExtractionResult) -> str:
    """Name what a truncated extraction left out, in counts a person can act on.

    Returns ``""`` when nothing was dropped. Deliberately free of file names
    and content: a workload refusal carries neither.
    """
    if result.outcome is not ExtractionOutcome.TRUNCATED:
        return ""
    parts = [
        f"{count} {noun}"
        for count, noun in (
            (result.dropped_pages, "page(s)"),
            (result.dropped_characters, "character(s)"),
            (result.dropped_regions, "layout region(s)"),
        )
        if count
    ]
    dropped = ", ".join(parts) if parts else "part of the document"
    detail = f" ({result.detail})" if result.detail else ""
    return f"read only {len(result.pages)} page(s); {dropped} went unread{detail}"


def _page_number(value: object, fallback: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_COORDINATE:
        return fallback + 1
    return value


def _geometry_ok(*values: object) -> bool:
    return all(
        not isinstance(value, bool) and isinstance(value, int) and 0 <= value <= MAX_COORDINATE
        for value in values
    )


def _bounded(items: object, limit: int) -> tuple:
    if isinstance(items, (str, bytes)) or not isinstance(items, Sequence):
        return ()
    return tuple(items)[:limit]


def _overflow(items: object, limit: int) -> int:
    if isinstance(items, (str, bytes)) or not isinstance(items, Sequence):
        return 0
    return max(len(items) - limit, 0)
