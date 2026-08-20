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
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Protocol, runtime_checkable

from ..stable_error import StableError
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


class ExtractionError(StableError, ValueError):
    """Stable extraction configuration failure."""


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
class ResumePoint:
    """Where the next extraction pass starts reading.

    A document larger than one pass is read in several, so a pass that stops
    at a ceiling says where it stopped rather than reporting the rest lost.
    The offset is into the *normalised* text of that page, which is what the
    reader is given and is derived from the raw page deterministically.
    """

    page_index: int
    character_offset: int = 0


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
    resume: ResumePoint | None = None

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
    stripped = normalised_text(text)
    if len(stripped) <= limit:
        return stripped, 0
    return stripped[:limit], len(stripped) - limit


def normalised_text(text: object) -> str:
    """Safe text, whole. Bounding it is a separate decision from cleaning it.

    Splitting these two apart is what lets a pass resume: an offset into this
    result is stable, because the same raw page always normalises the same way,
    while an offset into the raw text would move with every control character
    the cleaning removed.
    """
    if not isinstance(text, str):
        return ""
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
    return "\n".join(line.strip() for line in collapsed.splitlines()).strip()


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

    async def extract_all(
        self, item: IngestedFile, *, declared_uncompressed_bytes: int | None = None
    ) -> ExtractionResult:
        """Every page of a document, in as many passes as it takes.

        One pass is bounded — in pages, in characters and by a deadline — so a
        document larger than a pass used to come back `TRUNCATED` and every
        workload turned that into `source-truncated`: a selection refused for
        being long. It is answered instead, the way generation is: each pass
        resumes where the last one stopped and the results accumulate, so the
        only thing a ceiling now decides is how many passes it takes.

        A pass that lands on the same resume point twice stops the loop and
        is reported truncated. Nothing in this module can produce that — each
        pass takes at least one character — but an adapter whose pages change
        between reads could, and a loop that never ends is worse than a
        partial answer that says so.
        """
        merged: ExtractionResult | None = None
        resume: ResumePoint | None = None
        while True:
            result = await self.extract(
                item, declared_uncompressed_bytes=declared_uncompressed_bytes, resume=resume
            )
            merged = result if merged is None else _merge_passes(merged, result, resume)
            if result.resume is None:
                return merged
            if result.resume == resume:
                return replace(
                    merged,
                    outcome=ExtractionOutcome.TRUNCATED,
                    detail="a page is larger than one extraction pass",
                    resume=None,
                )
            resume = result.resume

    async def extract(
        self,
        item: IngestedFile,
        *,
        declared_uncompressed_bytes: int | None = None,
        resume: ResumePoint | None = None,
    ) -> ExtractionResult:
        """Extract one pass over a document, containing every adapter failure.

        `resume` starts the pass at a page and an offset a previous pass
        reported; the result carries the next such point, or `None` when the
        document is finished. `extract_all` is the loop over both.
        """
        if not isinstance(item, IngestedFile):
            raise ExtractionError("item-invalid", "item must be an IngestedFile")
        refusal = self._expansion_refusal(item, declared_uncompressed_bytes)
        if refusal is not None:
            # The bomb is refused from what it claims, before a parser is
            # started at all — expanding to find out is the attack.
            return ExtractionResult(ExtractionOutcome.REFUSED, detail=refusal)
        collector = _Collector(self._limits.for_file(item.size_bytes), resume)
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
        if (
            collector.resume is not None
            or collector.dropped_pages
            or collector.dropped_characters
            or collector.dropped_regions
        ):
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
    resume_from: ResumePoint | None = None
    pages: list[NormalisedPage] = field(default_factory=list)
    characters: int = 0
    dropped_pages: int = 0
    dropped_characters: int = 0
    dropped_regions: int = 0
    source_index: int = 0
    resume: ResumePoint | None = None

    def accept(self, page: object) -> bool:
        """Add one page; ``False`` means no further page will be taken."""
        index = self.source_index
        self.source_index += 1
        start = self._offset_for(index)
        if start is None:
            # Already read by an earlier pass. Skipped rather than dropped:
            # nothing was lost, and counting it would report it as loss.
            return True
        if not isinstance(page, PageContent):
            self.dropped_pages += 1
            return True
        if len(self.pages) >= self.limits.max_pages:
            # Not dropped either. The next pass starts exactly here.
            self.resume = ResumePoint(index, start)
            return False
        text = normalised_text(page.text)[start:]
        remaining = self.limits.max_characters - self.characters
        if len(text) > remaining:
            taken = text[:remaining]
            self.resume = ResumePoint(index, start + len(taken))
            if taken:
                self._add(page, index, taken, start)
            return False
        self._add(page, index, text, start)
        if self.characters >= self.limits.max_characters:
            # This page fitted exactly. Stopping without a resume point would
            # end the document here and call it complete.
            self.resume = ResumePoint(self.source_index, 0)
            return False
        return True

    def _add(self, page: PageContent, index: int, text: str, start: int) -> None:
        # A page continued from an earlier pass carries no regions: they were
        # emitted whole with its first part, and repeating them would count
        # every figure on that page twice.
        blocks, images, dropped_regions = ((), (), 0) if start else self._regions(page)
        self.dropped_regions += dropped_regions
        self.characters += len(text)
        self.pages.append(
            NormalisedPage(_page_number(page.page_number, index), text, blocks, images)
        )

    def _offset_for(self, index: int) -> int | None:
        """Where this pass starts inside source page `index`, or `None` to skip."""
        if self.resume_from is None:
            return 0
        if index < self.resume_from.page_index:
            return None
        if index == self.resume_from.page_index:
            return self.resume_from.character_offset
        return 0

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
            self.resume if outcome is ExtractionOutcome.TRUNCATED else None,
        )


def _merge_passes(
    first: ExtractionResult, second: ExtractionResult, resume: ResumePoint | None
) -> ExtractionResult:
    """One document's two passes as one result.

    A page split across the boundary is rejoined, so the reader sees the page
    it would have seen from a single pass rather than two entries claiming the
    same page number.
    """
    pages = list(first.pages)
    incoming = list(second.pages)
    if (
        resume is not None
        and resume.character_offset
        and pages
        and incoming
        and pages[-1].page_number == incoming[0].page_number
    ):
        head = pages.pop()
        tail = incoming.pop(0)
        pages.append(
            NormalisedPage(
                head.page_number,
                head.text + tail.text,
                head.blocks + tail.blocks,
                head.images + tail.images,
            )
        )
    pages.extend(incoming)
    return ExtractionResult(
        second.outcome,
        tuple(pages),
        "\n".join(page.text for page in pages if page.text),
        second.detail,
        first.dropped_pages + second.dropped_pages,
        first.dropped_characters + second.dropped_characters,
        first.dropped_regions + second.dropped_regions,
        second.resume,
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


class SelectedDocumentReader:
    """Read one selected document in full, or raise the workload's refusal.

    Four workloads held this block verbatim — resolve an adapter by suffix,
    refuse `source-unsupported` when there is none, extract in accumulating
    passes, and turn a non-`SUCCEEDED` outcome into `source-truncated` with
    its measured counts or `extraction-failed`. They differed only in which
    exception type they raised. It was edited identically in all four twice in
    one day, which is what a seam looks like before it is named.

    The adapter mapping is held by reference rather than copied: a workload
    exposes the same dictionary, and a caller that adds or removes an adapter
    on the plugin must change what the reader resolves.
    """

    def __init__(
        self,
        adapters: dict[str, ExtractionAdapter],
        error_type: Callable[[str, str], Exception],
        *,
        limits: ExtractionLimits | None = None,
    ) -> None:
        self._adapters = adapters
        self._error_type = error_type
        self._limits = limits

    @property
    def adapters(self) -> dict[str, ExtractionAdapter]:
        return self._adapters

    async def read(self, item) -> ExtractionResult:
        adapter = self._adapters.get(item.suffix)
        if adapter is None:
            raise self._error_type("source-unsupported", "selected source type is unsupported")
        extraction = await DocumentExtractor(adapter, limits=self._limits).extract_all(item)
        if extraction.outcome is ExtractionOutcome.SUCCEEDED:
            return extraction
        # A truncated extraction used to be answered as if whole, so a long
        # selection was answered from its opening pages and nobody was told.
        # Say what went unread instead of inventing an answer from part of it.
        summary = truncation_summary(extraction)
        if summary:
            raise self._error_type(TRUNCATED_SOURCE_CODE, summary)
        raise self._error_type("extraction-failed", "selected source could not be extracted")


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
