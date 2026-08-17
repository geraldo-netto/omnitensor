"""Refuse unsafe document containers before anything opens them.

A document is not a file, it is a container, and the dangerous ones are
dangerous before any parser sees them: an archive that expands to terabytes, a
container nested inside itself hundreds of times, a declared type that does not
match the bytes on disk.  Each is cheap to detect from the header and the
declared sizes, and ruinous to detect by attempting the parse.

So this layer answers one question — is this worth handing to a parser — using
the magic bytes, the declared expansion ratio, and the nesting depth, and it
never expands anything to find out.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .ingestion import (
    IngestedFile,
    IngestionRejection,
    IngestionScan,
    OptedInRootScanner,
    positive_integer_bound,
)

DEFAULT_MAX_DOCUMENT_BYTES = 128 * 1024 * 1024
DEFAULT_MAX_EXPANSION_RATIO = 100
DEFAULT_MAX_CONTAINER_DEPTH = 3
MAX_MAGIC_BYTES = 16

# Header signatures, so a declared suffix is never trusted on its own.
_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"%PDF-", "pdf"),
    (b"PK\x03\x04", "zip"),
    (b"\x1f\x8b", "gzip"),
    (b"BZh", "bzip2"),
    (b"\xfd7zXZ", "xz"),
    (b"ustar", "tar"),
    (b"{\\rtf", "rtf"),
)


class DocumentRejection(StrEnum):
    """Why a container was refused before parsing."""

    TYPE_MISMATCH = "declared-type-mismatch"
    EXPANSION_BOMB = "expansion-ratio-exceeded"
    TOO_DEEPLY_NESTED = "container-too-deeply-nested"
    UNREADABLE = "unreadable"


@dataclass(frozen=True, slots=True)
class DocumentCandidate:
    """One file judged safe enough to hand to a parser."""

    file: IngestedFile
    detected_type: str
    container_depth: int


@dataclass(frozen=True, slots=True)
class RefusedDocument:
    path: str
    reason: DocumentRejection
    detail: str


@dataclass(frozen=True, slots=True)
class DocumentScan:
    candidates: tuple[DocumentCandidate, ...]
    refused: tuple[RefusedDocument, ...]
    rejected: tuple[object, ...]
    truncated: bool


class DocumentIngestor:
    """Enumerate opted-in roots and refuse containers that are unsafe to open."""

    def __init__(
        self,
        roots: Sequence[Path | str],
        *,
        suffixes: Sequence[str] = (),
        max_document_bytes: int = DEFAULT_MAX_DOCUMENT_BYTES,
        max_expansion_ratio: int = DEFAULT_MAX_EXPANSION_RATIO,
        max_container_depth: int = DEFAULT_MAX_CONTAINER_DEPTH,
        **scanner_options,
    ) -> None:
        for name, value in (
            ("max_expansion_ratio", max_expansion_ratio),
            ("max_container_depth", max_container_depth),
        ):
            positive_integer_bound(value, name)
        self._scanner = OptedInRootScanner(
            roots,
            suffixes=suffixes,
            max_file_bytes=max_document_bytes,
            **scanner_options,
        )
        self._max_expansion_ratio = max_expansion_ratio
        self._max_container_depth = max_container_depth

    @property
    def roots(self) -> tuple[Path, ...]:
        return self._scanner.roots

    def scan(self) -> DocumentScan:
        base: IngestionScan = self._scanner.scan()
        candidates: list[DocumentCandidate] = []
        refused: list[RefusedDocument] = []
        for item in base.files:
            candidate, refusal = self._judge(item)
            if candidate is not None:
                candidates.append(candidate)
            elif refusal is not None:
                refused.append(refusal)
        return DocumentScan(tuple(candidates), tuple(refused), base.rejected, base.truncated)

    def _judge(self, item: IngestedFile) -> tuple[DocumentCandidate | None, RefusedDocument | None]:
        try:
            with Path(item.path).open("rb") as stream:
                header = stream.read(MAX_MAGIC_BYTES)
        except OSError as error:
            return None, RefusedDocument(item.path, DocumentRejection.UNREADABLE, str(error))
        detected = detect_document_type(header)
        expected = item.suffix.lstrip(".")
        if expected and detected and detected != expected and not _compatible(expected, detected):
            # A .pdf whose bytes are a zip is either a mistake or an attempt to
            # reach a parser that would not otherwise have been chosen.
            return None, RefusedDocument(
                item.path,
                DocumentRejection.TYPE_MISMATCH,
                f"declared {expected}, detected {detected}",
            )
        depth = container_depth(item.path)
        if depth > self._max_container_depth:
            return None, RefusedDocument(
                item.path,
                DocumentRejection.TOO_DEEPLY_NESTED,
                f"nested {depth} deep; limit is {self._max_container_depth}",
            )
        return DocumentCandidate(item, detected or expected or "unknown", depth), None

    def expansion_refusal(
        self, item: IngestedFile, declared_uncompressed_bytes: int
    ) -> RefusedDocument | None:
        """Refuse a container whose declared expansion is implausible.

        The ratio is checked against what the container *claims*, so nothing is
        expanded to find out — which is the entire point, because expanding is
        the attack.
        """
        if item.size_bytes <= 0:
            return None
        ratio = declared_uncompressed_bytes // max(item.size_bytes, 1)
        if ratio > self._max_expansion_ratio:
            return RefusedDocument(
                item.path,
                DocumentRejection.EXPANSION_BOMB,
                f"declared expansion {ratio}x exceeds {self._max_expansion_ratio}x",
            )
        return None


def detect_document_type(header: bytes) -> str:
    """The type the bytes actually are, ignoring what the name claims."""
    for signature, name in _SIGNATURES:
        if header.startswith(signature) or signature in header[:MAX_MAGIC_BYTES]:
            return name
    return ""


def container_depth(path: str) -> int:
    """How many container suffixes are stacked on one name."""
    known = {"zip", "gz", "bz2", "xz", "tar", "tgz", "7z"}
    parts = [part.lower() for part in Path(path).name.split(".")[1:]]
    depth = 0
    for part in reversed(parts):
        if part not in known:
            break
        depth += 1
    return depth


def _compatible(expected: str, detected: str) -> bool:
    # Modern office documents are zip containers, so a zip header under a docx
    # name is the correct shape rather than a mismatch.
    zip_backed = {"docx", "xlsx", "pptx", "odt", "ods", "odp", "epub"}
    if expected in zip_backed and detected == "zip":
        return True
    return {expected, detected} in ({"tgz", "gzip"}, {"tar", "gzip"})


__all__ = [
    "DEFAULT_MAX_CONTAINER_DEPTH",
    "DEFAULT_MAX_DOCUMENT_BYTES",
    "DEFAULT_MAX_EXPANSION_RATIO",
    "DocumentCandidate",
    "DocumentIngestor",
    "DocumentRejection",
    "DocumentScan",
    "IngestionRejection",
    "RefusedDocument",
    "container_depth",
    "detect_document_type",
]
