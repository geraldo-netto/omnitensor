"""The safe way to read an OOXML or ODF container, for whoever is reading one.

A `.pptx`, an `.odp` and a `.docx` are the same thing — a zip of XML — and the
rules for opening one somebody selected are the same too: an entry that
escapes the archive, an archive that expands to more than it declares, an
encrypted member, a symlink, a duplicate name, an XML parser that resolves
external entities. Those guards were written once for presentations, and a
Word document would have had to either copy them or borrow them and tell the
person their *presentation* was invalid.

So the guards moved here and the caller says what it is reading. The only
thing that differs between the two is the word in the refusal, which is the
part a person actually reads.
"""

from __future__ import annotations

import zipfile
from collections.abc import Mapping
from pathlib import PurePosixPath

from omnitensor.plugins.media_transcription import MediaTranscriptionError

MAX_ARCHIVE_ENTRIES = 4096
MAX_ARCHIVE_EXPANDED_BYTES = 256 * 1024 * 1024
MAX_PART_XML_BYTES = 4 * 1024 * 1024


def archive_entries(archive: zipfile.ZipFile, kind: str) -> dict[str, zipfile.ZipInfo]:
    """Every member of the archive, refusing the ones that are not safe to read."""
    infos = archive.infolist()
    if len(infos) > MAX_ARCHIVE_ENTRIES:
        raise MediaTranscriptionError(f"{kind}-invalid", f"{kind} archive has too many entries")
    entries: dict[str, zipfile.ZipInfo] = {}
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
            raise MediaTranscriptionError(f"{kind}-invalid", f"{kind} archive entry is unsafe")
        expanded += info.file_size
        if expanded > MAX_ARCHIVE_EXPANDED_BYTES:
            raise MediaTranscriptionError(
                f"{kind}-invalid", f"{kind} archive expands beyond its limit"
            )
        entries[info.filename] = info
    return entries


def read_archive_entry(
    archive: zipfile.ZipFile,
    entries: Mapping[str, zipfile.ZipInfo],
    name: str,
    maximum: int,
    kind: str,
) -> bytes:
    info = entries.get(name)
    if info is None or info.is_dir() or not 0 <= info.file_size <= maximum:
        raise MediaTranscriptionError(f"{kind}-invalid", f"{kind} archive entry is unavailable")
    content = archive.read(info)
    if len(content) != info.file_size:
        raise MediaTranscriptionError(f"{kind}-invalid", f"{kind} archive entry was truncated")
    return content


def xml_root(content: bytes, kind: str):
    """Parse container XML with the hardened parser, or say which failed.

    The import is checked separately from the parse: `defusedxml` is a
    declared dependency of this distribution, so its absence is a broken
    install rather than a fact about somebody's file — and reporting it as one
    sends them off to re-export a document that was fine.
    """
    try:
        from defusedxml import ElementTree  # noqa: PLC0415
    except ImportError as error:
        raise MediaTranscriptionError(
            f"{kind}-runtime-unavailable",
            "install the defusedxml container parser",
        ) from error
    try:
        return ElementTree.fromstring(content)
    except Exception as error:
        raise MediaTranscriptionError(f"{kind}-invalid", f"{kind} XML is invalid") from error


__all__ = [
    "MAX_ARCHIVE_ENTRIES",
    "MAX_ARCHIVE_EXPANDED_BYTES",
    "MAX_PART_XML_BYTES",
    "archive_entries",
    "read_archive_entry",
    "xml_root",
]
