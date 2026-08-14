from __future__ import annotations

from pathlib import Path

import pytest

from omnitensor.plugins.document_ingestion import (
    DocumentIngestor,
    DocumentRejection,
    container_depth,
    detect_document_type,
)
from omnitensor.plugins.ingestion import IngestedFile, IngestionRejection


def write(root, relative, content=b"%PDF-1.7 body"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_a_valid_document_is_offered_to_a_parser(tmp_path):
    root = tmp_path / "docs"
    write(root, "report.pdf")

    scan = DocumentIngestor([root]).scan()

    assert [Path(item.file.path).name for item in scan.candidates] == ["report.pdf"]
    assert scan.candidates[0].detected_type == "pdf"


def test_a_declared_type_that_the_bytes_contradict_is_refused(tmp_path):
    """A .pdf whose bytes are a zip is reaching for a parser nobody chose."""
    root = tmp_path / "docs"
    write(root, "invoice.pdf", b"PK\x03\x04 not a pdf at all")

    scan = DocumentIngestor([root]).scan()

    assert scan.candidates == ()
    assert scan.refused[0].reason is DocumentRejection.TYPE_MISMATCH
    assert "declared pdf, detected zip" in scan.refused[0].detail


def test_a_zip_backed_office_document_is_not_a_mismatch(tmp_path):
    root = tmp_path / "docs"
    write(root, "notes.docx", b"PK\x03\x04 office")

    scan = DocumentIngestor([root]).scan()

    assert [Path(item.file.path).name for item in scan.candidates] == ["notes.docx"]


def test_a_deeply_nested_container_is_refused(tmp_path):
    root = tmp_path / "docs"
    write(root, "bundle.tar.gz.xz.zip", b"PK\x03\x04")

    scan = DocumentIngestor([root], max_container_depth=2).scan()

    assert scan.candidates == ()
    assert scan.refused[0].reason is DocumentRejection.TOO_DEEPLY_NESTED


def test_an_expansion_bomb_is_refused_from_what_it_claims_not_by_expanding():
    """Expanding to find out is the attack, so the claim is what is checked."""
    ingestor = DocumentIngestor(["/tmp"], max_expansion_ratio=10)
    item = IngestedFile("/tmp/bomb.zip", 1_000, 0, ".zip", "a" * 64)

    refusal = ingestor.expansion_refusal(item, declared_uncompressed_bytes=1_000_000)

    assert refusal is not None
    assert refusal.reason is DocumentRejection.EXPANSION_BOMB
    assert "1000x exceeds 10x" in refusal.detail


def test_a_plausible_expansion_ratio_is_accepted():
    ingestor = DocumentIngestor(["/tmp"], max_expansion_ratio=10)
    item = IngestedFile("/tmp/normal.zip", 1_000, 0, ".zip", "a" * 64)
    assert ingestor.expansion_refusal(item, declared_uncompressed_bytes=5_000) is None


def test_an_empty_file_has_no_expansion_ratio():
    ingestor = DocumentIngestor(["/tmp"])
    item = IngestedFile("/tmp/empty.zip", 0, 0, ".zip", "a" * 64)
    assert ingestor.expansion_refusal(item, declared_uncompressed_bytes=1) is None


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        (b"%PDF-1.4", "pdf"),
        (b"PK\x03\x04", "zip"),
        (b"\x1f\x8b\x08", "gzip"),
        (b"BZh9", "bzip2"),
        (b"\xfd7zXZ\x00", "xz"),
        (b"nothing recognisable", ""),
    ],
)
def test_types_are_detected_from_bytes(header, expected):
    assert detect_document_type(header) == expected


@pytest.mark.parametrize(
    ("name", "depth"),
    [
        ("report.pdf", 0),
        ("bundle.tar.gz", 2),
        ("archive.zip", 1),
        ("deep.tar.gz.xz", 3),
        ("notes.txt", 0),
    ],
)
def test_container_depth_counts_stacked_suffixes(name, depth):
    assert container_depth(f"/tmp/{name}") == depth


def test_the_size_bound_is_inherited_from_the_scanner(tmp_path):
    root = tmp_path / "docs"
    write(root, "big.pdf", b"%PDF-" + b"x" * 200)

    scan = DocumentIngestor([root], max_document_bytes=10).scan()

    assert scan.candidates == ()
    assert scan.rejected


def test_only_opted_in_roots_are_enumerated(tmp_path):
    opted = tmp_path / "opted"
    write(opted, "a.pdf")
    write(tmp_path / "other", "b.pdf")

    scan = DocumentIngestor([opted]).scan()

    assert len(scan.candidates) == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"max_expansion_ratio": 0},
        {"max_container_depth": 0},
        {"max_expansion_ratio": True},
    ],
)
def test_container_bounds_are_validated(tmp_path, changes):
    with pytest.raises(ValueError, match="must be a positive integer"):
        DocumentIngestor([tmp_path], **changes)


def test_document_bounds_keep_their_order_before_scanner_validation():
    with pytest.raises(ValueError) as expansion_error:
        DocumentIngestor(
            [],
            max_expansion_ratio=0,
            max_container_depth=0,
            max_document_bytes=0,
        )
    assert type(expansion_error.value) is ValueError
    assert str(expansion_error.value) == "max_expansion_ratio must be a positive integer"

    with pytest.raises(ValueError) as depth_error:
        DocumentIngestor([], max_container_depth=0, max_document_bytes=0)
    assert type(depth_error.value) is ValueError
    assert str(depth_error.value) == "max_container_depth must be a positive integer"


def test_an_unreadable_document_is_reported_rather_than_raising(tmp_path, monkeypatch):
    """The scanner reaches it first, so the refusal surfaces there."""
    root = tmp_path / "docs"
    write(root, "a.pdf")
    original = Path.open

    def explode(self, *args, **kwargs):
        if self.suffix == ".pdf":
            raise OSError("device error")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", explode)
    scan = DocumentIngestor([root]).scan()

    assert scan.candidates == ()
    assert scan.rejected[0].reason is IngestionRejection.UNREADABLE


def test_a_header_that_cannot_be_read_is_refused(tmp_path, monkeypatch):
    root = tmp_path / "docs"
    write(root, "a.pdf")
    ingestor = DocumentIngestor([root])
    item = ingestor._scanner.scan().files[0]

    def explode(self, *args, **kwargs):
        raise OSError("device error")

    monkeypatch.setattr(Path, "open", explode)
    _candidate, refusal = ingestor._judge(item)

    assert refusal.reason is DocumentRejection.UNREADABLE
