from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from omnitensor.plugins.ingestion import (
    MAX_ROOTS,
    IngestedFile,
    IngestionError,
    IngestionRejection,
    OptedInRootScanner,
    RejectedFile,
)


def write(root, relative, content=b"content"):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_only_opted_in_roots_are_walked(tmp_path):
    opted = tmp_path / "opted"
    other = tmp_path / "other"
    write(opted, "a.txt")
    write(other, "b.txt")

    scan = OptedInRootScanner([opted]).scan()

    assert [Path(item.path).name for item in scan.files] == ["a.txt"]


def test_a_symlink_escaping_its_root_is_refused_not_followed(tmp_path):
    opted = tmp_path / "opted"
    secret = write(tmp_path / "elsewhere", "secret.txt", b"private")
    opted.mkdir()
    (opted / "link.txt").symlink_to(secret)

    scan = OptedInRootScanner([opted]).scan()

    assert scan.files == ()
    assert scan.rejected == (
        RejectedFile(
            str(opted / "link.txt"),
            IngestionRejection.OUTSIDE_ROOT,
            "path escapes its opted-in root",
        ),
    )


def test_a_symlink_within_its_root_is_still_not_a_regular_file(tmp_path):
    root = tmp_path / "opted"
    target = write(root, "target.txt")
    link = root / "link.txt"
    link.symlink_to(target)

    scan = OptedInRootScanner([root]).scan()

    assert scan.rejected == (
        RejectedFile(str(link), IngestionRejection.NOT_A_FILE, "not a regular file"),
    )


def test_a_file_beyond_the_size_bound_is_refused_before_it_is_read(tmp_path):
    root = tmp_path / "opted"
    write(root, "big.txt", b"x" * 100)

    scan = OptedInRootScanner([root], max_file_bytes=10).scan()

    assert scan.files == ()
    assert scan.rejected == (
        RejectedFile(
            str(root / "big.txt"),
            IngestionRejection.TOO_LARGE,
            "100 bytes exceeds 10",
        ),
    )


def test_a_file_that_grows_during_digest_maps_to_too_large(tmp_path, monkeypatch):
    import omnitensor.preparation as preparation
    from omnitensor.preparation import FileDigestTooLargeError

    root = tmp_path / "opted"
    source = write(root, "growing.txt", b"12345")

    def grew(path, *, max_bytes):
        assert path == source
        raise FileDigestTooLargeError(max_bytes, max_bytes + 1)

    monkeypatch.setattr(preparation, "file_digest", grew)
    scan = OptedInRootScanner([root], max_file_bytes=5).scan()

    assert scan.files == ()
    assert scan.rejected == (
        RejectedFile(
            str(source),
            IngestionRejection.TOO_LARGE,
            "6 bytes exceeds 5",
        ),
    )


def test_an_unsupported_type_is_refused(tmp_path):
    root = tmp_path / "opted"
    write(root, "a.txt")
    write(root, "b.png")

    scan = OptedInRootScanner([root], suffixes=(".png",)).scan()

    assert [Path(item.path).name for item in scan.files] == ["b.png"]
    assert scan.rejected == (
        RejectedFile(
            str(root / "a.txt"),
            IngestionRejection.UNSUPPORTED_TYPE,
            "unsupported type: .txt",
        ),
    )


def test_depth_is_bounded(tmp_path):
    root = tmp_path / "opted"
    write(root, "a/b/c/d/deep.txt")
    write(root, "shallow.txt")

    scan = OptedInRootScanner([root], max_depth=2).scan()

    assert [Path(item.path).name for item in scan.files] == ["shallow.txt"]


def test_identity_is_the_content_digest_so_a_rename_is_a_rename(tmp_path):
    root = tmp_path / "opted"
    original = write(root, "before.txt", b"same bytes")
    first = OptedInRootScanner([root]).scan()
    original.rename(root / "after.txt")

    second = OptedInRootScanner([root]).scan()

    assert first.files[0].digest == second.files[0].digest
    assert first.files[0].path != second.files[0].path


def test_exact_duplicates_are_grouped_by_digest(tmp_path):
    root = tmp_path / "opted"
    write(root, "one.txt", b"identical")
    write(root, "nested/two.txt", b"identical")
    write(root, "different.txt", b"other")

    duplicates = OptedInRootScanner([root]).scan().duplicates

    assert len(duplicates) == 1
    assert len(next(iter(duplicates.values()))) == 2


def test_a_file_removed_mid_scan_is_reported_and_the_pass_continues(tmp_path, monkeypatch):
    root = tmp_path / "opted"
    write(root, "a.txt")
    write(root, "b.txt")
    from pathlib import Path as RealPath

    original = RealPath.open

    def vanish(self, *args, **kwargs):
        if self.name == "a.txt":
            raise FileNotFoundError(2, "removed mid-scan")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(RealPath, "open", vanish)
    scan = OptedInRootScanner([root]).scan()

    assert [Path(item.path).name for item in scan.files] == ["b.txt"]
    assert scan.rejected == (
        RejectedFile(
            str(root / "a.txt"),
            IngestionRejection.UNREADABLE,
            "[Errno 2] removed mid-scan",
        ),
    )


def test_an_accepted_file_preserves_exact_metadata_and_digest(tmp_path):
    root = tmp_path / "opted"
    source = write(root, "sample.TXT", b"sample")
    os.utime(source, ns=(1_234_000_000, 1_234_000_000))

    scan = OptedInRootScanner([root]).scan()

    assert scan.rejected == ()
    assert scan.files == (
        IngestedFile(
            str(source),
            6,
            1_234,
            ".txt",
            hashlib.sha256(b"sample").hexdigest(),
        ),
    )


def test_the_file_count_is_bounded_and_truncation_is_reported(tmp_path):
    root = tmp_path / "opted"
    for index in range(5):
        write(root, f"file-{index}.txt", f"content-{index}".encode())

    scan = OptedInRootScanner([root], max_files=2).scan()

    assert len(scan.files) == 2
    assert scan.truncated is True


def test_a_complete_scan_is_not_marked_truncated(tmp_path):
    root = tmp_path / "opted"
    write(root, "a.txt")
    assert OptedInRootScanner([root]).scan().truncated is False


def test_a_missing_root_yields_nothing_rather_than_failing(tmp_path):
    assert OptedInRootScanner([tmp_path / "absent"]).scan().files == ()


def test_a_directory_is_not_ingested_as_a_file(tmp_path):
    root = tmp_path / "opted"
    (root / "nested").mkdir(parents=True)
    assert OptedInRootScanner([root]).scan().files == ()


@pytest.mark.parametrize(
    ("roots", "message"),
    [
        ([], "at least one"),
        ("/tmp", "sequence"),
        ([f"/tmp/root-{index}" for index in range(MAX_ROOTS + 1)], "at most"),
    ],
)
def test_roots_are_validated(roots, message):
    with pytest.raises(IngestionError, match=message):
        OptedInRootScanner(roots)


@pytest.mark.parametrize(
    "changes",
    [
        {"max_file_bytes": 0},
        {"max_files": 0},
        {"max_depth": 0},
        {"max_files": True},
    ],
)
def test_bounds_are_validated(tmp_path, changes):
    with pytest.raises(IngestionError, match="must be a positive integer"):
        OptedInRootScanner([tmp_path], **changes)


def test_nothing_here_decodes_or_parses(tmp_path):
    """Decoding untrusted media is where the bombs are; it is not done here."""
    import inspect

    from omnitensor.plugins import ingestion

    source = inspect.getsource(ingestion)
    for forbidden in ("PIL", "Image.open", "zipfile", "tarfile", "xml."):
        assert forbidden not in source
