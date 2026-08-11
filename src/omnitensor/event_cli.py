"""Preview/export and legacy-compatible client tools for grounded events."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from .atomicio import JsonTooLargeError, read_json_bounded
from .plugins.event_workload import EventWorkloadError, confirmed_ics, select_sources
from .plugins.events import EventResultError, parse_grounded_event_result
from .plugins.qwen import QwenProviderError, load_qwen_catalog
from .preparation import file_digest

MAX_RESULT_BYTES = 1024 * 1024


class EventClientError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def preview_document(document: object) -> dict:
    """Return editable preview metadata without source paths or source text."""
    result = parse_grounded_event_result(document)
    return {
        "requestId": result.request_id,
        "outcome": result.outcome,
        "confirmationState": result.confirmation_state,
        "duplicatesDropped": result.duplicates_dropped,
        "events": [
            {
                "candidateId": event.candidate_id,
                "title": event.title,
                "start": event.start.isoformat(),
                "end": None if event.end is None else event.end.isoformat(),
                "timezone": event.timezone,
                "location": event.location,
                "evidenceCount": len(event.evidence),
            }
            for event in result.events
        ],
    }


def verify_event_artifacts(root: Path) -> dict:
    """Verify the pinned Qwen files already placed in a private artifact root."""
    catalog = load_qwen_catalog()
    checked = []
    for source in catalog.sources:
        path = Path(root) / source.filename
        ready = path.is_file() and path.stat().st_size == source.size_bytes
        ready = ready and file_digest(path) == source.sha256
        checked.append(
            {
                "role": source.role,
                "filename": source.filename,
                "sha256": source.sha256,
                "ready": ready,
            }
        )
    return {
        "modelId": catalog.model_id,
        "version": catalog.version,
        "license": catalog.license_spdx,
        "ready": all(item["ready"] for item in checked),
        "artifacts": checked,
    }


def legacy_selected_files(directory: Path, *, recursive: bool = False) -> tuple[str, ...]:
    """Compatibility enumeration only after a user explicitly names a directory."""
    root = Path(directory)
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        raise EventClientError("directory-invalid", "input must be an existing absolute directory")
    iterator = root.rglob("*") if recursive else root.glob("*")
    paths = tuple(
        str(path)
        for path in sorted(iterator)
        if path.is_file() and not path.is_symlink()
    )
    # Reuse the workload's exact type, count, file, and size boundary.
    return tuple(source.item.path for source in select_sources("legacy-import", paths))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnitensor-import-events",
        description="Preview or explicitly export a private grounded event result.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    preview = commands.add_parser("preview", help="show pending candidates")
    preview.add_argument("result", type=Path)
    export = commands.add_parser("export", help="write confirmed candidates as ICS")
    export.add_argument("result", type=Path)
    export.add_argument("--confirm", action="append", default=[])
    export.add_argument("--reject", action="append", default=[])
    export.add_argument("--output", type=Path, required=True)
    setup = commands.add_parser("check-model", help="verify pinned local Qwen artifacts")
    setup.add_argument("artifact_root", type=Path)
    legacy = commands.add_parser("legacy-files", help="list an explicitly selected legacy folder")
    legacy.add_argument("directory", type=Path)
    legacy.add_argument("--recursive", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "check-model":
            document = verify_event_artifacts(args.artifact_root)
            _print_json(document)
            return 0 if document["ready"] else 1
        if args.command == "legacy-files":
            _print_json(
                {"sources": legacy_selected_files(args.directory, recursive=args.recursive)}
            )
            return 0
        document = _read_result(args.result)
        if args.command == "preview":
            _print_json(preview_document(document))
            return 0
        rendered = confirmed_ics(document, args.confirm, args.reject)
        _write_new(args.output, rendered.encode("utf-8"))
        _print_json({"written": True, "output": str(args.output), "confirmed": len(args.confirm)})
        return 0
    except (
        EventClientError,
        EventResultError,
        EventWorkloadError,
        QwenProviderError,
        OSError,
        ValueError,
    ) as error:
        code = getattr(error, "code", "client-failed")
        print(json.dumps({"ok": False, "code": str(code)}, sort_keys=True), file=sys.stderr)
        return 2


def _read_result(path: Path) -> dict:
    try:
        document = read_json_bounded(Path(path), MAX_RESULT_BYTES)
    except JsonTooLargeError as error:
        raise EventClientError("result-too-large", "result exceeds its byte limit") from error
    except (OSError, UnicodeError, ValueError) as error:
        raise EventClientError("result-unreadable", "result is not readable JSON") from error
    if not isinstance(document, dict):
        raise EventClientError("result-invalid", "result root must be an object")
    return document


def _write_new(path: Path, content: bytes) -> None:
    target = Path(path)
    if not target.is_absolute():
        raise EventClientError("output-invalid", "output path must be absolute")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def _print_json(document: object) -> None:
    print(json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
