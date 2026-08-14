"""Policy-driven bounded raw JSONL intake with exact-byte digests."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from ..contracts import TrainingError

Item = TypeVar("Item")


@dataclass(frozen=True, slots=True)
class JsonlPolicy:
    max_bytes: int
    max_lines: int
    max_line_bytes: int
    unsafe_code: str
    unsafe_detail: str
    too_large_code: str
    bytes_detail: str
    lines_detail: str
    invalid_code: str
    line_detail: str
    context_detail: str


def parse_jsonl_line(
    raw: bytes,
    line_number: int,
    policy: JsonlPolicy,
    parse_document: Callable[[object], Item],
) -> Item:
    """Parse one bounded line and preserve family error context."""
    if not raw.strip() or len(raw) > policy.max_line_bytes:
        raise TrainingError(
            policy.invalid_code,
            policy.line_detail.format(line_number=line_number),
        )
    try:
        return parse_document(json.loads(raw))
    except (UnicodeDecodeError, json.JSONDecodeError, TrainingError) as error:
        detail = error.detail if isinstance(error, TrainingError) else "invalid JSON"
        raise TrainingError(
            policy.invalid_code,
            policy.context_detail.format(line_number=line_number, detail=detail),
        ) from error


def load_bounded_jsonl(
    path: Path | str,
    policy: JsonlPolicy,
    parse_document: Callable[[object], Item],
    *,
    on_item: Callable[[Item], None] | None = None,
) -> tuple[tuple[Item, ...], str]:
    """Load bounded JSONL and hash its original bytes including line endings."""
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise TrainingError(policy.unsafe_code, policy.unsafe_detail)
    if source.stat().st_size > policy.max_bytes:
        raise TrainingError(policy.too_large_code, policy.bytes_detail)
    digest = hashlib.sha256()
    items = []
    with source.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            digest.update(raw)
            if line_number > policy.max_lines:
                raise TrainingError(policy.too_large_code, policy.lines_detail)
            item = parse_jsonl_line(raw, line_number, policy, parse_document)
            if on_item is not None:
                on_item(item)
            items.append(item)
    return tuple(items), digest.hexdigest()
