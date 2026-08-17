"""Trusted evidence binding and bounded result normalization for generation workloads."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from omnitensor.plugins.fragments import FragmentStoreError, PrivateFragmentStore
from omnitensor.plugins.generation import GenerationRequest, GenerationTask

_SPAN_REFERENCE = re.compile(r":span:(\d+)-(\d+)$")
_TAG_SEPARATOR = re.compile(r"[^a-z0-9]+")


def bind_grounding_metadata(
    raw: str,
    task: GenerationTask,
    request: GenerationRequest,
    store: PrivateFragmentStore,
) -> str:
    """Bind trusted metadata only after the model selects an exact private source."""
    evidence_groups = _model_evidence(task, request)
    if evidence_groups is None:
        return raw
    try:
        document = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError):
        return raw
    evidence_items = evidence_groups(document)
    if not evidence_items:
        return raw
    allowed = set(request.content_references)
    if task.task_id == "ask-selected-files":
        allowed.discard(request.content_references[0])
    elif task.task_id == "file-organizer":
        allowed = {reference for reference in allowed if ":metadata:" not in reference}
    if not all(
        _bind_evidence(evidence, task.task_id, request.request_id, allowed, store)
        for evidence in evidence_items
    ):
        return raw
    _normalize_bound_document(document, task.task_id)
    if task.task_id == "file-organizer":
        _normalize_organizer_name_extensions(document, request, store)
    return json.dumps(document, ensure_ascii=False, separators=(",", ":"))


def _model_evidence(task: GenerationTask, request: GenerationRequest):
    invalid_reference_count = (
        task.task_id == "selected-text-tools" and len(request.content_references) != 2
    ) or (task.task_id == "ask-selected-files" and len(request.content_references) < 2)
    if invalid_reference_count:
        return None
    return {
        "selected-text-tools": _selected_evidence,
        "ask-selected-files": _citation_evidence,
        "event-extraction": _event_evidence,
        "file-organizer": _organizer_evidence,
    }.get(task.task_id)


def _selected_evidence(document: object) -> tuple[object, ...]:
    evidence = document.get("evidence") if isinstance(document, dict) else None
    return (evidence,) if isinstance(evidence, dict) else ()


def _citation_evidence(document: object) -> tuple[object, ...]:
    values = document.get("citations") if isinstance(document, dict) else None
    return tuple(values) if isinstance(values, list) and values else ()


def _event_evidence(document: object) -> tuple[object, ...]:
    events = document.get("events") if isinstance(document, dict) else None
    if not isinstance(events, list) or not events:
        return ()
    groups = []
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get("evidence"), list):
            return ()
        groups.extend(event["evidence"])
    return tuple(groups)


def _organizer_evidence(document: object) -> tuple[object, ...]:
    suggestions = document.get("suggestions") if isinstance(document, dict) else None
    if not isinstance(suggestions, list) or not suggestions:
        return ()
    groups = []
    for suggestion in suggestions:
        if not isinstance(suggestion, dict) or not isinstance(suggestion.get("evidence"), list):
            return ()
        groups.extend(suggestion["evidence"])
    return tuple(groups)


def _normalize_bound_document(document: object, task_id: str) -> None:
    if task_id == "selected-text-tools":
        _normalize_selected_text_tasks(document)
    elif task_id == "ask-selected-files":
        _deduplicate_document_citations(document)
    elif task_id == "file-organizer":
        _merge_organizer_suggestions(document)
        _normalize_organizer_tags(document)


def _normalize_selected_text_tasks(document: object) -> None:
    """Remove a model's non-action placeholders from non-extraction results."""
    if isinstance(document, dict) and document.get("operation") != "extract-tasks":
        document["tasks"] = []


def _merge_organizer_suggestions(document: object) -> None:
    """Merge compatible same-file span suggestions without inventing metadata."""
    suggestions = document.get("suggestions") if isinstance(document, dict) else None
    if not isinstance(suggestions, list) or len(suggestions) < 2:
        return
    merged: list[dict] = []
    changed = False
    for suggestion in suggestions:
        if not isinstance(suggestion, dict):
            return
        if not merged or suggestion.get("fileId") != merged[-1].get("fileId"):
            merged.append(copy.deepcopy(suggestion))
            continue
        previous = merged[-1]
        if any(
            suggestion.get(field) != previous.get(field)
            for field in ("proposedName", "proposedFolder")
        ):
            return
        tags = _unique_strings(previous.get("tags"), suggestion.get("tags"), 16)
        evidence = _unique_evidence(previous.get("evidence"), suggestion.get("evidence"), 8)
        reasons = _unique_strings((previous.get("reason"),), (suggestion.get("reason"),), 2)
        reason = " ".join(reasons) if reasons is not None else ""
        if tags is None or evidence is None or not 1 <= len(reason) <= 2_048:
            return
        previous["tags"] = tags
        previous["evidence"] = evidence
        previous["reason"] = reason
        changed = True
    if changed:
        document["suggestions"] = merged


def _unique_strings(first: object, second: object, maximum: int) -> list[str] | None:
    if not isinstance(first, (list, tuple)) or not isinstance(second, (list, tuple)):
        return None
    values: list[str] = []
    for value in (*first, *second):
        if not isinstance(value, str):
            return None
        if value not in values:
            values.append(value)
    return values if len(values) <= maximum else None


def _unique_evidence(first: object, second: object, maximum: int) -> list[dict] | None:
    if not isinstance(first, list) or not isinstance(second, list):
        return None
    values: list[dict] = []
    references: set[str] = set()
    for value in (*first, *second):
        if not isinstance(value, dict) or not isinstance(value.get("sourceRef"), str):
            return None
        reference = value["sourceRef"]
        if reference not in references:
            references.add(reference)
            values.append(copy.deepcopy(value))
    return values if len(values) <= maximum else None


def _normalize_organizer_tags(document: object) -> None:
    """Project model-selected tag phrases onto the public slug contract."""
    suggestions = document.get("suggestions") if isinstance(document, dict) else None
    if not isinstance(suggestions, list):
        return
    for suggestion in suggestions:
        tags = suggestion.get("tags") if isinstance(suggestion, dict) else None
        if not isinstance(tags, list):
            continue
        normalized = []
        for tag in tags:
            if not isinstance(tag, str):
                continue
            slug = _TAG_SEPARATOR.sub("-", tag.casefold()).strip("-")[:48].rstrip("-")
            if slug and slug not in normalized:
                normalized.append(slug)
            if len(normalized) == 16:
                break
        suggestion["tags"] = normalized


def _normalize_organizer_name_extensions(
    document: object,
    request: GenerationRequest,
    store: PrivateFragmentStore,
) -> None:
    """Complete a safe extensionless suggestion from trusted file metadata."""
    suggestions = document.get("suggestions") if isinstance(document, dict) else None
    if not isinstance(suggestions, list):
        return
    suffixes = _organizer_suffixes(request, store)
    if suffixes is None:
        return
    for suggestion in suggestions:
        if isinstance(suggestion, dict):
            suggestion["proposedName"] = _completed_organizer_name(
                suggestion.get("proposedName"), suffixes.get(suggestion.get("fileId"))
            )


def _organizer_suffixes(
    request: GenerationRequest, store: PrivateFragmentStore
) -> dict[str, str] | None:
    suffixes: dict[str, str] = {}
    for reference in request.content_references:
        if ":metadata:" not in reference:
            continue
        try:
            metadata = json.loads(store.resolve(request.request_id, reference).text)
        except (FragmentStoreError, UnicodeError, json.JSONDecodeError):
            return None
        if not isinstance(metadata, dict) or set(metadata) != {"fileId", "fileName"}:
            return None
        file_id = metadata.get("fileId")
        file_name = metadata.get("fileName")
        if not isinstance(file_id, str) or not isinstance(file_name, str):
            return None
        suffixes[file_id] = Path(file_name).suffix
    return suffixes


def _completed_organizer_name(value: object, suffix: object) -> object:
    if not isinstance(value, str) or not isinstance(suffix, str) or not suffix:
        return value
    safe_extensionless = (
        Path(value).suffix == ""
        and Path(value).name == value
        and not value.startswith(".")
        and len(value) + len(suffix) <= 255
    )
    return value + suffix if safe_extensionless else value


def _deduplicate_document_citations(document: object) -> None:
    """Collapse repeated, already-bound references without selecting evidence."""
    citations = document.get("citations") if isinstance(document, dict) else None
    if not isinstance(citations, list):
        return
    unique = []
    references: set[str] = set()
    for citation in citations:
        reference = citation.get("sourceRef") if isinstance(citation, dict) else None
        if not isinstance(reference, str) or reference in references:
            continue
        references.add(reference)
        unique.append(citation)
    document["citations"] = unique


def _bind_evidence(
    evidence: object,
    task_id: str,
    request_id: str,
    allowed: set[str],
    store: PrivateFragmentStore,
) -> bool:
    if not isinstance(evidence, dict):
        return False
    reference = evidence.get("sourceRef")
    if not isinstance(reference, str) or reference not in allowed:
        return False
    source = store.resolve(request_id, reference)
    span = fragment_span(source)
    if task_id == "selected-text-tools" and evidence.get("span") != span:
        return False
    evidence["sourceSha256"] = source.source_sha256
    if task_id != "selected-text-tools":
        evidence["page"] = source.page
    evidence["span"] = span
    evidence["textSha256"] = source.text_sha256
    return True


def bind_selected_text_digests(
    raw: str,
    task: GenerationTask,
    request: GenerationRequest,
    store: PrivateFragmentStore,
) -> str:
    """Compatibility wrapper for the original selected-text binder."""
    return bind_grounding_metadata(raw, task, request, store)


def fragment_span(source) -> dict[str, int]:
    match = _SPAN_REFERENCE.search(source.reference)
    return (
        {"start": int(match.group(1)), "end": int(match.group(2))}
        if match is not None
        else {"start": 0, "end": len(source.text)}
    )
