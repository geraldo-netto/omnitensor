"""Trusted evidence binding and bounded result normalization for generation workloads."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
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
    binding = TASK_BINDINGS.get(task.task_id)
    if binding is None or not binding.accepts(request.content_references):
        return raw
    try:
        document = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError):
        return raw
    evidence_items = binding.evidence(document)
    if not evidence_items:
        return raw
    allowed = set(binding.bindable(request.content_references))
    if not all(
        _bind_evidence(evidence, binding, request.request_id, allowed, store)
        for evidence in evidence_items
    ):
        return raw
    for normalize in binding.normalize:
        normalize(document)
    if binding.finalize is not None:
        binding.finalize(document, request, store)
    return json.dumps(document, ensure_ascii=False, separators=(",", ":"))


def citable_references(task_id: str, references: tuple[str, ...]) -> tuple[str, ...]:
    """The opaque references this task may be told to cite, and no others.

    Narrower than what may be *bound*: a selected-text control fragment is a
    legitimate binding target but is never something the model should cite.
    """
    binding = TASK_BINDINGS.get(task_id)
    return () if binding is None else binding.citable(references)


def _every_reference(references: tuple[str, ...]) -> tuple[str, ...]:
    return references


def _no_reference(references: tuple[str, ...]) -> tuple[str, ...]:
    return ()


def _after_the_control_fragment(references: tuple[str, ...]) -> tuple[str, ...]:
    return references[1:]


def _content_references(references: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(reference for reference in references if ":metadata:" not in reference)


@dataclass(frozen=True, slots=True)
class TaskBinding:
    """One workload's grounded-output policy, declared once.

    Adding a workload used to mean editing four dispatch tables in this module
    and three more beside the loader; it now means adding one entry here and,
    if the task says anything extra to the model, one in :mod:`hints`.
    """

    evidence: Callable[[object], tuple[object, ...]]
    exact_references: int | None = None
    minimum_references: int = 0
    citable: Callable[[tuple[str, ...]], tuple[str, ...]] = _no_reference
    bindable: Callable[[tuple[str, ...]], tuple[str, ...]] = _every_reference
    normalize: tuple[Callable[[object], None], ...] = ()
    finalize: Callable[[object, GenerationRequest, PrivateFragmentStore], None] | None = None
    span_must_match: bool = False
    binds_page: bool = True

    def accepts(self, references: tuple[str, ...]) -> bool:
        """Whether this request carries the fragments the task is defined for."""
        if self.exact_references is not None and len(references) != self.exact_references:
            return False
        return len(references) >= self.minimum_references


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
    binding: TaskBinding,
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
    if binding.span_must_match and evidence.get("span") != span:
        return False
    evidence["sourceSha256"] = source.source_sha256
    if binding.binds_page:
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


TASK_BINDINGS: dict[str, TaskBinding] = {
    "selected-text-tools": TaskBinding(
        evidence=_selected_evidence,
        exact_references=2,
        citable=_after_the_control_fragment,
        normalize=(_normalize_selected_text_tasks,),
        span_must_match=True,
        binds_page=False,
    ),
    "ask-selected-files": TaskBinding(
        evidence=_citation_evidence,
        minimum_references=2,
        citable=_after_the_control_fragment,
        bindable=_after_the_control_fragment,
        normalize=(_deduplicate_document_citations,),
    ),
    "event-extraction": TaskBinding(evidence=_event_evidence),
    "file-organizer": TaskBinding(
        evidence=_organizer_evidence,
        citable=_content_references,
        bindable=_content_references,
        normalize=(_merge_organizer_suggestions, _normalize_organizer_tags),
        finalize=_normalize_organizer_name_extensions,
    ),
}
