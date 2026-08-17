from __future__ import annotations

import asyncio
import hashlib
import json
import tempfile
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.plugins import event_workload
from omnitensor.plugins.event_workload import (
    EventExtractionPlugin,
    EventRecoveryJournal,
    EventWorkloadError,
    MemoryFragmentStore,
    PlainTextAdapter,
    event_generation_task,
    grounded_event_document,
    select_sources,
    validate_event_grounding,
)
from omnitensor.plugins.events import EventResultError
from omnitensor.plugins.fragments import SourceFragment
from omnitensor.plugins.generation import (
    ArtifactProvenance,
    GenerationError,
    GenerationProviderDescriptor,
    GenerationRouter,
)
from omnitensor.plugins.ingestion import IngestedFile
from omnitensor.plugins.protocol import (
    PluginContext,
    PluginRequest,
)
from omnitensor.registry import validate_workload_document


class Progress:
    def __init__(self):
        self.items = []

    async def report(self, item):
        self.items.append(item)


class Worker:
    descriptor = GenerationProviderDescriptor(
        "fake-qwen-gpu",
        "gpu",
        "llama.cpp-vulkan",
        True,
        ArtifactProvenance("qwen", "1", "a" * 64, "https://example.invalid/model", "Apache-2.0"),
    )

    def __init__(self, source: Path, *, failure=None):
        self.source = source
        self.failure = failure
        self.requests = []
        self.terminated = []

    async def generate(self, task, request, cancellation, progress):
        self.requests.append(request)
        await progress.report(type("P", (), {"fraction": 0.5})())
        if self.failure:
            raise self.failure
        source_digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
        text_digest = hashlib.sha256(self.source.read_text().encode()).hexdigest()
        return json.dumps(
            {
                "version": 2,
                "requestId": request.request_id,
                "outcome": "succeeded",
                "code": "events-found",
                "detail": "",
                "duplicatePolicy": "keep-first-label-date-time-place",
                "confirmationState": "pending",
                "events": [
                    {
                        "candidateId": "event-1",
                        "label": {"text": "Review", "source": "stated"},
                        "when": {
                            "date": "2026-08-12",
                            "time": "10:00",
                            "timezone": {
                                "name": "Europe/Rome",
                                "utcOffset": "+02:00",
                                "source": "stated",
                            },
                            "needs": [],
                        },
                        "confirmation": "pending",
                        "evidence": [
                            {
                                "sourceRef": request.content_references[0],
                                "sourceSha256": source_digest,
                                "page": 1,
                                "span": {"start": 0, "end": 6},
                                "textSha256": text_digest,
                                "readAs": "text",
                            }
                        ],
                    }
                ],
            }
        )

    async def terminate(self, request_id):
        self.terminated.append(request_id)


def item(path: Path) -> IngestedFile:
    status = path.stat()
    return IngestedFile(
        str(path),
        status.st_size,
        status.st_mtime_ns // 1_000_000,
        path.suffix,
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def request(path: Path) -> PluginRequest:
    return PluginRequest("job-1", "event-extraction", "manual", {"sources": [str(path)]}, 1, None)


async def running_plugin(tmp_path: Path, source: Path, **kwargs):
    worker = Worker(source)
    store = MemoryFragmentStore()
    plugin = EventExtractionPlugin(
        GenerationRouter((worker,)),
        store,
        EventRecoveryJournal(tmp_path / "recovery.json"),
        clock_ms=lambda: 10,
        **kwargs,
    )
    await plugin.start(PluginContext("event-extraction", 1, {}, frozenset({"files:read-selected"})))
    return plugin, worker, store


@pytest.mark.asyncio
async def test_event_worker_start_requires_a_qualified_ready_provider(tmp_path):
    plugin = EventExtractionPlugin(
        GenerationRouter(()),
        MemoryFragmentStore(),
        EventRecoveryJournal(tmp_path / "recovery.json"),
    )

    with pytest.raises(GenerationError) as error:
        await plugin.start(
            PluginContext("event-extraction", 1, {}, frozenset({"files:read-selected"}))
        )

    assert error.value.code == "provider-unavailable"


def test_manifest_is_closed_and_coordinates_manual_private_workload():
    path = Path(__file__).parents[1] / "plugin-manifests/event-extraction.json"
    manifest = json.loads(path.read_text())
    assert validate_workload_document(manifest) == []
    assert manifest["plugin"]["triggers"] == ["manual"]
    assert manifest["plugin"]["permissions"] == [
        "accelerator:gpu",
        "files:read-selected",
    ]
    assert manifest["plugin"]["protocol"]["capabilities"] == [
        "cancel",
        "execute",
        "health",
        "progress",
    ]
    assert manifest["requirements"]["acceleratorPreference"] == ["gpu"]
    # More than one generation model is declared so a person can choose between
    # them; the first is only the default.
    # Each states where it came from and under what licence, because the worker
    # reports provenance for whichever of them a person chose.
    assert manifest["plugin"]["artifacts"] == [
        {
            "id": "qwen3-5-9b-iq4-xs",
            "version": "1.0.0",
            "format": "gguf",
            "sha256": "7e918aeca06c52bcb528ea6b04b4ec957e75ee8c0a73138854c0dfcf371ea429",
            "sourceUri": (
                "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-IQ4_XS.gguf"
            ),
            "licenseSpdx": "Apache-2.0",
        },
        {
            "id": "qwen3-8b-q4-k-m",
            "version": "1.0.0",
            "format": "gguf",
            "sha256": "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785",
            "sourceUri": (
                "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/"
                "7c41481f57cb95916b40956ab2f0b139b296d974/Qwen3-8B-Q4_K_M.gguf"
            ),
            "licenseSpdx": "Apache-2.0",
        },
    ]


def test_task_is_closed_grounded_contract():
    task = event_generation_task()

    assert (task.task_id, task.task_version) == ("event-extraction", 1)
    assert (task.prompt_id, task.prompt_version) == ("grounded-calendar-events", 1)
    assert task.modalities == ("text", "image")
    assert task.output_schema == json.loads(
        (Path(__file__).parents[1] / "schemas/event-extraction-result.schema.json").read_text()
    )
    assert task.output_schema["properties"]["version"] == {"const": 2}
    # No ceiling on the answer: a programme with a hundred talks is a hundred
    # events, and the context after the prompt is the only real bound.
    assert (task.limits.output_tokens, task.limits.output_bytes) == (None, None)
    assert task.limits.context_tokens == 32_768

    # The prompt is not pinned word for word here: the receipt-versus-task test
    # already fails loudly when it changes, which is the check that matters,
    # and a verbatim copy of 1,800 characters only ever gets pasted over. What
    # is pinned is what the measurements showed was missing — the success path
    # stated as precisely as the refusal, and a refusal that means "no event"
    # rather than "no complete event".
    assert "never set any confirmation to confirmed" in task.system_prompt
    assert "a missing time, place or timezone is reported, not filled in" in task.system_prompt
    for rule in (
        "needs lists exactly what is missing from date, time and timezone",
        "is still extracted — it is a question for the person",
        "only when the sources contain no event",
        "readAs text, or ocr when the fragment came from a scanned page",
        "{{UNTRUSTED_CONTENT}}",
    ):
        assert rule in task.instruction_template


def test_recovery_and_active_fail_closed_on_corrupt_or_oversized_state(tmp_path):
    path = tmp_path / "recovery.json"
    journal = EventRecoveryJournal(path)
    assert journal.recover() == ()
    path.write_text("{")
    assert journal.recover() == ()
    path.write_text(json.dumps({"active": {"ok": "select", "bad/id": "private"}}))
    assert journal.recover() == ("ok",)
    path.write_bytes(b"x" * (256 * 1024 + 1))
    journal.finish("job")
    assert json.loads(path.read_text()) == {"active": {}, "version": 1}
    path.write_text('{"active":{"one":"select"}}')
    assert journal._active() == {"one": "select"}
    raw = b'{"active":{"exact":"generate"}}'
    path.write_bytes(raw + (b" " * (event_workload.MAX_JOURNAL_BYTES - len(raw))))
    assert journal._active() == {"exact": "generate"}
    assert (tmp_path / ".recovery.json.lock").is_file()


def test_recovery_serializes_concurrent_read_modify_write(tmp_path, monkeypatch):
    journal = EventRecoveryJournal(tmp_path / "recovery.json")
    held = []
    original = event_workload.store_lock

    def observed(root, name):
        held.append((root, name))
        return original(root, name)

    monkeypatch.setattr(event_workload, "store_lock", observed)
    journal.stage("one", "select")
    journal.stage("two", "generate")
    journal.finish("one")
    assert journal._active() == {"two": "generate"}
    assert held == [
        (tmp_path, ".recovery.json.lock"),
        (tmp_path, ".recovery.json.lock"),
        (tmp_path, ".recovery.json.lock"),
    ]


@pytest.mark.parametrize(
    ("sources", "code"),
    [
        (None, "sources-invalid"),
        ("/tmp/x.txt", "sources-invalid"),
        ([], "sources-invalid"),
        (["relative.txt"], "source-invalid"),
    ],
)
def test_selected_source_shape_is_strict(sources, code):
    with pytest.raises(EventWorkloadError) as error:
        select_sources("job-1", sources)
    assert error.value.code == code


def test_selected_source_errors_are_stable_and_exact(tmp_path, monkeypatch):
    source = tmp_path / "exact.txt"
    source.write_bytes(b"12345")
    monkeypatch.setattr(event_workload, "MAX_SOURCE_BYTES", 5)
    exact = select_sources("r" * 120, [str(source)])[0]
    assert exact.item.size_bytes == 5
    assert exact.item.modified_at_ms == source.stat().st_mtime_ns // 1_000_000
    source.write_bytes(b"123456")
    with pytest.raises(EventWorkloadError) as large:
        select_sources("job", [str(source)])
    assert (large.value.code, large.value.detail) == (
        "source-invalid",
        "selected source size is outside bounds",
    )
    with pytest.raises(EventWorkloadError) as request_id:
        select_sources("r" * 121, [str(source)])
    assert str(request_id.value) == "request-invalid: request id is invalid"
    for raw, expected in (
        (None, "source-invalid: selected source path is invalid"),
        (1, "source-invalid: selected source path is invalid"),
        ("", "source-invalid: selected source path is invalid"),
        ("bad\0path", "source-invalid: selected source path is invalid"),
        ("relative.txt", "source-invalid: selected source path must be absolute"),
    ):
        with pytest.raises(EventWorkloadError) as invalid:
            event_workload._selected_source(raw)
        assert str(invalid.value) == expected

    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(EventWorkloadError) as not_file:
        event_workload._selected_source(str(directory))
    assert str(not_file.value) == "source-invalid: selected source is not a regular file"
    unsupported = tmp_path / "file.bin"
    unsupported.write_bytes(b"one")
    with pytest.raises(EventWorkloadError) as wrong_type:
        event_workload._selected_source(str(unsupported))
    assert str(wrong_type.value) == ("source-unsupported: selected source type is unsupported")

    original = Path.lstat

    def unreadable(path):
        if path == source:
            raise OSError("private")
        return original(path)

    monkeypatch.setattr(Path, "lstat", unreadable)
    with pytest.raises(EventWorkloadError) as unreadable_error:
        event_workload._selected_source(str(source))
    assert str(unreadable_error.value) == ("source-unreadable: selected source cannot be read")


def test_selection_rejects_directories_symlinks_types_empty_and_duplicates(tmp_path):
    text = tmp_path / "one.txt"
    text.write_text("event", encoding="utf-8")
    link = tmp_path / "link.txt"
    link.symlink_to(text)
    empty = tmp_path / "empty.txt"
    empty.touch()
    unsupported = tmp_path / "source.exe"
    unsupported.write_text("event", encoding="utf-8")

    cases = [
        ([str(tmp_path)], "source-invalid"),
        ([str(link)], "source-invalid"),
        ([str(empty)], "source-invalid"),
        ([str(unsupported)], "source-unsupported"),
        ([str(text), str(text)], "source-duplicate"),
    ]
    for sources, code in cases:
        with pytest.raises(EventWorkloadError) as error:
            select_sources("job-1", sources)
        assert error.value.code == code


@given(st.text(min_size=1, max_size=200).filter(lambda value: "\x00" not in value))
def test_text_selection_property_preserves_digest_and_never_content_in_reference(value):
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "chosen.txt"
        source.write_text(value, encoding="utf-8")
        selected = select_sources("property-job", [str(source)])[0]
        assert selected.item.digest == hashlib.sha256(value.encode()).hexdigest()
        assert selected.reference == "private:property-job:source:1"
        assert str(source) not in selected.reference


def test_adapter_mapping_and_journal_stage_are_strict(tmp_path):
    source = tmp_path / "event.txt"
    source.write_text("event", encoding="utf-8")
    with pytest.raises(EventWorkloadError) as adapter:
        EventExtractionPlugin(
            GenerationRouter((Worker(source),)),
            MemoryFragmentStore(),
            EventRecoveryJournal(tmp_path / "j.json"),
            adapters={".bad": PlainTextAdapter()},
        )
    assert str(adapter.value) == "adapter-invalid: source adapter mapping is invalid"
    replacement = PlainTextAdapter()
    configured = EventExtractionPlugin(
        GenerationRouter((Worker(source),)),
        MemoryFragmentStore(),
        EventRecoveryJournal(tmp_path / "configured.json"),
        adapters={".pdf": replacement},
    )
    assert configured._adapters[".pdf"] is replacement
    with pytest.raises(EventWorkloadError) as stage:
        EventRecoveryJournal(tmp_path / "j.json").stage("job", "private-content")
    assert stage.value.code == "stage-invalid"


def test_exact_file_reader_detects_replacement_and_bounds(tmp_path, monkeypatch):
    source = tmp_path / "source.txt"
    source.write_bytes(b"event")
    assert event_workload._read_exact_file(source, 5) == b"event"
    with pytest.raises(EventWorkloadError) as changed:
        event_workload._read_exact_file(source, 4)
    assert str(changed.value) == "source-changed: selected source changed during extraction"
    monkeypatch.setattr(event_workload, "MAX_SOURCE_BYTES", 4)
    with pytest.raises(EventWorkloadError):
        event_workload._read_exact_file(source, 5)

    class Stream:
        def __init__(self):
            self.sizes = []

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self, size):
            self.sizes.append(size)
            return b"event"

    stream = Stream()
    fake_path = type("FakePath", (), {"open": lambda self, mode: stream})()
    with pytest.raises(EventWorkloadError):
        event_workload._read_exact_file(fake_path, 5)
    assert stream.sizes == [5]


def test_serialized_grounded_result_is_defensive(valid_event_document):
    parsed = __import__(
        "omnitensor.plugins.events", fromlist=["parse_grounded_event_result"]
    ).parse_grounded_event_result(valid_event_document)
    serialized = grounded_event_document(parsed)
    assert serialized == valid_event_document
    # The document handed out is a copy: editing it must not reach back into
    # the parsed result the worker still holds.
    serialized["events"][0]["label"]["text"] = "changed"
    assert parsed.events[0].label.text != "changed"


def test_serialized_grounded_result_fails_closed_on_public_contract_drift(
    valid_event_document, monkeypatch
):
    parsed = __import__(
        "omnitensor.plugins.events", fromlist=["parse_grounded_event_result"]
    ).parse_grounded_event_result(valid_event_document)
    monkeypatch.setattr(
        event_workload,
        "validate_document",
        lambda _schema, _document: ["events/0/title: too long"],
    )

    with pytest.raises(EventResultError) as error:
        grounded_event_document(parsed)

    assert (error.value.code, error.value.detail) == (
        "result-invalid",
        "events/0/title: too long",
    )


def test_grounding_requires_exact_request_fragment_digest_page_and_span(valid_event_document):
    parsed = __import__(
        "omnitensor.plugins.events", fromlist=["parse_grounded_event_result"]
    ).parse_grounded_event_result(valid_event_document)
    fragment = SourceFragment(
        "private:request-1:source:1:page:1",
        "a" * 64,
        1,
        "Review",
        "b" * 64,
    )
    validate_event_grounding(parsed, "request-1", {fragment.reference: fragment})

    for field, replacement, detail in (
        ("sourceSha256", "c" * 64, "evidence disagrees with its source"),
        ("page", 2, "evidence disagrees with its source"),
        ("textSha256", "c" * 64, "evidence disagrees with its source"),
    ):
        changed = json.loads(json.dumps(valid_event_document))
        changed["events"][0]["evidence"][0][field] = replacement
        candidate = __import__(
            "omnitensor.plugins.events", fromlist=["parse_grounded_event_result"]
        ).parse_grounded_event_result(changed)
        with pytest.raises(EventResultError) as error:
            validate_event_grounding(candidate, "request-1", {fragment.reference: fragment})
        assert error.value.detail == detail

    too_long = json.loads(json.dumps(valid_event_document))
    too_long["events"][0]["evidence"][0]["span"]["end"] = 7
    candidate = __import__(
        "omnitensor.plugins.events", fromlist=["parse_grounded_event_result"]
    ).parse_grounded_event_result(too_long)
    with pytest.raises(EventResultError, match="evidence disagrees"):
        validate_event_grounding(candidate, "request-1", {fragment.reference: fragment})


def test_grounding_rejects_wrong_request_missing_source_and_invalid_map(valid_event_document):
    parsed = __import__(
        "omnitensor.plugins.events", fromlist=["parse_grounded_event_result"]
    ).parse_grounded_event_result(valid_event_document)
    with pytest.raises(EventResultError) as wrong_request:
        validate_event_grounding(parsed, "other", {})
    assert wrong_request.value.detail == "result request id does not match"
    with pytest.raises(EventResultError) as missing:
        validate_event_grounding(parsed, "request-1", {})
    assert missing.value.detail == "evidence source is unavailable"
    with pytest.raises(EventResultError) as invalid_map:
        validate_event_grounding(parsed, "request-1", {"private:x": object()})
    assert invalid_map.value.detail == "private fragment map is invalid"


@pytest.fixture
def valid_event_document():
    return {
        "version": 2,
        "requestId": "request-1",
        "outcome": "succeeded",
        "code": "events-found",
        "detail": "",
        "duplicatePolicy": "keep-first-label-date-time-place",
        "confirmationState": "pending",
        "events": [
            {
                "candidateId": "event-1",
                "label": {"text": "Review", "source": "stated"},
                # The canonical form a round trip produces: `needs`, `allDay`
                # and `isPast` are always written, because a reader should not
                # have to know that their absence meant anything.
                "when": {
                    "needs": [],
                    "allDay": False,
                    "isPast": False,
                    "date": "2026-08-12",
                    "time": "10:00",
                    "timezone": {"name": "Europe/Rome", "utcOffset": "+02:00", "source": "stated"},
                },
                "confirmation": "pending",
                "status": "scheduled",
                "evidence": [
                    {
                        "sourceRef": "private:request-1:source:1:page:1",
                        "sourceSha256": "a" * 64,
                        "page": 1,
                        "span": {"start": 0, "end": 6},
                        "textSha256": "b" * 64,
                        "readAs": "text",
                    }
                ],
            }
        ],
    }
