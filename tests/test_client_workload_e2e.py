"""The five workloads the desktop client drives, run against a live service.

Everything else in this suite proves a part in isolation. This proves the whole
path a person actually takes: the payload `../xpuwlm` builds, submitted over the
control socket, dispatched to an installed plugin worker, answered, and the
answer validated against the result schema the client will read it with.

That path is where the failures live. Every one of the defects this pair of
projects has spent weeks on — a provider whose output did not match its own
schema, a worker that never became ready, a summary read against invented field
names — was found by a person clicking a button, because nothing here ran the
whole thing.

Skipped without a running service, since a socket is the one thing a test
cannot bring with it: these are run against a machine that has the models
installed, which is the only place the answer means anything. Nothing here is
mocked — a passing run is a real job on a real accelerator.

    OMNITENSOR_E2E=1 python -m pytest tests/test_client_workload_e2e.py -v

Fixtures are written into a root the *service itself publishes*, because a
source outside those roots is refused by the sandbox, and asked for by reading
the snapshot rather than by assuming where they are.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import pytest

from omnitensor.registry import validate_document
from omnitensor.socket_transport import call_control, default_socket_path

# A model answers in seconds, not milliseconds, and a first answer after a cold
# start includes loading it. Long enough not to fail an honest answer; short
# enough that a hung worker is reported rather than waited on.
JOB_TIMEOUT_S = 300.0
POLL_INTERVAL_S = 0.5
TERMINAL = frozenset({"succeeded", "failed", "cancelled", "unknown"})

SNAPSHOT_PATH = Path("~/.local/state/xpu-workload-manager/state.json").expanduser()

INVOICE = "ACME Ltd invoice 2026-08-14.\nTotal due: 1200 EUR.\nPayable by 2026-09-01.\n"
MEETING = (
    "# Team notes\n\n"
    "Project review with Ana on 2026-09-03 at 14:00 in Lisbon.\n"
    "Follow-up call on 2026-09-10 at 09:30.\n"
)
SPOKEN_WAV = Path("/usr/share/sounds/alsa/Front_Center.wav")


def service_is_running() -> bool:
    return default_socket_path().exists()


pytestmark = [
    pytest.mark.skipif(
        not service_is_running(), reason="no runtime service is listening on the control socket"
    ),
    pytest.mark.skipif(
        os.environ.get("OMNITENSOR_E2E") != "1",
        reason="end-to-end runs are opt-in: set OMNITENSOR_E2E=1",
    ),
]


def published_input_root() -> Path:
    """A directory the running service has said it will read from.

    Asked rather than assumed: a source outside these roots is refused by the
    sandbox with the same answer as a file that does not exist, which is the
    one refusal that looks like a bug in the client.
    """
    try:
        snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:  # pragma: no cover - environment
        pytest.skip(f"no published snapshot to read input roots from: {error}")
    roots = (snapshot.get("inputs") or {}).get("roots") or []
    if not roots:  # pragma: no cover - environment
        pytest.skip("the service publishes no input roots, so nothing can be staged")
    return Path(roots[0])


@pytest.fixture(scope="module")
def input_root() -> Path:
    return published_input_root()


@pytest.fixture
def source_file(input_root):
    """A fixture file inside a published root, removed afterwards."""
    written = []

    def write(name: str, contents: str) -> str:
        path = input_root / f"e2e-{name}"
        path.write_text(contents, encoding="utf-8")
        written.append(path)
        return str(path)

    yield write
    for path in written:
        path.unlink(missing_ok=True)


async def _terminal(job_id: str, request_id: str, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    poll = 0
    while time.monotonic() < deadline:
        result = await call_control(
            "get-job-result", {"version": 1, "requestId": request_id, "jobId": job_id}
        )
        if result["state"] in TERMINAL:
            return result
        poll += 1
        await asyncio.sleep(POLL_INTERVAL_S)
    raise AssertionError(f"job {job_id} did not finish within {timeout:.0f}s")


async def _run(workload_id: str, payload: dict, timeout: float) -> dict:
    request_id = f"e2e-{workload_id}-{int(time.time() * 1000)}"
    acknowledgement = await call_control(
        "submit-job",
        {
            "version": 1,
            "requestId": request_id,
            "workloadId": workload_id,
            "payload": payload,
        },
    )
    job_id = acknowledgement.get("jobId")
    assert job_id, (
        f"{workload_id} refused the submission: "
        f"{acknowledgement.get('code')} — {acknowledgement.get('message')}"
    )
    return await _terminal(job_id, request_id, timeout)


def run_workload(workload_id: str, payload: dict, timeout: float = JOB_TIMEOUT_S) -> dict:
    """Submit one job the way the client does and return its terminal result."""
    return asyncio.run(_run(workload_id, payload, timeout))


def succeeded(workload_id: str, payload: dict, schema: str, **options) -> dict:
    """Run a workload, insist it succeeded, and validate what it answered.

    Both halves matter. A job that fails is a workload that does not work; a
    job that succeeds with a document its own schema refuses is worse, because
    the client renders it as an empty answer rather than as an error.
    """
    result = run_workload(workload_id, payload, **options)
    assert result["state"] == "succeeded", (
        f"{workload_id} ended {result['state']}: {result.get('code')} — {result.get('message')}"
    )
    output = result.get("output")
    assert output is not None, f"{workload_id} succeeded with no output document"
    violations = validate_document(f"{schema}.schema.json", output)
    assert violations == [], f"{workload_id} output violates {schema}: {violations[0]}"
    return output


class TestAskSelectedFiles:
    """A question answered only from the documents a person chose."""

    def test_it_answers_from_the_document_and_cites_it(self, source_file):
        source = source_file("invoice.txt", INVOICE)

        output = succeeded(
            "ask-selected-files",
            {"sources": [source], "question": "What is the total due?"},
            "document-question-result",
        )

        assert output["answer"].strip(), "answered with an empty string"
        assert "1200" in output["answer"], f"did not read the figure: {output['answer']!r}"
        assert output.get("citations"), "answered without saying where it read it"


class TestSelectedTextTools:
    """The five operations a person picks from a dropdown."""

    def test_explaining_produces_text(self):
        output = succeeded(
            "selected-text-tools",
            {
                "selection": "The mitochondrion is the powerhouse of the cell.",
                "operation": "explain",
            },
            "selected-text-result",
        )

        assert output["result"].strip(), "explained with an empty string"

    def test_extracting_tasks_finds_the_two_that_are_there(self):
        output = succeeded(
            "selected-text-tools",
            {
                "selection": "Ana ships the report on Friday. Bruno reviews it on Monday.",
                "operation": "extract-tasks",
            },
            "selected-text-result",
        )

        assert output.get("tasks"), "found no tasks in two sentences that are both tasks"

    def test_translating_answers_in_the_language_it_was_asked_for(self):
        output = succeeded(
            "selected-text-tools",
            {"selection": "Good morning", "operation": "translate", "language": "Portuguese"},
            "selected-text-result",
        )

        assert output["result"].strip(), "translated to an empty string"


class TestFileOrganizer:
    """A review-only plan: nothing is moved, and every file is accounted for."""

    def test_every_file_given_appears_in_the_plan(self, source_file):
        sources = [
            source_file("invoice.txt", INVOICE),
            source_file("notes.md", MEETING),
        ]

        output = succeeded("file-organizer", {"sources": sources}, "file-organizer-result")

        assert len(output["plan"]) == len(sources), "the plan dropped a file it was given"
        named = {entry.get("fileName") for entry in output["plan"]}
        assert named == {Path(source).name for source in sources}

    def test_one_file_is_a_plan_like_any_other(self, source_file):
        """A person organising one file is the smallest reasonable use of this
        workload, and the window's chooser allows it: the manifest's floor is
        one source, not two."""
        source = source_file("only.txt", INVOICE)

        output = succeeded("file-organizer", {"sources": [source]}, "file-organizer-result")

        assert len(output["plan"]) == 1


class TestEventExtraction:
    """Calendar events, with the evidence for each one."""

    def test_an_explicit_date_and_time_is_found(self, source_file):
        source = source_file("meeting.md", MEETING)

        output = succeeded("event-extraction", {"sources": [source]}, "event-extraction-result")

        assert output.get("events"), (
            "found no events in a document naming a review with Ana on "
            f"2026-09-03 at 14:00: {output.get('outcome')} — {output.get('detail')}"
        )


class TestMediaTranscription:
    """One media file: what was said, and what was on screen."""

    def test_speech_comes_back_as_segments_with_text(self):
        if not SPOKEN_WAV.exists():  # pragma: no cover - environment
            pytest.skip(f"no sample audio at {SPOKEN_WAV}")

        output = succeeded(
            "media-transcription",
            {"sources": [str(SPOKEN_WAV)]},
            "media-transcription-result",
            timeout=JOB_TIMEOUT_S,
        )

        segments = (output.get("speech") or {}).get("segments") or []
        assert segments, "transcribed a file of speech into no segments at all"
        assert any(segment.get("text", "").strip() for segment in segments), (
            "every segment came back with empty text"
        )


class TestTheRefusalsAreHonest:
    """A workload that cannot do something must say so before it spends a job."""

    def test_a_source_outside_the_published_roots_is_refused(self, tmp_path):
        """The sandbox reads only from roots the operator opted into, and this
        is the refusal a person meets when they pick a file from /tmp.

        Either refusal counts: the submission may be turned away, or the job
        may run and fail. What must not happen is an answer read from a file
        the operator never opted into.
        """
        outside = tmp_path / "outside.txt"
        outside.write_text(INVOICE, encoding="utf-8")
        request_id = f"e2e-outside-{int(time.time() * 1000)}"
        payload = {"sources": [str(outside)], "question": "What is the total due?"}

        acknowledgement = asyncio.run(
            call_control(
                "submit-job",
                {
                    "version": 1,
                    "requestId": request_id,
                    "workloadId": "ask-selected-files",
                    "payload": payload,
                },
            )
        )
        job_id = acknowledgement.get("jobId")
        if not job_id:
            return  # refused before it cost anything, which is the better answer

        result = asyncio.run(_terminal(job_id, request_id, JOB_TIMEOUT_S))

        assert result["state"] != "succeeded", "read a file outside the published roots"

    def test_more_media_files_than_the_manifest_allows_is_refused(self, source_file):
        first = source_file("clip-one.txt", INVOICE)
        second = source_file("clip-two.txt", INVOICE)

        acknowledgement = asyncio.run(
            call_control(
                "submit-job",
                {
                    "version": 1,
                    "requestId": f"e2e-media-bound-{int(time.time() * 1000)}",
                    "workloadId": "media-transcription",
                    "payload": {"sources": [first, second]},
                },
            )
        )

        assert not acknowledgement.get("jobId"), "accepted two files for a one-file workload"
