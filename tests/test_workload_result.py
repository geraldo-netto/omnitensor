"""The terminal-result envelope every workload shares."""

from __future__ import annotations

import asyncio

import pytest

from omnitensor.plugins.protocol import PluginRequest, PluginResultStatus
from omnitensor.sdk import PluginCancelledError, succeeded_result, workload_result


def request() -> PluginRequest:
    return PluginRequest("job-1", "sample-workload", "manual", {}, 1, None)


class RefusalError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def run(body, **options):
    discarded: list[str] = []

    async def discard(job_id: str) -> None:
        discarded.append(job_id)

    result = asyncio.run(
        workload_result(
            request(),
            body,
            completed_at_ms=lambda: 7,
            cancelled_detail="sample cancelled",
            failures=(RefusalError,),
            discard=discard,
            **options,
        )
    )
    return result, discarded


def test_a_body_that_finishes_returns_its_own_result_and_still_discards():
    async def body():
        return succeeded_result(request(), {"value": 1}, completed_at_ms=7)

    result, discarded = run(body)

    assert result.status is PluginResultStatus.SUCCEEDED
    assert discarded == ["job-1"]


def test_a_cancelled_body_is_reported_as_cancelled_not_failed():
    async def body():
        raise PluginCancelledError("the caller withdrew it")

    result, discarded = run(body)

    assert result.status is PluginResultStatus.CANCELLED
    assert result.detail == "sample cancelled"
    assert discarded == ["job-1"]


def test_a_declared_refusal_becomes_a_failed_result_carrying_its_code():
    async def body():
        raise RefusalError("source-unsupported")

    result, discarded = run(body)

    assert result.status is PluginResultStatus.FAILED
    assert result.detail == "source-unsupported"
    assert result.completed_at_ms == 7
    assert discarded == ["job-1"]


def test_an_undeclared_error_still_reaches_the_caller_and_still_discards():
    """One workload's bug is one job's failure, but the cleanup is not optional."""

    async def body():
        raise KeyError("nobody declared this")

    with pytest.raises(KeyError):
        run(body)


def test_every_document_workload_is_built_on_it():
    """OMNI-0527: five workloads each wrote the same three arms out."""
    import inspect

    from omnitensor.plugins.document_qa import DocumentQuestionPlugin
    from omnitensor.plugins.event_workload import EventExtractionPlugin
    from omnitensor.plugins.file_organizer import FileOrganizerPlugin
    from omnitensor.plugins.media_transcription import MediaTranscriptionPlugin
    from omnitensor.plugins.selected_text import SelectedTextPlugin

    for plugin in (
        DocumentQuestionPlugin,
        EventExtractionPlugin,
        FileOrganizerPlugin,
        MediaTranscriptionPlugin,
        SelectedTextPlugin,
    ):
        source = inspect.getsource(plugin.execute)
        assert "workload_result(" in source, plugin.__name__
        assert "cancelled_result(" not in source, plugin.__name__
        assert "failed_result(" not in source, plugin.__name__
