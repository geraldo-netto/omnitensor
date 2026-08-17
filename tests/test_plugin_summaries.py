from __future__ import annotations

import itertools
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.discovery import Device
from omnitensor.plugins import (
    MAX_RESULT_SUMMARIES,
    MAX_SUMMARY_TEXT_CHARS,
    MAX_SUMMARY_TITLE_CHARS,
    ResultSummary,
    ResultSummaryRegistry,
    SecretRedactor,
    SummaryError,
    SummarySeverity,
    severity_for_risk,
)
from omnitensor.registry import validate_document
from omnitensor.snapshot import build_snapshot


def registry(*ids, maximum=MAX_RESULT_SUMMARIES):
    values = iter(ids or (f"alert-{index}" for index in itertools.count()))
    return ResultSummaryRegistry(
        max_summaries=maximum,
        alert_id_factory=lambda: next(values),
    )


def publish(instance, **changes):
    arguments = {
        "plugin_id": "hardware-health",
        "title": "Hardware anomaly",
        "summary": "Temperature pattern changed",
        "timestamp_ms": 10,
        "confidence": 0.9,
        "risk_score": 0.85,
        "job_id": "job-1",
        "redactor": SecretRedactor([]),
    }
    arguments.update(changes)
    return instance.publish(**arguments)


@pytest.mark.parametrize(
    "risk, expected",
    [
        (None, SummarySeverity.ADVISORY),
        (0, SummarySeverity.ADVISORY),
        (0.499999, SummarySeverity.ADVISORY),
        (0.5, SummarySeverity.WARNING),
        (0.799999, SummarySeverity.WARNING),
        (0.8, SummarySeverity.CRITICAL),
        (1, SummarySeverity.CRITICAL),
    ],
)
def test_severity_thresholds_are_stable_and_boundary_inclusive(risk, expected):
    assert severity_for_risk(risk) is expected


@given(risk=st.floats(min_value=0, max_value=1, allow_nan=False))
def test_severity_is_monotonic_over_valid_risk(risk):
    rank = {
        SummarySeverity.ADVISORY: 0,
        SummarySeverity.WARNING: 1,
        SummarySeverity.CRITICAL: 2,
    }
    lower = severity_for_risk(risk / 2)
    upper = severity_for_risk(risk)
    assert rank[lower] <= rank[upper]


def test_publish_redacts_bounds_and_emits_only_allowlisted_snapshot_fields():
    instance = registry("alert-public")
    secret = "private-token"

    item = publish(
        instance,
        title=f"Hardware {secret}" + "x" * 200,
        summary=f"Evidence {secret}\x00" + "y" * 600,
        redactor=SecretRedactor([secret]),
    )

    assert item.alert_id == "alert-public"
    assert item.severity is SummarySeverity.CRITICAL
    assert secret not in item.title + item.summary
    assert "\x00" not in item.summary
    assert len(item.title) == MAX_SUMMARY_TITLE_CHARS
    assert len(item.summary) == MAX_SUMMARY_TEXT_CHARS
    assert [field.name for field in fields(ResultSummary)] == [
        "alert_id",
        "plugin_id",
        "title",
        "summary",
        "severity",
        "timestamp_ms",
        "confidence",
        "risk_score",
        "resolved",
        "job_id",
    ]
    assert item.document() == {
        "id": "alert-public",
        "profileId": "hardware-health",
        "title": item.title,
        "summary": item.summary,
        "severity": "critical",
        "timestamp": 10,
        "confidence": 0.9,
        "riskScore": 0.85,
        "resolved": False,
        "jobId": "job-1",
    }

    snapshot = build_snapshot(
        [Device("tpu-0", "tpu", "TPU", "pcie")],
        {"queueDepth": 0, "runningProfiles": 0},
        {},
        instance.documents(),
        generated_at_ms=1,
    )
    assert validate_document("runtime-snapshot.schema.json", snapshot) == []


def test_reference_is_opaque_and_full_result_is_never_retained():
    item = publish(registry("alert-1"), job_id="random-capability")
    assert item.job_id == "random-capability"
    assert "output" not in {field.name for field in fields(item)}
    assert "payload" not in item.document()


def test_resolution_is_idempotent_and_unknown_alert_is_stable():
    instance = registry("alert-1")
    original = publish(instance)

    resolved = instance.resolve("alert-1")

    assert not original.resolved
    assert resolved.resolved
    assert instance.resolve("alert-1") is resolved
    with pytest.raises(SummaryError) as excinfo:
        instance.resolve("missing")
    assert excinfo.value.code == "alert-not-found"
    assert excinfo.value.detail == "alert was not found"


def test_documents_are_newest_first_with_alert_id_tie_break():
    instance = registry("alert-z", "alert-b", "alert-a")
    publish(instance, timestamp_ms=2)
    publish(instance, timestamp_ms=3)
    publish(instance, timestamp_ms=3)

    assert [item.alert_id for item in instance.summaries()] == [
        "alert-a",
        "alert-b",
        "alert-z",
    ]
    assert [item["id"] for item in instance.documents()] == [
        "alert-a",
        "alert-b",
        "alert-z",
    ]


def test_capacity_evicts_resolved_before_unresolved_then_lowest_risk_oldest():
    instance = registry("critical", "resolved", "advisory", "new", maximum=3)
    publish(instance, risk_score=0.9, timestamp_ms=1)
    publish(instance, risk_score=0.7, timestamp_ms=2)
    instance.resolve("resolved")
    publish(instance, risk_score=0.1, timestamp_ms=0)

    publish(instance, risk_score=0.6, timestamp_ms=3)

    assert {item.alert_id for item in instance.summaries()} == {
        "critical",
        "advisory",
        "new",
    }

    second = registry("old-warning", "old-advisory", "new-critical", maximum=2)
    publish(second, risk_score=0.6, timestamp_ms=1)
    publish(second, risk_score=0.1, timestamp_ms=2)
    publish(second, risk_score=0.9, timestamp_ms=3)
    assert {item.alert_id for item in second.summaries()} == {
        "old-warning",
        "new-critical",
    }


@pytest.mark.parametrize(
    "field, value",
    [
        ("confidence", True),
        ("confidence", -0.1),
        ("confidence", 1.1),
        ("confidence", math.nan),
        ("risk_score", False),
        ("risk_score", -1),
        ("risk_score", math.inf),
        ("risk_score", "0.5"),
    ],
)
def test_probability_fields_are_finite_unit_interval(field, value):
    with pytest.raises(SummaryError) as excinfo:
        publish(registry("alert"), **{field: value})
    assert excinfo.value.code == "invalid-probability"
    assert field in excinfo.value.detail


@pytest.mark.parametrize("timestamp", [-1, True, 1.5, "1", None])
def test_timestamp_is_non_negative_integer(timestamp):
    with pytest.raises(SummaryError, match="timestamp_ms"):
        publish(registry("alert"), timestamp_ms=timestamp)


@pytest.mark.parametrize("plugin_id", ["", "Bad_ID", "a" * 81, None, 1])
def test_plugin_identity_is_strict(plugin_id):
    with pytest.raises(SummaryError) as excinfo:
        publish(registry("alert"), plugin_id=plugin_id)
    assert excinfo.value.code == "invalid-plugin-id"


@pytest.mark.parametrize(
    "reference",
    # A job id is what `get-job-result` accepts: bounded, and without the
    # spaces or slashes a path would carry.
    ["", "job 1", "job/1", "x" * 121, None],
)
def test_the_job_id_is_the_one_get_job_result_accepts(reference):
    with pytest.raises(SummaryError) as excinfo:
        publish(registry("alert"), job_id=reference)
    assert excinfo.value.code == "invalid-job-id"


@pytest.mark.parametrize("alert_id", ["", "bad id", "x" * 121, None, 1])
def test_generated_alert_ids_are_strict_and_collisions_reject(alert_id):
    instance = registry(alert_id)
    with pytest.raises(SummaryError) as excinfo:
        publish(instance)
    assert excinfo.value.code == "invalid-public-id"

    collision = registry("same", "same")
    publish(collision)
    with pytest.raises(SummaryError) as duplicate:
        publish(collision)
    assert duplicate.value.code == "alert-id-collision"


@pytest.mark.parametrize(
    "field, value",
    [("title", None), ("summary", None), ("title", ""), ("title", "\x00")],
)
def test_public_text_contract_is_typed_and_title_nonempty_after_sanitizing(field, value):
    with pytest.raises(SummaryError) as excinfo:
        publish(registry("alert"), **{field: value})
    assert excinfo.value.code == "invalid-summary"


class MissingRedactor:
    pass


def test_redactor_is_required_before_any_summary_is_retained():
    instance = registry("alert")
    with pytest.raises(TypeError, match="must provide redact_text"):
        publish(instance, redactor=MissingRedactor())
    assert instance.summaries() == ()


@pytest.mark.parametrize("maximum", [0, -1, True, 101, 1.5, "1"])
def test_registry_capacity_is_strict(maximum):
    with pytest.raises(ValueError, match="max_summaries"):
        registry(maximum=maximum)
    with pytest.raises(TypeError, match="must be callable"):
        ResultSummaryRegistry(alert_id_factory=None)


def test_concurrent_publish_is_bounded_without_lost_or_partial_documents():
    counter = itertools.count()
    instance = ResultSummaryRegistry(alert_id_factory=lambda: f"alert-{next(counter)}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        items = list(pool.map(lambda index: publish(instance, timestamp_ms=index), range(150)))

    assert len(items) == 150
    documents = instance.documents()
    assert len(documents) == MAX_RESULT_SUMMARIES
    assert len({document["id"] for document in documents}) == MAX_RESULT_SUMMARIES
