"""Privacy-bounded public result and alert summaries."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from threading import Lock
from typing import Protocol

MAX_RESULT_SUMMARIES = 100
MAX_SUMMARY_TITLE_CHARS = 160
MAX_SUMMARY_TEXT_CHARS = 500
MAX_RESULT_REFERENCE_CHARS = 120
MAX_ALERT_ID_CHARS = 120
ADVISORY_RISK_MAX = 0.5
WARNING_RISK_MAX = 0.8
_PLUGIN_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_PUBLIC_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_RESULT_REFERENCE = re.compile(r"^result-[A-Za-z0-9._-]+$")


class SummarySeverity(StrEnum):
    ADVISORY = "advisory"
    WARNING = "warning"
    CRITICAL = "critical"


class TextRedactor(Protocol):
    def redact_text(self, text: str) -> str: ...


class SummaryError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class ResultSummary:
    alert_id: str
    plugin_id: str
    title: str
    summary: str
    severity: SummarySeverity
    timestamp_ms: int
    confidence: float | None
    risk_score: float | None
    resolved: bool
    result_reference: str

    def document(self) -> dict:
        return {
            "id": self.alert_id,
            "profileId": self.plugin_id,
            "title": self.title,
            "summary": self.summary,
            "severity": self.severity.value,
            "timestamp": self.timestamp_ms,
            "confidence": self.confidence,
            "riskScore": self.risk_score,
            "resolved": self.resolved,
            "resultRef": self.result_reference,
        }


class ResultSummaryRegistry:
    """Keep only bounded allowlisted public fields; never retain full output."""

    def __init__(
        self,
        *,
        max_summaries: int = MAX_RESULT_SUMMARIES,
        alert_id_factory: Callable[[], str],
    ) -> None:
        if (
            isinstance(max_summaries, bool)
            or not isinstance(max_summaries, int)
            or not 1 <= max_summaries <= MAX_RESULT_SUMMARIES
        ):
            raise ValueError(
                f"max_summaries must be between 1 and {MAX_RESULT_SUMMARIES}"
            )
        if not callable(alert_id_factory):
            raise TypeError("alert_id_factory must be callable")
        self._max_summaries = max_summaries
        self._alert_id_factory = alert_id_factory
        self._summaries: dict[str, ResultSummary] = {}
        self._lock = Lock()

    def publish(
        self,
        *,
        plugin_id: str,
        title: str,
        summary: str,
        timestamp_ms: int,
        confidence: float | None,
        risk_score: float | None,
        result_reference: str,
        redactor: TextRedactor,
    ) -> ResultSummary:
        _validate_plugin_id(plugin_id)
        _validate_timestamp(timestamp_ms)
        confidence = _optional_probability("confidence", confidence)
        risk_score = _optional_probability("risk_score", risk_score)
        _validate_result_reference(result_reference)
        if not hasattr(redactor, "redact_text"):
            raise TypeError("redactor must provide redact_text")
        public_title = _public_text(
            "title", title, redactor, MAX_SUMMARY_TITLE_CHARS, required=True
        )
        public_summary = _public_text(
            "summary", summary, redactor, MAX_SUMMARY_TEXT_CHARS, required=False
        )
        alert_id = self._alert_id_factory()
        _validate_public_id("generated alert ID", alert_id, MAX_ALERT_ID_CHARS)
        item = ResultSummary(
            alert_id,
            plugin_id,
            public_title,
            public_summary,
            severity_for_risk(risk_score),
            timestamp_ms,
            confidence,
            risk_score,
            False,
            result_reference,
        )
        with self._lock:
            if alert_id in self._summaries:
                raise SummaryError("alert-id-collision", "generated alert ID already exists")
            if len(self._summaries) >= self._max_summaries:
                self._summaries.pop(self._eviction_candidate().alert_id)
            self._summaries[alert_id] = item
        return item

    def resolve(self, alert_id: str) -> ResultSummary:
        _validate_public_id("alert ID", alert_id, MAX_ALERT_ID_CHARS)
        with self._lock:
            try:
                current = self._summaries[alert_id]
            except KeyError as error:
                raise SummaryError("alert-not-found", "alert was not found") from error
            if current.resolved:
                return current
            resolved = replace(current, resolved=True)
            self._summaries[alert_id] = resolved
            return resolved

    def summaries(self) -> tuple[ResultSummary, ...]:
        with self._lock:
            return tuple(sorted(self._summaries.values(), key=_display_order))

    def documents(self) -> list[dict]:
        return [summary.document() for summary in self.summaries()]

    def _eviction_candidate(self) -> ResultSummary:
        return min(self._summaries.values(), key=_eviction_order)


def severity_for_risk(risk_score: float | None) -> SummarySeverity:
    risk_score = _optional_probability("risk_score", risk_score)
    if risk_score is None or risk_score < ADVISORY_RISK_MAX:
        return SummarySeverity.ADVISORY
    if risk_score < WARNING_RISK_MAX:
        return SummarySeverity.WARNING
    return SummarySeverity.CRITICAL


def _display_order(summary: ResultSummary) -> tuple[int, str]:
    return (-summary.timestamp_ms, summary.alert_id)


def _eviction_order(summary: ResultSummary) -> tuple[bool, int, int, str]:
    severity_rank = {
        SummarySeverity.ADVISORY: 0,
        SummarySeverity.WARNING: 1,
        SummarySeverity.CRITICAL: 2,
    }[summary.severity]
    return (not summary.resolved, severity_rank, summary.timestamp_ms, summary.alert_id)


def _public_text(
    name: str,
    value: object,
    redactor: TextRedactor,
    maximum: int,
    *,
    required: bool,
) -> str:
    if not isinstance(value, str):
        raise SummaryError("invalid-summary", f"{name} must be a string")
    redacted = redactor.redact_text(value).replace("\x00", "")
    bounded = redacted[:maximum]
    if required and not bounded:
        raise SummaryError("invalid-summary", f"{name} must not be empty")
    return bounded


def _optional_probability(name: str, value: object) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise SummaryError(
            "invalid-probability", f"{name} must be finite and between 0 and 1"
        )
    return float(value)


def _validate_plugin_id(value: object) -> None:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= 80
        or _PLUGIN_ID.fullmatch(value) is None
    ):
        raise SummaryError("invalid-plugin-id", "plugin ID is invalid")


def _validate_result_reference(value: object) -> None:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= MAX_RESULT_REFERENCE_CHARS
        or _RESULT_REFERENCE.fullmatch(value) is None
    ):
        raise SummaryError("invalid-result-reference", "result reference is invalid")


def _validate_public_id(name: str, value: object, maximum: int) -> None:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= maximum
        or _PUBLIC_ID.fullmatch(value) is None
    ):
        raise SummaryError("invalid-public-id", f"{name} is invalid")


def _validate_timestamp(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SummaryError(
            "invalid-timestamp", "timestamp_ms must be a non-negative integer"
        )
