"""Bounded JSON codec for the versioned runtime job wire contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass

from .job_ports import MAX_JOB_MESSAGE_CHARS
from .plugins.job_results import JobRecord
from .registry import validate_document

JOB_API_VERSION = 1
DEFAULT_MAX_JOB_REQUEST_BYTES = 64 * 1024
MAX_JOB_REQUEST_BYTES_LIMIT = 1024 * 1024


@dataclass(frozen=True, slots=True)
class JobRequest:
    request_id: str
    workload_id: str
    payload: dict


class _RequestError(ValueError):
    def __init__(self, request_id: str, code: str, message: str) -> None:
        super().__init__(message)
        self.request_id = request_id
        self.code = code
        self.message = message


def _parse_request(text: str, schema: str, max_bytes: int) -> dict:
    if not isinstance(text, str):
        raise _RequestError("invalid", "invalid-request", "Request must be JSON text")
    try:
        encoded_size = len(text.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise _RequestError(
            "invalid", "invalid-request", "Request is not valid UTF-8 text"
        ) from error
    if encoded_size > max_bytes:
        raise _RequestError("invalid", "request-too-large", f"Request exceeds {max_bytes} bytes")
    try:
        document = json.loads(text, parse_constant=_reject_json_constant)
    except (ValueError, RecursionError) as error:
        raise _RequestError("invalid", "invalid-json", "Request is not valid JSON") from error
    request_id = _request_id(document)
    if not isinstance(document, dict):
        raise _RequestError(request_id, "invalid-request", "Request must be a JSON object")
    if validate_document(schema, document):
        raise _RequestError(
            request_id,
            "invalid-request",
            "Request does not match the version 1 contract",
        )
    return document


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant: {value}")


def _request_id(document: object) -> str:
    if isinstance(document, dict):
        candidate = document.get("requestId")
        if _valid_identifier(candidate):
            return candidate
    return "invalid"


def _request_id_from_text(text: object, max_request_bytes: int) -> str:
    """Recover a bounded safe request id from a request rejected elsewhere."""
    if not isinstance(text, str):
        return "invalid"
    if len(text.encode("utf-8", "surrogatepass")) > max_request_bytes:
        return "invalid"
    try:
        return _request_id(json.loads(text, parse_constant=_reject_json_constant))
    except (ValueError, RecursionError):
        return "invalid"


def _valid_identifier(value: object) -> bool:
    return (
        isinstance(value, str)
        and 1 <= len(value) <= 120
        and all(
            character.isascii() and (character.isalnum() or character in "._-")
            for character in value
        )
    )


def _validate_identifier(name: str, value: object) -> None:
    if not _valid_identifier(value):
        raise ValueError(f"{name} is invalid")


def _acknowledgement_reply(
    request_id: str,
    job_id: str | None,
    status: str,
    code: str,
    message: str,
    timestamp: int,
) -> str:
    document = {
        "version": JOB_API_VERSION,
        "requestId": request_id,
        "jobId": job_id,
        "status": status,
        "code": code,
        "message": message[:MAX_JOB_MESSAGE_CHARS],
        "timestamp": timestamp,
    }
    violations = validate_document("runtime-job-acknowledgement.schema.json", document)
    if violations:  # pragma: no cover - contract self-check
        raise RuntimeError(f"job acknowledgement violates contract: {violations}")
    return json.dumps(document, separators=(",", ":"), allow_nan=False)


def _validated_result_reply(reply: str) -> str:
    """Validate the exact job-result text at its single emission boundary."""
    try:
        document = json.loads(reply, parse_constant=_reject_json_constant)
    except (TypeError, ValueError, RecursionError) as error:
        raise RuntimeError("job result reply is not valid JSON") from error
    violations = validate_document("runtime-job-result.schema.json", document)
    if violations:
        raise RuntimeError(f"job result reply violates contract: {violations}")
    return reply


def _result_reply(
    request_id: str, job_id: str, state: str, code: str, message: str, timestamp: int
) -> str:
    return json.dumps(
        {
            "version": 1,
            "requestId": request_id if isinstance(request_id, str) else "unknown",
            "jobId": job_id if isinstance(job_id, str) and job_id else "unknown",
            "state": state,
            "code": code,
            "message": message[:500],
            "timestamp": timestamp,
            "progress": None,
            "output": None,
        },
        separators=(",", ":"),
        allow_nan=False,
    )


def _record_reply(request_id: str, record: JobRecord, timestamp: int) -> str:
    """Project a stored record into the published result contract."""
    result = record.result
    progress = record.progress
    if result is None:
        return json.dumps(
            {
                "version": 1,
                "requestId": request_id,
                "jobId": record.job_id,
                "state": "running",
                "code": "job-running",
                "message": progress.detail[:500] if progress else "Job is still running",
                "timestamp": timestamp,
                "progress": (
                    None
                    if progress is None
                    else {"fraction": progress.fraction, "detail": progress.detail[:200]}
                ),
                "output": None,
            },
            separators=(",", ":"),
            allow_nan=False,
        )
    return json.dumps(
        {
            "version": 1,
            "requestId": request_id,
            "jobId": record.job_id,
            "state": str(result.status),
            "code": f"job-{result.status}",
            "message": result.detail[:500] or "Job finished",
            "timestamp": timestamp,
            "progress": None,
            "output": result.output if isinstance(result.output, dict) else None,
        },
        separators=(",", ":"),
        allow_nan=False,
    )
