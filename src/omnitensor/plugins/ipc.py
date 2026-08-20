"""Bounded, versioned JSON framing for isolated plugin workers."""

from __future__ import annotations

import asyncio
import json
import math
import re
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from ..stable_error import StableError
from .protocol import (
    JsonObject,
    PluginProgress,
    PluginRequest,
    PluginResult,
    PluginResultStatus,
)

FRAME_FORMAT_VERSION = 1
DEFAULT_MAX_FRAME_BYTES = 1024 * 1024
_HEADER = struct.Struct("!I")
_PLUGIN_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_MAX_REQUEST_ID_CHARS = 128
_MAX_CAPABILITIES = 32
_MAX_CAPABILITY_CHARS = 64
_MAX_STAGE_CHARS = 80
# A JSON string costs at most six bytes for one character: an escaped control
# character. Every other character is cheaper written literally.
_WORST_CASE_CHAR_BYTES = 6
# Room for the chunk counter to grow past its first digit.
_SEQUENCE_DIGITS = 20


class WorkerMessageType(StrEnum):
    """Complete message vocabulary shared by service and worker."""

    HELLO = "hello"
    READY = "ready"
    CONFIGURE = "configure"
    COLLECT = "collect"
    PREPROCESS = "preprocess"
    INFER = "infer"
    POSTPROCESS = "postprocess"
    DELIVER = "deliver"
    PROGRESS = "progress"
    RESULT = "result"
    RESULT_CHUNK = "result-chunk"
    ERROR = "error"
    CANCEL = "cancel"
    HEALTH = "health"
    EXECUTE = "execute"


class IPCProtocolError(StableError, ValueError):
    """Stable boundary error safe to map to worker diagnostics."""


@dataclass(frozen=True, slots=True)
class IPCFrame:
    """One independently bounded request, response, or observation."""

    version: int
    type: WorkerMessageType
    request_id: str | None
    payload: JsonObject


@dataclass(frozen=True, slots=True)
class HandshakeOffer:
    """Identity, protocol range, and optional features offered by one peer."""

    plugin_id: str
    minimum_version: int
    maximum_version: int
    capabilities: frozenset[str]


@dataclass(frozen=True, slots=True)
class HandshakeAgreement:
    """Highest common protocol and mutually supported optional features."""

    plugin_id: str
    protocol_version: int
    capabilities: frozenset[str]


def encode_frame(
    frame: IPCFrame,
    *,
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> bytes:
    """Encode one length-prefixed frame after validating its boundary fields."""
    _validate_limit(max_frame_bytes)
    _validate_frame(frame)
    document = {
        "payload": dict(frame.payload),
        "requestId": frame.request_id,
        "type": str(frame.type),
        "version": frame.version,
    }
    try:
        body = json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise IPCProtocolError("invalid-payload", "payload must contain JSON values") from error
    if len(body) > max_frame_bytes:
        raise IPCProtocolError(
            "frame-too-large",
            f"frame is {len(body)} bytes; limit is {max_frame_bytes}",
        )
    return _HEADER.pack(len(body)) + body


def decode_frame(
    data: bytes,
    *,
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> IPCFrame:
    """Decode exactly one complete frame and reject truncation or trailing data."""
    _validate_limit(max_frame_bytes)
    if len(data) < _HEADER.size:
        raise IPCProtocolError("truncated-frame", "missing length header")
    (declared_size,) = _HEADER.unpack(data[: _HEADER.size])
    _validate_declared_size(declared_size, max_frame_bytes)
    actual_size = len(data) - _HEADER.size
    if actual_size != declared_size:
        raise IPCProtocolError(
            "frame-size-mismatch",
            f"declared {declared_size} bytes; received {actual_size}",
        )
    return _decode_body(data[_HEADER.size :])


async def read_frame(
    reader: asyncio.StreamReader,
    *,
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> IPCFrame:
    """Read one frame without allocating a body above the configured bound."""
    _validate_limit(max_frame_bytes)
    try:
        header = await reader.readexactly(_HEADER.size)
        (declared_size,) = _HEADER.unpack(header)
        _validate_declared_size(declared_size, max_frame_bytes)
        body = await reader.readexactly(declared_size)
    except asyncio.IncompleteReadError as error:
        raise IPCProtocolError(
            "truncated-frame",
            f"expected {error.expected} bytes; received {len(error.partial)}",
        ) from error
    return _decode_body(body)


async def write_frame(
    writer: asyncio.StreamWriter,
    frame: IPCFrame,
    *,
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> None:
    """Write and drain exactly one validated frame."""
    writer.write(encode_frame(frame, max_frame_bytes=max_frame_bytes))
    await writer.drain()


def handshake_frame(offer: HandshakeOffer) -> IPCFrame:
    """Encode a validated protocol offer as the initial peer message."""
    _validate_offer(offer)
    return IPCFrame(
        FRAME_FORMAT_VERSION,
        WorkerMessageType.HELLO,
        None,
        {
            "pluginId": offer.plugin_id,
            "minimumVersion": offer.minimum_version,
            "maximumVersion": offer.maximum_version,
            "capabilities": sorted(offer.capabilities),
        },
    )


def _require_frame_version(frame: IPCFrame, expected: int) -> None:
    """Refuse a frame whose version is not the one in force.

    Only the handshake pair checked this, so every later frame was parsed as
    though its field meanings were settled -- in whatever version the peer
    happened to send.  Before the handshake the version in force is
    :data:`FRAME_FORMAT_VERSION`; after it, the negotiated one.
    """
    if frame.version != expected:
        raise IPCProtocolError(
            "frame-version-incompatible",
            f"expected {expected}; received {frame.version}",
        )


def parse_handshake(frame: IPCFrame) -> HandshakeOffer:
    """Parse a strict initial hello frame into a validated offer."""
    _require_frame_version(frame, FRAME_FORMAT_VERSION)
    if frame.type is not WorkerMessageType.HELLO or frame.request_id is not None:
        raise IPCProtocolError("invalid-handshake", "first frame must be an uncorrelated hello")
    expected = {"pluginId", "minimumVersion", "maximumVersion", "capabilities"}
    if set(frame.payload) != expected:
        raise IPCProtocolError("invalid-handshake", "hello fields do not match the contract")
    capabilities = frame.payload["capabilities"]
    if not isinstance(capabilities, list) or not all(
        isinstance(capability, str) for capability in capabilities
    ):
        raise IPCProtocolError("invalid-handshake", "capabilities must be a string array")
    offer = HandshakeOffer(
        plugin_id=frame.payload["pluginId"],
        minimum_version=frame.payload["minimumVersion"],
        maximum_version=frame.payload["maximumVersion"],
        capabilities=frozenset(capabilities),
    )
    _validate_offer(offer)
    return offer


def ready_frame(plugin_id: str) -> IPCFrame:
    """Encode the worker's signal that its plugin finished starting.

    Startup is a separate phase from the handshake: the handshake settles the
    protocol in milliseconds, while a plugin's ``start`` may legitimately load
    models or open devices.  Signalling them with distinct frames lets each be
    bounded by its own deadline.
    """
    return IPCFrame(FRAME_FORMAT_VERSION, WorkerMessageType.READY, None, {"pluginId": plugin_id})


def execute_frame(request: PluginRequest) -> IPCFrame:
    """Encode one manual plugin request without placing content outside payload."""
    _validate_request(request)
    return IPCFrame(
        FRAME_FORMAT_VERSION,
        WorkerMessageType.EXECUTE,
        request.job_id,
        {
            "pluginId": request.plugin_id,
            "trigger": request.trigger,
            "payload": dict(request.payload),
            "submittedAt": request.submitted_at_ms,
            "deadlineAt": request.deadline_at_ms,
        },
    )


def parse_execute(
    frame: IPCFrame, *, protocol_version: int = FRAME_FORMAT_VERSION
) -> PluginRequest:
    """Parse the exact service-to-worker execution request contract."""
    _require_frame_version(frame, protocol_version)
    if frame.type is not WorkerMessageType.EXECUTE or frame.request_id is None:
        raise IPCProtocolError("invalid-execute", "expected a correlated execute frame")
    if set(frame.payload) != {"pluginId", "trigger", "payload", "submittedAt", "deadlineAt"}:
        raise IPCProtocolError("invalid-execute", "execute fields do not match the contract")
    request = PluginRequest(
        frame.request_id,
        frame.payload["pluginId"],
        frame.payload["trigger"],
        frame.payload["payload"],
        frame.payload["submittedAt"],
        frame.payload["deadlineAt"],
    )
    _validate_request(request)
    return request


def progress_frame(progress: PluginProgress) -> IPCFrame:
    """Encode one bounded worker progress update."""
    _validate_progress(progress)
    return IPCFrame(
        FRAME_FORMAT_VERSION,
        WorkerMessageType.PROGRESS,
        progress.job_id,
        {
            "stage": progress.stage,
            "fraction": float(progress.fraction),
            "detail": progress.detail,
            "observedAt": progress.observed_at_ms,
        },
    )


def parse_progress(
    frame: IPCFrame, *, protocol_version: int = FRAME_FORMAT_VERSION
) -> PluginProgress:
    """Parse a bounded progress update from one worker."""
    _require_frame_version(frame, protocol_version)
    if frame.type is not WorkerMessageType.PROGRESS or frame.request_id is None:
        raise IPCProtocolError("invalid-progress", "expected a correlated progress frame")
    if set(frame.payload) != {"stage", "fraction", "detail", "observedAt"}:
        raise IPCProtocolError("invalid-progress", "progress fields do not match the contract")
    progress = PluginProgress(
        frame.request_id,
        frame.payload["stage"],
        frame.payload["fraction"],
        frame.payload["detail"],
        frame.payload["observedAt"],
    )
    _validate_progress(progress)
    return progress


def result_frame(result: PluginResult) -> IPCFrame:
    """Encode exactly one terminal worker result."""
    _validate_result(result)
    return IPCFrame(
        FRAME_FORMAT_VERSION,
        WorkerMessageType.RESULT,
        result.job_id,
        {
            "status": str(result.status),
            "output": dict(result.output),
            "detail": result.detail,
            "completedAt": result.completed_at_ms,
        },
    )


def result_frames(
    result: PluginResult,
    *,
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> tuple[IPCFrame, ...]:
    """One result as however many frames it takes.

    An answer is as long as the question deserves: a document-QA answer or a
    long transcript over the frame size used to be refused outright, and the
    refusal cost a full model reload on the next request. A result that fits
    is still exactly one RESULT frame — the common case, byte for byte as
    before — and one that does not is carried as ordered RESULT_CHUNK frames
    that :func:`assemble_result_chunks` joins back into the same result.
    """
    frame = result_frame(result)
    try:
        encode_frame(frame, max_frame_bytes=max_frame_bytes)
    except IPCProtocolError as error:
        if error.code != "frame-too-large":
            raise
    else:
        return (frame,)
    body = json.dumps(
        dict(frame.payload),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    parts = _split_body(body, result.job_id, max_frame_bytes)
    return tuple(
        IPCFrame(
            FRAME_FORMAT_VERSION,
            WorkerMessageType.RESULT_CHUNK,
            result.job_id,
            {"sequence": index, "final": index == len(parts) - 1, "body": part},
        )
        for index, part in enumerate(parts)
    )


def _split_body(body: str, request_id: str, max_frame_bytes: int) -> tuple[str, ...]:
    """Cut the encoded result at character boundaries that always fit a frame.

    Sized against the worst case a JSON string can cost per character — six
    bytes, for a control character written as an escape — rather than against
    the length of this particular text, because a chunk that overflows on
    encoding would be a protocol failure rather than a slower answer.
    """
    envelope = len(
        encode_frame(
            IPCFrame(
                FRAME_FORMAT_VERSION,
                WorkerMessageType.RESULT_CHUNK,
                request_id,
                {"sequence": 0, "final": False, "body": ""},
            ),
            max_frame_bytes=max_frame_bytes,
        )
    )
    span = max(1, (max_frame_bytes - envelope - _SEQUENCE_DIGITS) // _WORST_CASE_CHAR_BYTES)
    return tuple(body[start : start + span] for start in range(0, len(body), span)) or ("",)


def parse_result_chunk(
    frame: IPCFrame,
    expected_sequence: int,
    *,
    protocol_version: int = FRAME_FORMAT_VERSION,
) -> tuple[str, bool]:
    """One chunk of a split result, in the order the worker sent it."""
    _require_frame_version(frame, protocol_version)
    if frame.type is not WorkerMessageType.RESULT_CHUNK or frame.request_id is None:
        raise IPCProtocolError("invalid-result", "expected a correlated result chunk")
    if set(frame.payload) != {"sequence", "final", "body"}:
        raise IPCProtocolError("invalid-result", "result chunk fields do not match the contract")
    sequence = frame.payload["sequence"]
    final = frame.payload["final"]
    body = frame.payload["body"]
    if type(sequence) is not int or sequence != expected_sequence:
        raise IPCProtocolError("invalid-result", "result chunks arrived out of order")
    if not isinstance(final, bool) or not isinstance(body, str):
        raise IPCProtocolError("invalid-result", "result chunk fields are invalid")
    return body, final


def assemble_result_chunks(request_id: str, parts: Sequence[str]) -> PluginResult:
    """Rebuild the result a worker had to split, or refuse the whole answer."""
    try:
        payload = json.loads("".join(parts))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise IPCProtocolError("invalid-result", "split result is not valid JSON") from error
    if not isinstance(payload, dict):
        raise IPCProtocolError("invalid-result", "split result is not an object")
    return parse_result(
        IPCFrame(FRAME_FORMAT_VERSION, WorkerMessageType.RESULT, request_id, payload)
    )


def parse_result(frame: IPCFrame, *, protocol_version: int = FRAME_FORMAT_VERSION) -> PluginResult:
    """Parse exactly one terminal result from a worker."""
    _require_frame_version(frame, protocol_version)
    if frame.type is not WorkerMessageType.RESULT or frame.request_id is None:
        raise IPCProtocolError("invalid-result", "expected a correlated result frame")
    if set(frame.payload) != {"status", "output", "detail", "completedAt"}:
        raise IPCProtocolError("invalid-result", "result fields do not match the contract")
    try:
        status = PluginResultStatus(frame.payload["status"])
    except (TypeError, ValueError) as error:
        raise IPCProtocolError("invalid-result", "result status is invalid") from error
    result = PluginResult(
        frame.request_id,
        status,
        frame.payload["output"],
        frame.payload["detail"],
        frame.payload["completedAt"],
    )
    _validate_result(result)
    return result


def parse_ready(frame: IPCFrame, plugin_id: str) -> None:
    """Accept a strict ready frame for ``plugin_id`` or reject the worker."""
    _require_frame_version(frame, FRAME_FORMAT_VERSION)
    if frame.type is not WorkerMessageType.READY or frame.request_id is not None:
        raise IPCProtocolError("invalid-ready", "expected an uncorrelated ready frame")
    if set(frame.payload) != {"pluginId"}:
        raise IPCProtocolError("invalid-ready", "ready fields do not match the contract")
    if frame.payload["pluginId"] != plugin_id:
        raise IPCProtocolError(
            "plugin-identity-mismatch",
            f"expected {plugin_id}; received {frame.payload['pluginId']!r}",
        )


async def await_worker_ready(
    reader: asyncio.StreamReader,
    plugin_id: str,
    *,
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> None:
    """Wait for the worker to report that its plugin finished starting."""
    parse_ready(await read_frame(reader, max_frame_bytes=max_frame_bytes), plugin_id)


def negotiate_handshake(
    service: HandshakeOffer,
    worker: HandshakeOffer,
) -> HandshakeAgreement:
    """Select the highest common protocol and reject identity substitution."""
    _validate_offer(service)
    _validate_offer(worker)
    if service.plugin_id != worker.plugin_id:
        raise IPCProtocolError(
            "plugin-identity-mismatch",
            f"expected {service.plugin_id}; received {worker.plugin_id}",
        )
    minimum = max(service.minimum_version, worker.minimum_version)
    maximum = min(service.maximum_version, worker.maximum_version)
    if minimum > maximum:
        raise IPCProtocolError(
            "protocol-version-incompatible",
            f"service {service.minimum_version}-{service.maximum_version}; "
            f"worker {worker.minimum_version}-{worker.maximum_version}",
        )
    return HandshakeAgreement(
        service.plugin_id,
        maximum,
        service.capabilities & worker.capabilities,
    )


async def perform_service_handshake(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    offer: HandshakeOffer,
    *,
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> HandshakeAgreement:
    """Send the service offer, receive the worker identity, and negotiate."""
    await write_frame(writer, handshake_frame(offer), max_frame_bytes=max_frame_bytes)
    worker = parse_handshake(await read_frame(reader, max_frame_bytes=max_frame_bytes))
    return negotiate_handshake(offer, worker)


async def perform_worker_handshake(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    offer: HandshakeOffer,
    *,
    max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
) -> HandshakeAgreement:
    """Verify the service offer before returning the worker identity."""
    service = parse_handshake(await read_frame(reader, max_frame_bytes=max_frame_bytes))
    agreement = negotiate_handshake(service, offer)
    await write_frame(writer, handshake_frame(offer), max_frame_bytes=max_frame_bytes)
    return agreement


def _decode_body(body: bytes) -> IPCFrame:
    try:
        document = json.loads(
            body.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise IPCProtocolError("invalid-json", "frame body must be finite UTF-8 JSON") from error
    if not isinstance(document, dict) or set(document) != {
        "version",
        "type",
        "requestId",
        "payload",
    }:
        raise IPCProtocolError("invalid-frame", "frame fields do not match the contract")
    try:
        message_type = WorkerMessageType(document["type"])
    except (TypeError, ValueError) as error:
        raise IPCProtocolError(
            "unknown-message-type",
            f"unsupported type: {document['type']!r}",
        ) from error
    frame = IPCFrame(
        document["version"],
        message_type,
        document["requestId"],
        document["payload"],
    )
    _validate_frame(frame)
    return frame


def _validate_frame(frame: IPCFrame) -> None:
    if isinstance(frame.version, bool) or not isinstance(frame.version, int) or frame.version < 1:
        raise IPCProtocolError("invalid-frame-version", "version must be a positive integer")
    if not isinstance(frame.type, WorkerMessageType):
        raise IPCProtocolError("unknown-message-type", f"unsupported type: {frame.type!r}")
    if frame.request_id is not None and (
        not isinstance(frame.request_id, str)
        or not frame.request_id
        or len(frame.request_id) > _MAX_REQUEST_ID_CHARS
    ):
        raise IPCProtocolError(
            "invalid-request-id",
            f"requestId must be null or 1-{_MAX_REQUEST_ID_CHARS} characters",
        )
    if not isinstance(frame.payload, dict):
        raise IPCProtocolError("invalid-payload", "payload must be a JSON object")


def _validate_request(request: PluginRequest) -> None:
    if not isinstance(request, PluginRequest):
        raise IPCProtocolError("invalid-execute", "request must be a PluginRequest")
    if not isinstance(request.job_id, str) or not 1 <= len(request.job_id) <= _MAX_REQUEST_ID_CHARS:
        raise IPCProtocolError("invalid-execute", "request job id is invalid")
    if not isinstance(request.plugin_id, str) or _PLUGIN_ID.fullmatch(request.plugin_id) is None:
        raise IPCProtocolError("invalid-execute", "request plugin id is invalid")
    if not isinstance(request.trigger, str) or not 1 <= len(request.trigger) <= 80:
        raise IPCProtocolError("invalid-execute", "request trigger is invalid")
    if not isinstance(request.payload, dict):
        raise IPCProtocolError("invalid-execute", "request payload must be an object")
    if (
        isinstance(request.submitted_at_ms, bool)
        or not isinstance(request.submitted_at_ms, int)
        or request.submitted_at_ms < 1
    ):
        raise IPCProtocolError("invalid-execute", "request timestamp is invalid")
    deadline = request.deadline_at_ms
    if deadline is not None and (
        isinstance(deadline, bool)
        or not isinstance(deadline, int)
        or deadline < request.submitted_at_ms
    ):
        raise IPCProtocolError("invalid-execute", "request deadline is invalid")


def _validate_progress(progress: PluginProgress) -> None:
    if not isinstance(progress, PluginProgress):
        raise IPCProtocolError("invalid-progress", "progress must be PluginProgress")
    fraction = progress.fraction
    if (
        not isinstance(progress.stage, str)
        or not 1 <= len(progress.stage) <= _MAX_STAGE_CHARS
        or isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(fraction)
        or not 0 <= fraction <= 1
        or not isinstance(progress.detail, str)
        or isinstance(progress.observed_at_ms, bool)
        or not isinstance(progress.observed_at_ms, int)
        or progress.observed_at_ms < 1
    ):
        raise IPCProtocolError("invalid-progress", "progress fields are invalid")
    _validate_request_id(progress.job_id, "progress")


def _validate_result(result: PluginResult) -> None:
    if not isinstance(result, PluginResult):
        raise IPCProtocolError("invalid-result", "result must be PluginResult")
    if (
        not isinstance(result.status, PluginResultStatus)
        or not isinstance(result.output, dict)
        or not isinstance(result.detail, str)
        or isinstance(result.completed_at_ms, bool)
        or not isinstance(result.completed_at_ms, int)
        or result.completed_at_ms < 1
    ):
        raise IPCProtocolError("invalid-result", "result fields are invalid")
    _validate_request_id(result.job_id, "result")


def _validate_request_id(value: object, label: str) -> None:
    if not isinstance(value, str) or not 1 <= len(value) <= _MAX_REQUEST_ID_CHARS:
        raise IPCProtocolError(f"invalid-{label}", f"{label} job id is invalid")


def _validate_offer(offer: HandshakeOffer) -> None:
    if not isinstance(offer.plugin_id, str) or not _PLUGIN_ID.fullmatch(offer.plugin_id):
        raise IPCProtocolError("invalid-plugin-id", f"invalid plugin id: {offer.plugin_id!r}")
    versions = (offer.minimum_version, offer.maximum_version)
    if any(isinstance(version, bool) or not isinstance(version, int) for version in versions):
        raise IPCProtocolError("invalid-protocol-range", "protocol versions must be integers")
    if offer.minimum_version < 1 or offer.maximum_version < offer.minimum_version:
        raise IPCProtocolError(
            "invalid-protocol-range",
            "protocol range must be positive and ordered",
        )
    if len(offer.capabilities) > _MAX_CAPABILITIES or any(
        not isinstance(capability, str) or not capability or len(capability) > _MAX_CAPABILITY_CHARS
        for capability in offer.capabilities
    ):
        raise IPCProtocolError(
            "invalid-capabilities",
            f"at most {_MAX_CAPABILITIES} capabilities of 1-{_MAX_CAPABILITY_CHARS} characters",
        )


def _validate_limit(max_frame_bytes: int) -> None:
    invalid_type = isinstance(max_frame_bytes, bool) or not isinstance(max_frame_bytes, int)
    if invalid_type or max_frame_bytes < 1:
        raise ValueError("max_frame_bytes must be a positive integer")


def _validate_declared_size(declared_size: int, max_frame_bytes: int) -> None:
    if declared_size > max_frame_bytes:
        raise IPCProtocolError(
            "frame-too-large",
            f"declared {declared_size} bytes; limit is {max_frame_bytes}",
        )
