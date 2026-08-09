"""Bounded, versioned JSON framing for isolated plugin workers."""

from __future__ import annotations

import asyncio
import json
import re
import struct
from dataclasses import dataclass
from enum import StrEnum

from .protocol import JsonObject

FRAME_FORMAT_VERSION = 1
DEFAULT_MAX_FRAME_BYTES = 1024 * 1024
_HEADER = struct.Struct("!I")
_PLUGIN_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_MAX_REQUEST_ID_CHARS = 128
_MAX_CAPABILITIES = 32
_MAX_CAPABILITY_CHARS = 64


class WorkerMessageType(StrEnum):
    """Complete message vocabulary shared by service and worker."""

    HELLO = "hello"
    CONFIGURE = "configure"
    COLLECT = "collect"
    PREPROCESS = "preprocess"
    INFER = "infer"
    POSTPROCESS = "postprocess"
    DELIVER = "deliver"
    PROGRESS = "progress"
    RESULT = "result"
    ERROR = "error"
    CANCEL = "cancel"
    HEALTH = "health"


class IPCProtocolError(ValueError):
    """Stable boundary error safe to map to worker diagnostics."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


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


def parse_handshake(frame: IPCFrame) -> HandshakeOffer:
    """Parse a strict initial hello frame into a validated offer."""
    if frame.version != FRAME_FORMAT_VERSION:
        raise IPCProtocolError(
            "frame-version-incompatible",
            f"expected {FRAME_FORMAT_VERSION}; received {frame.version}",
        )
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
        not isinstance(capability, str)
        or not capability
        or len(capability) > _MAX_CAPABILITY_CHARS
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
