from __future__ import annotations

import asyncio
import json
import struct

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.plugins import (
    FRAME_FORMAT_VERSION,
    HandshakeAgreement,
    HandshakeOffer,
    IPCFrame,
    IPCProtocolError,
    PluginProgress,
    PluginRequest,
    PluginResult,
    PluginResultStatus,
    WorkerMessageType,
    assemble_result_chunks,
    decode_frame,
    encode_frame,
    execute_frame,
    handshake_frame,
    negotiate_handshake,
    parse_execute,
    parse_handshake,
    parse_progress,
    parse_result,
    parse_result_chunk,
    perform_service_handshake,
    perform_worker_handshake,
    progress_frame,
    read_frame,
    result_frame,
    result_frames,
    write_frame,
)


def offer(
    plugin_id="echo-plugin",
    minimum=1,
    maximum=3,
    capabilities=frozenset({"progress", "cancel"}),
):
    return HandshakeOffer(plugin_id, minimum, maximum, capabilities)


def raw_frame(document) -> bytes:
    body = json.dumps(document, separators=(",", ":")).encode()
    return struct.pack("!I", len(body)) + body


@pytest.mark.parametrize("message_type", list(WorkerMessageType))
def test_every_worker_message_has_a_bounded_round_trip(message_type):
    frame = IPCFrame(1, message_type, "job-1", {"nested": [True, None, 3.5, "✓"]})

    encoded = encode_frame(frame)

    assert struct.unpack("!I", encoded[:4]) == (len(encoded) - 4,)
    assert decode_frame(encoded) == frame


def test_encoding_is_deterministic_and_enforces_the_exact_byte_limit():
    frame = IPCFrame(1, WorkerMessageType.RESULT, None, {"z": 1, "a": 2})
    encoded = encode_frame(frame)
    body_size = len(encoded) - 4

    assert encode_frame(frame) == encoded
    assert encode_frame(frame, max_frame_bytes=body_size) == encoded
    with pytest.raises(IPCProtocolError) as excinfo:
        encode_frame(frame, max_frame_bytes=body_size - 1)
    assert excinfo.value.code == "frame-too-large"


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "1"])
def test_frame_limit_must_be_a_positive_integer(limit):
    frame = IPCFrame(1, WorkerMessageType.HEALTH, None, {})
    with pytest.raises(ValueError, match="max_frame_bytes must be a positive integer"):
        encode_frame(frame, max_frame_bytes=limit)
    with pytest.raises(ValueError, match="max_frame_bytes must be a positive integer"):
        decode_frame(b"", max_frame_bytes=limit)


@pytest.mark.parametrize(
    ("data", "code"),
    [
        (b"", "truncated-frame"),
        (struct.pack("!I", 2) + b"{}x", "frame-size-mismatch"),
        (struct.pack("!I", 3) + b"{}", "frame-size-mismatch"),
        (struct.pack("!I", 2) + b"\xff\xff", "invalid-json"),
        (struct.pack("!I", 3) + b"NaN", "invalid-json"),
    ],
)
def test_decoder_rejects_truncation_trailing_data_and_invalid_json(data, code):
    with pytest.raises(IPCProtocolError) as excinfo:
        decode_frame(data)
    assert excinfo.value.code == code


def test_decoder_rejects_an_oversized_declared_body_before_receiving_it():
    with pytest.raises(IPCProtocolError) as excinfo:
        decode_frame(struct.pack("!I", 11), max_frame_bytes=10)
    assert str(excinfo.value) == "frame-too-large: declared 11 bytes; limit is 10"


def test_decoder_accepts_a_declared_body_equal_to_the_limit():
    frame = IPCFrame(1, WorkerMessageType.HEALTH, None, {})
    encoded = encode_frame(frame)

    assert decode_frame(encoded, max_frame_bytes=len(encoded) - 4) == frame


@pytest.mark.parametrize(
    ("document", "code"),
    [
        ([], "invalid-frame"),
        ({"version": 1, "type": "health", "requestId": None}, "invalid-frame"),
        (
            {
                "version": 1,
                "type": "future",
                "requestId": None,
                "payload": {},
            },
            "unknown-message-type",
        ),
        (
            {
                "version": 0,
                "type": "health",
                "requestId": None,
                "payload": {},
            },
            "invalid-frame-version",
        ),
        (
            {
                "version": True,
                "type": "health",
                "requestId": None,
                "payload": {},
            },
            "invalid-frame-version",
        ),
        (
            {
                "version": 1,
                "type": "health",
                "requestId": "",
                "payload": {},
            },
            "invalid-request-id",
        ),
        (
            {
                "version": 1,
                "type": "health",
                "requestId": None,
                "payload": [],
            },
            "invalid-payload",
        ),
    ],
)
def test_decoder_enforces_the_strict_frame_contract(document, code):
    with pytest.raises(IPCProtocolError) as excinfo:
        decode_frame(raw_frame(document))
    assert excinfo.value.code == code


def test_encoder_rejects_invalid_types_ids_and_non_json_payloads():
    with pytest.raises(IPCProtocolError, match="unknown-message-type"):
        encode_frame(IPCFrame(1, "health", None, {}))
    with pytest.raises(IPCProtocolError, match="invalid-request-id"):
        encode_frame(IPCFrame(1, WorkerMessageType.HEALTH, "x" * 129, {}))
    with pytest.raises(IPCProtocolError, match="invalid-payload"):
        encode_frame(IPCFrame(1, WorkerMessageType.RESULT, None, {"value": float("nan")}))
    with pytest.raises(IPCProtocolError, match="invalid-payload"):
        encode_frame(IPCFrame(1, WorkerMessageType.RESULT, None, {"value": object()}))


def test_handshake_frame_round_trips_a_validated_offer():
    expected = offer()
    frame = handshake_frame(expected)

    assert frame.version == FRAME_FORMAT_VERSION
    assert frame.type is WorkerMessageType.HELLO
    assert frame.request_id is None
    assert frame.payload["capabilities"] == ["cancel", "progress"]
    assert parse_handshake(decode_frame(encode_frame(frame))) == expected


@pytest.mark.parametrize(
    ("frame", "code"),
    [
        (IPCFrame(2, WorkerMessageType.HELLO, None, {}), "frame-version-incompatible"),
        (IPCFrame(1, WorkerMessageType.HEALTH, None, {}), "invalid-handshake"),
        (IPCFrame(1, WorkerMessageType.HELLO, "job", {}), "invalid-handshake"),
        (IPCFrame(1, WorkerMessageType.HELLO, None, {}), "invalid-handshake"),
        (
            IPCFrame(
                1,
                WorkerMessageType.HELLO,
                None,
                {
                    "pluginId": "echo",
                    "minimumVersion": 1,
                    "maximumVersion": 1,
                    "capabilities": "health",
                },
            ),
            "invalid-handshake",
        ),
    ],
)
def test_parser_rejects_invalid_initial_handshake_frames(frame, code):
    with pytest.raises(IPCProtocolError) as excinfo:
        parse_handshake(frame)
    assert excinfo.value.code == code


@pytest.mark.parametrize(
    ("candidate", "code"),
    [
        (offer(plugin_id="../echo"), "invalid-plugin-id"),
        (offer(minimum=True), "invalid-protocol-range"),
        (offer(minimum=0), "invalid-protocol-range"),
        (offer(minimum=3, maximum=2), "invalid-protocol-range"),
        (offer(capabilities=frozenset({""})), "invalid-capabilities"),
        (offer(capabilities=frozenset({"x" * 65})), "invalid-capabilities"),
        (offer(capabilities=frozenset(str(index) for index in range(33))), "invalid-capabilities"),
    ],
)
def test_handshake_offer_validation_is_bounded(candidate, code):
    with pytest.raises(IPCProtocolError) as excinfo:
        handshake_frame(candidate)
    assert excinfo.value.code == code


def test_negotiation_selects_the_highest_common_version_and_capabilities():
    service = offer(minimum=1, maximum=4, capabilities=frozenset({"cancel", "progress"}))
    worker = offer(minimum=2, maximum=3, capabilities=frozenset({"progress", "health"}))

    assert negotiate_handshake(service, worker) == HandshakeAgreement(
        "echo-plugin",
        3,
        frozenset({"progress"}),
    )


def test_negotiation_rejects_identity_substitution_and_disjoint_versions():
    with pytest.raises(IPCProtocolError) as identity:
        negotiate_handshake(offer(), offer(plugin_id="other-plugin"))
    assert identity.value.code == "plugin-identity-mismatch"

    with pytest.raises(IPCProtocolError) as version:
        negotiate_handshake(offer(minimum=1, maximum=2), offer(minimum=3, maximum=4))
    assert version.value.code == "protocol-version-incompatible"


class PeerWriter:
    def __init__(self, peer: asyncio.StreamReader):
        self.peer = peer
        self.writes = []
        self.drains = 0

    def write(self, data):
        assert isinstance(data, bytes)
        self.writes.append(data)
        self.peer.feed_data(data)

    async def drain(self):
        self.drains += 1


def test_service_and_worker_complete_the_same_handshake_over_streams():
    async def scenario():
        service_reader = asyncio.StreamReader()
        worker_reader = asyncio.StreamReader()
        service_writer = PeerWriter(worker_reader)
        worker_writer = PeerWriter(service_reader)
        service_offer = offer(minimum=1, maximum=4)
        worker_offer = offer(minimum=2, maximum=3)

        service_result, worker_result = await asyncio.gather(
            perform_service_handshake(service_reader, service_writer, service_offer),
            perform_worker_handshake(worker_reader, worker_writer, worker_offer),
        )

        assert (
            service_result
            == worker_result
            == HandshakeAgreement(
                "echo-plugin",
                3,
                frozenset({"cancel", "progress"}),
            )
        )
        assert service_writer.drains == worker_writer.drains == 1
        assert len(service_writer.writes) == len(worker_writer.writes) == 1

    asyncio.run(scenario())


def test_async_reader_rejects_truncated_and_oversized_frames():
    async def scenario():
        truncated = asyncio.StreamReader()
        truncated.feed_data(b"\x00\x00")
        truncated.feed_eof()
        with pytest.raises(IPCProtocolError) as missing_header:
            await read_frame(truncated)
        assert missing_header.value.code == "truncated-frame"

        oversized = asyncio.StreamReader()
        oversized.feed_data(struct.pack("!I", 11))
        with pytest.raises(IPCProtocolError) as too_large:
            await read_frame(oversized, max_frame_bytes=10)
        assert too_large.value.code == "frame-too-large"

        missing_body = asyncio.StreamReader()
        missing_body.feed_data(struct.pack("!I", 3) + b"{}")
        missing_body.feed_eof()
        with pytest.raises(IPCProtocolError) as body:
            await read_frame(missing_body)
        assert body.value.code == "truncated-frame"

    asyncio.run(scenario())


def test_async_writer_emits_one_decodable_frame_and_drains():
    async def scenario():
        reader = asyncio.StreamReader()
        writer = PeerWriter(reader)
        frame = IPCFrame(1, WorkerMessageType.CANCEL, "job-1", {"reason": "shutdown"})

        await write_frame(writer, frame)

        assert writer.drains == 1
        assert await read_frame(reader) == frame

    asyncio.run(scenario())


def test_execute_progress_and_result_frames_round_trip_exact_domain_values():
    request = PluginRequest(
        "job-1",
        "event-extraction",
        "manual",
        {"sources": ["/private/event.txt"]},
        10,
        20,
    )
    progress = PluginProgress("job-1", "extract", 0.5, "", 12)
    result = PluginResult(
        "job-1",
        PluginResultStatus.SUCCEEDED,
        {"events": []},
        "done",
        15,
    )

    assert parse_execute(decode_frame(encode_frame(execute_frame(request)))) == request
    assert parse_progress(decode_frame(encode_frame(progress_frame(progress)))) == progress
    assert parse_result(decode_frame(encode_frame(result_frame(result)))) == result


@pytest.mark.parametrize(
    ("frame", "parser", "code"),
    [
        (IPCFrame(1, WorkerMessageType.HEALTH, "job", {}), parse_execute, "invalid-execute"),
        (IPCFrame(1, WorkerMessageType.EXECUTE, None, {}), parse_execute, "invalid-execute"),
        (IPCFrame(1, WorkerMessageType.HEALTH, "job", {}), parse_progress, "invalid-progress"),
        (IPCFrame(1, WorkerMessageType.PROGRESS, None, {}), parse_progress, "invalid-progress"),
        (IPCFrame(1, WorkerMessageType.HEALTH, "job", {}), parse_result, "invalid-result"),
        (IPCFrame(1, WorkerMessageType.RESULT, None, {}), parse_result, "invalid-result"),
    ],
)
def test_execution_frame_parsers_require_correlated_message_types(frame, parser, code):
    with pytest.raises(IPCProtocolError) as error:
        parser(frame)
    assert error.value.code == code
    assert error.value.detail == f"expected a correlated {code.removeprefix('invalid-')} frame"


@pytest.mark.parametrize(
    ("invalid_request", "detail"),
    [
        (object(), "request must be a PluginRequest"),
        (
            PluginRequest("", "event-extraction", "manual", {}, 1, None),
            "request job id is invalid",
        ),
        (
            PluginRequest("job", "Bad", "manual", {}, 1, None),
            "request plugin id is invalid",
        ),
        (
            PluginRequest("job", "event-extraction", "", {}, 1, None),
            "request trigger is invalid",
        ),
        (
            PluginRequest("job", "event-extraction", "manual", [], 1, None),
            "request payload must be an object",
        ),
        (
            PluginRequest("job", "event-extraction", "manual", {}, True, None),
            "request timestamp is invalid",
        ),
        (
            PluginRequest("job", "event-extraction", "manual", {}, 2, 1),
            "request deadline is invalid",
        ),
    ],
)
def test_execute_frame_rejects_invalid_request_boundaries(invalid_request, detail):
    with pytest.raises(IPCProtocolError) as error:
        execute_frame(invalid_request)
    assert (error.value.code, error.value.detail) == ("invalid-execute", detail)


@pytest.mark.parametrize("job_id", ["j", "j" * 128])
@pytest.mark.parametrize("trigger", ["m", "m" * 80])
def test_execute_frame_accepts_exact_request_boundaries(job_id, trigger):
    request = PluginRequest(job_id, "event-extraction", trigger, {}, 1, 1)
    assert parse_execute(execute_frame(request)) == request


@pytest.mark.parametrize(
    ("progress", "detail"),
    [
        (object(), "progress must be PluginProgress"),
        (PluginProgress("", "stage", 0.5, "", 1), "progress job id is invalid"),
        (PluginProgress("job", "", 0.5, "", 1), "progress fields are invalid"),
        (PluginProgress("job", "stage", True, "", 1), "progress fields are invalid"),
        (
            PluginProgress("job", "stage", float("nan"), "", 1),
            "progress fields are invalid",
        ),
        (PluginProgress("job", "stage", 1.1, "", 1), "progress fields are invalid"),
        (
            PluginProgress("job", "stage", 0.5, "x" * 1025, 1),
            "progress fields are invalid",
        ),
        (PluginProgress("job", "stage", 0.5, "", True), "progress fields are invalid"),
    ],
)
def test_progress_frame_rejects_invalid_boundaries(progress, detail):
    with pytest.raises(IPCProtocolError) as error:
        progress_frame(progress)
    assert (error.value.code, error.value.detail) == ("invalid-progress", detail)


@pytest.mark.parametrize("fraction", [0, 1])
@pytest.mark.parametrize("stage", ["s", "s" * 80])
def test_progress_frame_accepts_exact_boundaries(fraction, stage):
    progress = PluginProgress("j" * 128, stage, fraction, "x" * 1024, 1)
    assert parse_progress(progress_frame(progress)) == progress


@pytest.mark.parametrize(
    ("result", "detail"),
    [
        (object(), "result must be PluginResult"),
        (
            PluginResult("", PluginResultStatus.SUCCEEDED, {}, "", 1),
            "result job id is invalid",
        ),
        (PluginResult("job", "succeeded", {}, "", 1), "result fields are invalid"),
        (
            PluginResult("job", PluginResultStatus.SUCCEEDED, [], "", 1),
            "result fields are invalid",
        ),
        (
            PluginResult("job", PluginResultStatus.SUCCEEDED, {}, "x" * 2049, 1),
            "result fields are invalid",
        ),
        (
            PluginResult("job", PluginResultStatus.SUCCEEDED, {}, "", True),
            "result fields are invalid",
        ),
    ],
)
def test_result_frame_rejects_invalid_boundaries(result, detail):
    with pytest.raises(IPCProtocolError) as error:
        result_frame(result)
    assert (error.value.code, error.value.detail) == ("invalid-result", detail)


@pytest.mark.parametrize("job_id", ["j", "j" * 128])
def test_result_frame_accepts_exact_boundaries(job_id):
    result = PluginResult(job_id, PluginResultStatus.SUCCEEDED, {}, "x" * 2048, 1)
    assert parse_result(result_frame(result)) == result


def test_a_result_that_fits_is_still_exactly_one_frame():
    result = PluginResult("job", PluginResultStatus.SUCCEEDED, {"ok": True}, "", 1)

    assert result_frames(result) == (result_frame(result),)


@given(st.text(min_size=1, max_size=32), st.integers(min_value=200, max_value=4096))
def test_a_result_too_large_for_one_frame_survives_the_round_trip(value, frame_bytes):
    """The answer is not the place to enforce a size: an oversized result used
    to be discarded whole and to cost a model reload on the next request."""
    result = PluginResult("job", PluginResultStatus.SUCCEEDED, {"value": value * 4096}, "", 1)

    frames = result_frames(result, max_frame_bytes=frame_bytes)

    assert len(frames) > 1
    assert all(
        len(encode_frame(frame, max_frame_bytes=frame_bytes)) <= frame_bytes for frame in frames
    )
    parts = []
    for index, frame in enumerate(frames):
        body, final = parse_result_chunk(frame, index)
        parts.append(body)
        assert final is (index == len(frames) - 1)
    assert assemble_result_chunks("job", parts) == result


def test_result_chunks_are_refused_out_of_order_or_malformed():
    frames = result_frames(
        PluginResult("job", PluginResultStatus.SUCCEEDED, {"value": "x" * 8192}, "", 1),
        max_frame_bytes=512,
    )

    with pytest.raises(IPCProtocolError, match="out of order"):
        parse_result_chunk(frames[1], 0)
    with pytest.raises(IPCProtocolError, match="do not match the contract"):
        parse_result_chunk(
            IPCFrame(FRAME_FORMAT_VERSION, WorkerMessageType.RESULT_CHUNK, "job", {}), 0
        )
    with pytest.raises(IPCProtocolError, match="expected a correlated result chunk"):
        parse_result_chunk(
            result_frame(PluginResult("job", PluginResultStatus.SUCCEEDED, {}, "", 1)), 0
        )
    with pytest.raises(IPCProtocolError, match="not valid JSON"):
        assemble_result_chunks("job", ("{",))
    with pytest.raises(IPCProtocolError, match="not an object"):
        assemble_result_chunks("job", ("[]",))


def test_execution_parsers_reject_wrong_field_sets_and_status():
    with pytest.raises(IPCProtocolError, match="execute fields") as execute:
        parse_execute(IPCFrame(1, WorkerMessageType.EXECUTE, "job", {}))
    assert (execute.value.code, execute.value.detail) == (
        "invalid-execute",
        "execute fields do not match the contract",
    )
    with pytest.raises(IPCProtocolError, match="progress fields") as progress:
        parse_progress(IPCFrame(1, WorkerMessageType.PROGRESS, "job", {}))
    assert (progress.value.code, progress.value.detail) == (
        "invalid-progress",
        "progress fields do not match the contract",
    )
    malformed = IPCFrame(
        1,
        WorkerMessageType.RESULT,
        "job",
        {"status": "unknown", "output": {}, "detail": "", "completedAt": 1},
    )
    with pytest.raises(IPCProtocolError, match="result status") as status:
        parse_result(malformed)
    assert (status.value.code, status.value.detail) == (
        "invalid-result",
        "result status is invalid",
    )
    with pytest.raises(IPCProtocolError) as fields:
        parse_result(IPCFrame(1, WorkerMessageType.RESULT, "job", {}))
    assert (fields.value.code, fields.value.detail) == (
        "invalid-result",
        "result fields do not match the contract",
    )


json_scalars = (
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text()
)
json_values = st.recursive(
    json_scalars,
    lambda children: (
        st.lists(children, max_size=4) | st.dictionaries(st.text(max_size=20), children, max_size=4)
    ),
    max_leaves=12,
)


@given(payload=st.dictionaries(st.text(max_size=20), json_values, max_size=5))
def test_arbitrary_finite_json_payloads_round_trip(payload):
    frame = IPCFrame(1, WorkerMessageType.COLLECT, "job", payload)
    assert decode_frame(encode_frame(frame)) == frame
