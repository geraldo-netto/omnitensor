"""The control socket: framing, dispatch, identity, and lifecycle."""

from __future__ import annotations

import asyncio
import json
import os
import stat
import struct

import msgpack
import pytest

from omnitensor.callers import current_sender
from omnitensor.contract import RUNTIME_METHODS
from omnitensor.socket_transport import (
    MAX_FRAME_BYTES,
    ControlSocketError,
    SocketControlTransport,
    call_control,
    default_socket_path,
    encode_frame,
    read_frame,
)


class RecordingHandler:
    """A :class:`RuntimeHandler` that records the text each method received."""

    def __init__(self, replies=None, error=None):
        self.calls = []
        self.replies = dict(replies or {})
        self.error = error
        self.senders = []

    async def _answer(self, method, text):
        self.calls.append((method, text))
        self.senders.append(current_sender())
        if self.error is not None:
            raise self.error
        return json.dumps(self.replies.get(method, {"answered": method}))

    async def apply_command_text(self, text):
        return await self._answer("apply-command", text)

    async def submit_job_text(self, text):
        return await self._answer("submit-job", text)

    async def cancel_job_text(self, text):
        return await self._answer("cancel-job", text)

    async def job_result_text(self, text):
        return await self._answer("get-job-result", text)

    def describe_plugins_text(self):
        self.calls.append(("describe-plugins", ""))
        self.senders.append(current_sender())
        if self.error is not None:
            raise self.error
        return json.dumps(self.replies.get("describe-plugins", {"answered": "describe-plugins"}))

    def describe_contract_text(self):
        self.calls.append(("describe-contract", ""))
        self.senders.append(current_sender())
        return json.dumps(self.replies.get("describe-contract", {"answered": "describe-contract"}))


@pytest.fixture
def socket_path(tmp_path):
    return tmp_path / "run" / "control.sock"


def serve(handler, socket_path):
    transport = SocketControlTransport(socket_path=socket_path)

    class Running:
        async def __aenter__(self):
            await transport.start(handler)
            return transport

        async def __aexit__(self, *_exc):
            await transport.stop()

    return Running()


def test_the_default_path_prefers_the_override_then_the_runtime_dir():
    assert default_socket_path({"OMNITENSOR_CONTROL_SOCKET": "/x/y.sock"}) == __import__(
        "pathlib"
    ).Path("/x/y.sock")
    assert default_socket_path({"XDG_RUNTIME_DIR": "/run/user/7"}) == __import__("pathlib").Path(
        "/run/user/7/omnitensor/control.sock"
    )
    assert default_socket_path({}) == __import__("pathlib").Path(
        f"/run/user/{os.getuid()}/omnitensor/control.sock"
    )


def test_a_frame_round_trips_through_the_codec():
    async def scenario():
        reader = asyncio.StreamReader()
        reader.feed_data(encode_frame({"version": 1, "id": 7, "method": "x", "params": {}}))
        reader.feed_eof()
        assert await read_frame(reader) == {"version": 1, "id": 7, "method": "x", "params": {}}
        assert await read_frame(reader) is None

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        (b"\x00\x00\x00\x00", "frame-length-invalid"),
        (struct.pack(">I", MAX_FRAME_BYTES + 1), "frame-length-invalid"),
        (struct.pack(">I", 4) + b"\xc1\xc1\xc1\xc1", "frame-undecodable"),
        (struct.pack(">I", 1) + msgpack.packb(7), "frame-undecodable"),
        (b"\x00\x00", "frame-truncated"),
        (struct.pack(">I", 10) + b"abc", "frame-truncated"),
    ],
)
def test_a_broken_stream_is_an_error_not_a_guess(payload, code):
    async def scenario():
        reader = asyncio.StreamReader()
        reader.feed_data(payload)
        reader.feed_eof()
        with pytest.raises(ControlSocketError) as captured:
            await read_frame(reader)
        assert captured.value.code == code

    asyncio.run(scenario())


def test_a_document_larger_than_the_frame_cap_is_refused_before_the_wire():
    with pytest.raises(ControlSocketError) as captured:
        encode_frame({"blob": b"\x00" * (MAX_FRAME_BYTES + 1)})
    assert captured.value.code == "frame-too-large"


def test_the_transport_serves_a_request_end_to_end(socket_path):
    handler = RecordingHandler(replies={"apply-command": {"status": "rejected", "revision": 4}})

    async def scenario():
        async with serve(handler, socket_path):
            return await call_control(
                "apply-command", {"version": 2, "commandId": "c-1"}, socket_path=socket_path
            )

    result = asyncio.run(scenario())

    assert result == {"status": "rejected", "revision": 4}
    assert handler.calls == [("apply-command", '{"version":2,"commandId":"c-1"}')]


@pytest.mark.parametrize("method", sorted(set(RUNTIME_METHODS)))
def test_every_announced_method_dispatches(method, socket_path):
    handler = RecordingHandler()

    async def scenario():
        async with serve(handler, socket_path):
            return await call_control(method, {}, socket_path=socket_path)

    assert asyncio.run(scenario()) == {"answered": method}
    assert handler.calls[0][0] == method


def test_the_socket_is_created_private_and_removed_on_stop(socket_path):
    async def scenario():
        async with serve(RecordingHandler(), socket_path):
            mode = socket_path.stat().st_mode
            assert stat.S_ISSOCK(mode)
            assert stat.S_IMODE(mode) == 0o600
        assert not socket_path.exists()

    asyncio.run(scenario())


def test_a_stale_socket_file_is_replaced_on_start(socket_path):
    socket_path.parent.mkdir(parents=True)
    socket_path.write_bytes(b"")

    async def scenario():
        async with serve(RecordingHandler(), socket_path):
            return await call_control("describe-contract", {}, socket_path=socket_path)

    assert asyncio.run(scenario()) == {"answered": "describe-contract"}


def test_an_invalid_envelope_is_refused_with_the_salvaged_id(socket_path):
    async def scenario():
        async with serve(RecordingHandler(), socket_path):
            reader, writer = await asyncio.open_unix_connection(str(socket_path))
            writer.write(encode_frame({"version": 1, "id": 9, "method": "Bad Name"}))
            await writer.drain()
            reply = await read_frame(reader)
            writer.close()
            return reply

    reply = asyncio.run(scenario())

    assert reply["id"] == 9
    assert reply["error"]["code"] == "request-invalid"


def test_an_unknown_method_is_named_as_such(socket_path):
    async def scenario():
        async with serve(RecordingHandler(), socket_path):
            await call_control("no-such-method", {}, socket_path=socket_path)

    with pytest.raises(ControlSocketError) as captured:
        asyncio.run(scenario())
    assert captured.value.code == "method-unknown"


def test_params_msgpack_can_carry_but_json_cannot_are_refused(socket_path):
    """The method schemas are JSON documents; binary keys never reach them."""

    async def scenario():
        async with serve(RecordingHandler(), socket_path):
            await call_control("submit-job", {"blob": b"\x00\x01"}, socket_path=socket_path)

    with pytest.raises(ControlSocketError) as captured:
        asyncio.run(scenario())
    assert captured.value.code == "request-invalid"


def test_a_handler_failure_is_an_internal_error_not_a_dead_connection(socket_path):
    handler = RecordingHandler(error=RuntimeError("scheduler exploded"))

    async def scenario():
        async with serve(handler, socket_path):
            try:
                await call_control("submit-job", {}, socket_path=socket_path)
            except ControlSocketError as error:
                return error

    error = asyncio.run(scenario())

    assert error.code == "internal-error"
    assert "scheduler exploded" in error.detail


@pytest.mark.parametrize("method", ["submit-job", "describe-plugins"])
def test_a_guard_refusal_is_an_envelope_error_not_the_wrong_method_document(method, socket_path):
    """The guard refuses *before* the method runs, so there is no method result.

    Answering `submit-job` with a runtime-refusal document made a client
    validating the reply against runtime-job-acknowledgement fail with no
    envelope error to branch on -- and `describe-plugins` answer with a
    refusal where the applet expects a plugin inventory.
    """
    from omnitensor.guard import GuardRefusedError

    handler = RecordingHandler(
        error=GuardRefusedError("rate-limit-exceeded", "too many calls", method=method)
    )

    async def scenario():
        async with serve(handler, socket_path):
            try:
                await call_control(method, {}, socket_path=socket_path)
            except ControlSocketError as error:
                return error
            return None

    error = asyncio.run(scenario())

    assert error is not None
    assert error.code == "rate-limit-exceeded"
    assert error.detail == "too many calls"


def test_each_connection_is_stamped_with_the_peer_uid(socket_path):
    handler = RecordingHandler()

    async def scenario():
        async with serve(handler, socket_path):
            await call_control("describe-plugins", {}, socket_path=socket_path)
            await call_control("describe-plugins", {}, socket_path=socket_path)

    asyncio.run(scenario())

    uid = os.getuid()
    assert [sender.split(":")[:2] for sender in handler.senders] == [["peer", str(uid)]] * 2
    # Serials differ per connection, for audit; ownership ignores them.
    assert len(set(handler.senders)) == 2


def test_one_connection_can_carry_several_requests_in_order(socket_path):
    handler = RecordingHandler()

    async def scenario():
        async with serve(handler, socket_path):
            reader, writer = await asyncio.open_unix_connection(str(socket_path))
            for index in range(3):
                writer.write(
                    encode_frame(
                        {"version": 1, "id": index, "method": "describe-contract", "params": {}}
                    )
                )
            await writer.drain()
            replies = [await read_frame(reader) for _ in range(3)]
            writer.close()
            return replies

    replies = asyncio.run(scenario())

    assert [reply["id"] for reply in replies] == [0, 1, 2]
    assert all(reply["result"] == {"answered": "describe-contract"} for reply in replies)
    assert len(set(handler.senders)) == 1


def test_the_client_times_out_against_a_silent_service(socket_path):
    async def scenario():
        socket_path.parent.mkdir(parents=True)
        writers = []

        async def silent(_reader, writer):
            writers.append(writer)

        server = await asyncio.start_unix_server(silent, path=str(socket_path))
        try:
            await call_control("describe-contract", {}, socket_path=socket_path, timeout_s=0.2)
        finally:
            server.close()
            for writer in writers:
                writer.close()
            await server.wait_closed()

    with pytest.raises(ControlSocketError) as captured:
        asyncio.run(scenario())
    assert captured.value.code == "timeout"


def test_the_client_refuses_a_nonpositive_timeout(socket_path):
    with pytest.raises(ControlSocketError) as captured:
        asyncio.run(call_control("describe-contract", {}, socket_path=socket_path, timeout_s=0))
    assert captured.value.code == "timeout-invalid"


def test_a_missing_service_raises_the_underlying_connection_error(socket_path):
    with pytest.raises((FileNotFoundError, ConnectionRefusedError)):
        asyncio.run(call_control("describe-contract", {}, socket_path=socket_path))


def test_stop_is_not_held_hostage_by_an_open_connection(socket_path):
    """Python 3.12's wait_closed() waits for live connections; stop closes them."""

    async def scenario():
        transport = SocketControlTransport(socket_path=socket_path)
        await transport.start(RecordingHandler())
        _reader, writer = await asyncio.open_unix_connection(str(socket_path))
        try:
            await asyncio.wait_for(transport.stop(), timeout=5)
        finally:
            writer.close()
        assert not socket_path.exists()

    asyncio.run(scenario())


def test_a_second_instance_refuses_rather_than_stealing_the_socket(socket_path):
    """Two instances would publish to the same snapshot path."""

    async def scenario():
        async with serve(RecordingHandler(), socket_path):
            second = SocketControlTransport(socket_path=socket_path)
            with pytest.raises(RuntimeError, match="already served"):
                await second.start(RecordingHandler())
        # The refused instance took nothing down with it.
        assert not socket_path.exists()

    asyncio.run(scenario())


def test_stop_of_a_refused_instance_leaves_the_live_socket_alone(socket_path):
    async def scenario():
        async with serve(RecordingHandler(), socket_path):
            second = SocketControlTransport(socket_path=socket_path)
            with pytest.raises(RuntimeError, match="already served"):
                await second.start(RecordingHandler())
            await second.stop()
            # The running instance still answers.
            return await call_control("describe-contract", {}, socket_path=socket_path)

    assert asyncio.run(scenario()) == {"answered": "describe-contract"}


def test_stop_without_start_is_a_no_op(socket_path):
    asyncio.run(SocketControlTransport(socket_path=socket_path).stop())


def test_a_result_too_large_to_send_is_named_not_dropped(socket_path):
    """An oversized reply is diagnosable; the connection is not just cut.

    The framing violation the connection is dropped for is inbound. For a reply
    the boundary is intact, so a client that asked for more than a frame can
    hold must be able to tell that from a daemon that died mid-answer.
    """
    handler = RecordingHandler(replies={"get-job-result": {"text": "x" * (MAX_FRAME_BYTES + 1)}})

    async def scenario():
        async with serve(handler, socket_path):
            await call_control("get-job-result", {"jobId": "j"}, socket_path=socket_path)

    with pytest.raises(ControlSocketError) as captured:
        asyncio.run(scenario())
    assert captured.value.code == "result-too-large"
    assert str(MAX_FRAME_BYTES) in captured.value.detail


def test_the_lock_refuses_a_racer_even_when_the_served_check_lies(socket_path, monkeypatch):
    """Startup is serialized by the flock, not by the probe (OMNI-0604).

    Two starts could both pass _is_served() before either bound; the loser
    then unlinked the winner's just-bound socket, and its stop() later
    unlinked the replacement too. The probe is forced to lie here — the
    exact race window — and the lock alone must still refuse the second
    instance without touching the first's socket.
    """

    async def scenario():
        async with serve(RecordingHandler(), socket_path):
            second = SocketControlTransport(socket_path=socket_path)

            async def nothing_served():
                return False

            monkeypatch.setattr(second, "_is_served", nothing_served)
            with pytest.raises(RuntimeError, match="already served"):
                await second.start(RecordingHandler())
            await second.stop()
            # The first instance's socket is untouched and still answers.
            return await call_control("describe-contract", {}, socket_path=socket_path)

    assert asyncio.run(scenario()) == {"answered": "describe-contract"}
