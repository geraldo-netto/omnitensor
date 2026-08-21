"""The control socket: framed msgpack over a per-user unix-domain socket.

This replaced the session D-Bus transport outright.  What D-Bus provided —
per-user reachability, caller credentials, a method vocabulary — the socket
provides more directly: the socket lives under ``$XDG_RUNTIME_DIR``, which is
per-user by construction; ``SO_PEERCRED`` stamps every connection with the
peer's uid at accept time, with no daemon round trip to ask; and the method
vocabulary is :data:`omnitensor.contract.RUNTIME_METHODS`, announced by
``describe-contract`` like every other part of the wire contract.

The wire format is the smallest thing that carries a document: a four-byte
big-endian length prefix, then one msgpack map, validated against
``control-request.schema.json`` before anything is dispatched.  msgpack was
measured against framed JSON on this system's real payloads
(``scripts/benchmark-transport-codecs.py``) and won every race; its ``bin``
type is the difference between 44.9 microseconds and 52.7 milliseconds for an
inline image tensor, which is why there is no codec negotiation — one codec,
chosen where it matters most.

Envelope errors and method answers travel differently on purpose.  A method's
own rejection (a command the policy declines) is a *result* — the method ran,
and its answer is the rejection its own schema describes.  An ``error`` in the
reply envelope means the method never ran: the frame did not decode, the
request did not validate, the method does not exist, or the guard refused the
call at the boundary — a quota, an oversized payload, an asserted identity —
in which case the envelope error carries the guard's own stable code.  A
client can therefore branch on transport health without parsing method
documents, and validate every result against the method's own schema without
a second shape to allow for.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import itertools
import json
import os
import socket
import struct
from collections.abc import Awaitable, Callable
from pathlib import Path

import msgpack

from .callers import bind_sender
from .contract import RUNTIME_METHODS
from .guard import GuardRefusedError
from .ports import RuntimeHandler
from .registry import validate_document
from .stable_error import StableError

CONTROL_PROTOCOL_VERSION = 1
CONTROL_REQUEST_SCHEMA = "control-request.schema.json"
CONTROL_REPLY_SCHEMA = "control-reply.schema.json"
SOCKET_ENV = "OMNITENSOR_CONTROL_SOCKET"
SOCKET_DIR_NAME = "omnitensor"
SOCKET_FILE_NAME = "control.sock"
# One frame must hold the largest admitted request plus envelope overhead.
# The guard admits at most 256 KiB of method text today; the headroom above
# that is for inline binary tensors, the payload class msgpack was chosen for.
MAX_FRAME_BYTES = 16 * 1024 * 1024
_LENGTH_PREFIX = struct.Struct(">I")
_PEERCRED = struct.Struct("3i")


class ControlSocketError(StableError, Exception):
    """A control-socket failure with a stable code a client can branch on."""


def default_socket_path(environ: dict | None = None) -> Path:
    """Where the control socket lives unless ``OMNITENSOR_CONTROL_SOCKET`` says.

    ``$XDG_RUNTIME_DIR`` is the honest default: it is per-user, mode 0700, and
    cleared with the session, so a stale socket cannot outlive the login that
    created it.  The ``/run/user`` fallback is the same directory under the
    name systemd gives it when the variable is not exported to this process.
    """
    env = os.environ if environ is None else environ
    override = env.get(SOCKET_ENV, "")
    if override:
        return Path(override)
    runtime_dir = env.get("XDG_RUNTIME_DIR", "") or f"/run/user/{os.getuid()}"
    return Path(runtime_dir) / SOCKET_DIR_NAME / SOCKET_FILE_NAME


def encode_frame(document: dict) -> bytes:
    """One wire frame: length prefix plus the msgpack encoding of ``document``."""
    payload = msgpack.packb(document, use_bin_type=True)
    if len(payload) > MAX_FRAME_BYTES:
        raise ControlSocketError(
            "frame-too-large", f"{len(payload)} bytes exceeds {MAX_FRAME_BYTES}"
        )
    return _LENGTH_PREFIX.pack(len(payload)) + payload


async def read_frame(reader: asyncio.StreamReader) -> dict | None:
    """The next decoded frame, or ``None`` on a clean end of stream.

    Clean means the peer closed between frames; a stream that ends inside a
    frame is an error, because silently dropping a half-received request would
    make a crashed peer indistinguishable from a quiet one.
    """
    try:
        header = await reader.readexactly(_LENGTH_PREFIX.size)
    except asyncio.IncompleteReadError as error:
        if not error.partial:
            return None
        raise ControlSocketError("frame-truncated", "stream ended inside a length prefix") from None
    (length,) = _LENGTH_PREFIX.unpack(header)
    if length == 0 or length > MAX_FRAME_BYTES:
        raise ControlSocketError("frame-length-invalid", f"declared {length} bytes")
    try:
        payload = await reader.readexactly(length)
    except asyncio.IncompleteReadError:
        raise ControlSocketError("frame-truncated", "stream ended inside a frame") from None
    try:
        document = msgpack.unpackb(payload)
    except Exception:  # noqa: BLE001 - any undecodable frame gets the same verdict
        raise ControlSocketError("frame-undecodable", "frame is not valid msgpack") from None
    if not isinstance(document, dict):
        raise ControlSocketError("frame-undecodable", "frame is not a map")
    return document


def _error_reply(request_id: int, code: str, detail: str) -> dict:
    message = detail.strip() or code
    return {
        "version": CONTROL_PROTOCOL_VERSION,
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def _salvaged_id(request: dict) -> int:
    """The id to echo on a request that failed validation, if one is usable."""
    candidate = request.get("id")
    if isinstance(candidate, bool) or not isinstance(candidate, int):
        return 0
    return candidate if 0 <= candidate <= 4294967295 else 0


def _dispatch_table(handler: RuntimeHandler) -> dict[str, Callable[[str], Awaitable[str]]]:
    """Method name to handler call, checked against the announced contract.

    Built here and asserted against :data:`RUNTIME_METHODS` so a method added
    to the contract without a dispatch entry — or dispatched without being
    announced — fails at startup rather than as a live client's surprise.
    """

    async def describe_plugins(_text: str) -> str:
        return handler.describe_plugins_text()

    async def describe_contract(_text: str) -> str:
        return handler.describe_contract_text()

    table: dict[str, Callable[[str], Awaitable[str]]] = {
        "apply-command": handler.apply_command_text,
        "submit-job": handler.submit_job_text,
        "cancel-job": handler.cancel_job_text,
        "get-job-result": handler.job_result_text,
        "describe-plugins": describe_plugins,
        "describe-contract": describe_contract,
    }
    if set(table) != set(RUNTIME_METHODS):
        raise RuntimeError(
            "control dispatch and RUNTIME_METHODS disagree: "
            f"{sorted(set(table) ^ set(RUNTIME_METHODS))}"
        )
    return table


class SocketControlTransport:
    """:class:`omnitensor.ports.ControlTransport` over the control socket."""

    def __init__(self, *, socket_path: Path | None = None):
        self._path = Path(socket_path) if socket_path is not None else default_socket_path()
        self._server: asyncio.Server | None = None
        self._dispatch: dict[str, Callable[[str], Awaitable[str]]] = {}
        # Serial numbers keep two connections from the same uid tellable
        # apart in audit output; ownership is scoped by uid alone.
        self._serials = itertools.count(1)
        self._connections: set[asyncio.StreamWriter] = set()
        self._bound = False
        self._lock = None

    @property
    def socket_path(self) -> Path:
        return self._path

    async def start(self, handler: RuntimeHandler) -> None:
        self._dispatch = _dispatch_table(handler)
        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        # The check/unlink/bind below was a TOCTOU (OMNI-0604): two starts
        # could both find nothing served, and the loser unlinked the winner's
        # just-bound socket — then its stop() unlinked the other's too. The
        # whole startup is now serialized by an flock on a sibling lock file,
        # held for the instance's lifetime: the kernel releases it when the
        # holder dies, so a stale lock cannot refuse anybody.
        lock = (self._path.parent / f"{self._path.name}.lock").open("a+b")
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            raise RuntimeError(
                f"{self._path} is already served; another omnitensor instance is running"
            ) from None
        except BaseException:
            lock.close()
            raise
        self._lock = lock
        # Held lock in hand, a socket that still answers is another instance
        # running without the lock discipline — refuse rather than fight it.
        # A socket file nobody answers is the residue of an instance that
        # died without stop(); bind() refuses an existing path, and the
        # unlink is safe because only this user can reach the directory.
        if self._path.exists() and await self._is_served():
            self._release_lock()
            raise RuntimeError(
                f"{self._path} is already served; another omnitensor instance is running"
            )
        with contextlib.suppress(FileNotFoundError):
            self._path.unlink()
        self._server = await asyncio.start_unix_server(self._serve, path=str(self._path))
        os.chmod(self._path, 0o600)
        self._bound = True

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            # wait_closed() waits for live connections too, and a client that
            # holds one open — the applet's poll loop, say — must not be able
            # to hold the whole service's shutdown hostage.
            for writer in tuple(self._connections):
                writer.close()
            await self._server.wait_closed()
            self._server = None
        # Unlink only a socket this instance bound: stop() on an instance that
        # was refused startup must not tear down the running instance's socket.
        if self._bound:
            self._bound = False
            with contextlib.suppress(FileNotFoundError):
                self._path.unlink()
        self._release_lock()

    def _release_lock(self) -> None:
        lock, self._lock = self._lock, None
        if lock is not None:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()

    async def _is_served(self) -> bool:
        """Whether something is accepting connections on the socket path."""
        try:
            _reader, writer = await asyncio.wait_for(
                asyncio.open_unix_connection(str(self._path)), timeout=2.0
            )
        except (OSError, TimeoutError):
            return False
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()
        return True

    def _peer_sender(self, writer: asyncio.StreamWriter) -> str:
        """The sender token for one accepted connection, from ``SO_PEERCRED``.

        The kernel stamps the credentials; nothing here trusts the peer's own
        claims.  A connection whose credentials cannot be read is anonymous
        rather than guessed at.
        """
        raw = writer.get_extra_info("socket")
        try:
            credentials = raw.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, _PEERCRED.size)
            _pid, uid, _gid = _PEERCRED.unpack(credentials)
        except (AttributeError, OSError, struct.error):
            return ""
        return "" if uid < 0 else f"peer:{uid}:{next(self._serials)}"

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        # Each connection is served in its own task, so the binding scopes the
        # identity to exactly this connection's requests and their children.
        bind_sender(self._peer_sender(writer))
        self._connections.add(writer)
        try:
            while True:
                request = await read_frame(reader)
                if request is None:
                    break
                writer.write(self._encoded_reply(await self._reply_for(request)))
                await writer.drain()
        except (ControlSocketError, ConnectionResetError, BrokenPipeError):
            # A peer that violates the framing gets a closed connection, not a
            # reply: there is no frame boundary left to write a reply into.
            pass
        finally:
            self._connections.discard(writer)
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    @staticmethod
    def _encoded_reply(reply: dict) -> bytes:
        """One reply frame, or an envelope error saying the result did not fit.

        An oversized *reply* is not a framing violation: the boundary is intact
        and the connection is healthy, so dropping it would leave the client
        unable to tell a result too big to send from a daemon that crashed.
        The substituted envelope names the cause and always fits, because it
        carries a bounded code and message and nothing of the result.
        """
        try:
            return encode_frame(reply)
        except ControlSocketError as oversized:
            if oversized.code != "frame-too-large":
                raise
            return encode_frame(_error_reply(reply["id"], "result-too-large", oversized.detail))

    async def _reply_for(self, request: dict) -> dict:
        violations = validate_document(CONTROL_REQUEST_SCHEMA, request)
        if violations:
            return _error_reply(_salvaged_id(request), "request-invalid", violations[0])
        request_id = request["id"]
        call = self._dispatch.get(request["method"])
        if call is None:
            return _error_reply(
                request_id, "method-unknown", f"this service does not export {request['method']}"
            )
        try:
            text = json.dumps(request["params"], separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError):
            # msgpack can carry what JSON cannot (binary keys, non-string
            # keys); the method schemas are JSON documents, so refuse here.
            return _error_reply(request_id, "request-invalid", "params are not a JSON document")
        try:
            result = json.loads(await call(text))
        except GuardRefusedError as refusal:
            # The guard refused before the method ran, so there is no method
            # document to answer with. `text()` validates the refusal against
            # runtime-refusal.schema.json on the way past, so a code outside
            # that vocabulary cannot reach the wire as if it were stable.
            document = json.loads(refusal.text())
            return _error_reply(request_id, document["code"], document["message"])
        except Exception as error:  # noqa: BLE001 - one request must not kill the connection
            return _error_reply(request_id, "internal-error", f"{type(error).__name__}: {error}")
        return {"version": CONTROL_PROTOCOL_VERSION, "id": request_id, "result": result}


async def call_control(
    method: str,
    params: dict | None = None,
    *,
    socket_path: Path | None = None,
    timeout_s: float = 30.0,
) -> dict:
    """One request over one connection: the client half of the protocol.

    Returns the method's result document, raises :class:`ControlSocketError`
    carrying the envelope error's code otherwise.  Connection setup and the
    reply share one deadline, so a hung service costs a caller ``timeout_s``
    rather than forever.
    """
    if timeout_s <= 0:
        raise ControlSocketError("timeout-invalid", "timeout_s must be positive")
    path = Path(socket_path) if socket_path is not None else default_socket_path()

    async def exchange() -> dict:
        reader, writer = await asyncio.open_unix_connection(str(path))
        try:
            request = {
                "version": CONTROL_PROTOCOL_VERSION,
                "id": 1,
                "method": method,
                "params": params if params is not None else {},
            }
            writer.write(encode_frame(request))
            await writer.drain()
            reply = await read_frame(reader)
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
        if reply is None:
            raise ControlSocketError("connection-closed", "the service closed without replying")
        violations = validate_document(CONTROL_REPLY_SCHEMA, reply)
        if violations:
            raise ControlSocketError("reply-invalid", violations[0])
        if reply["id"] != request["id"]:
            raise ControlSocketError("reply-mismatched", "reply answers a different request")
        error = reply.get("error")
        if error is not None:
            raise ControlSocketError(error["code"], error["message"])
        return reply["result"]

    try:
        return await asyncio.wait_for(exchange(), timeout_s)
    except TimeoutError:
        raise ControlSocketError("timeout", f"no reply within {timeout_s:g}s from {path}") from None
