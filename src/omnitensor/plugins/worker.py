"""Minimal stdio bootstrap for one separately installed workload plugin."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from importlib import metadata
from pathlib import Path
from typing import BinaryIO

from ..sdk import CancellationController, cancelled_result
from ..sdk.bootstrap import (
    BootstrapArtifact,
    PluginBootstrap,
    configure_plugin_bootstrap,
)
from .discovery import PLUGIN_ENTRY_POINT_GROUP
from .ipc import (
    DEFAULT_MAX_FRAME_BYTES,
    HandshakeAgreement,
    HandshakeOffer,
    IPCFrame,
    IPCProtocolError,
    WorkerMessageType,
    decode_frame,
    encode_frame,
    handshake_frame,
    negotiate_handshake,
    parse_execute,
    parse_handshake,
    progress_frame,
    ready_frame,
    result_frames,
)
from .offloop import run_off_loop
from .protocol import PluginContext, PluginProgress, PluginRequest, WorkloadPlugin
from .seccomp import confinement_error, install_filter

WORKER_CAPABILITIES = frozenset({"cancel", "execute", "health", "progress"})
WORKER_CANCEL_GRACE_SECONDS = 0.05
WORKER_SHUTDOWN_CANCEL_SECONDS = 0.05


class ExternalPluginLoadError(RuntimeError):
    """An installed entry point cannot be loaded as the validated identity."""


def load_external_plugin(
    plugin_id: str,
    entry_point_name: str,
    target: str,
    distribution_name: str,
    *,
    entry_points_provider: Callable[..., Iterable] = metadata.entry_points,
) -> WorkloadPlugin:
    """Load exactly one prevalidated entry point inside the worker process."""
    try:
        candidates = tuple(entry_points_provider(group=PLUGIN_ENTRY_POINT_GROUP))
    except Exception as error:
        raise ExternalPluginLoadError(
            f"entry-point enumeration failed: {type(error).__name__}"
        ) from error
    matches = [
        candidate
        for candidate in candidates
        if candidate.name == entry_point_name
        and candidate.value == target
        and candidate.dist is not None
        and candidate.dist.name == distribution_name
    ]
    if len(matches) != 1:
        raise ExternalPluginLoadError("installed entry-point identity changed")
    try:
        factory = matches[0].load()
        plugin = factory()
    except Exception as error:
        raise ExternalPluginLoadError(f"entry-point load failed: {type(error).__name__}") from error
    # WorkloadPlugin does not declare plugin_id, so isinstance() admits an
    # object without it; reading the attribute directly would surface a raw
    # AttributeError instead of the load error callers handle.
    if not isinstance(plugin, WorkloadPlugin) or getattr(plugin, "plugin_id", None) != plugin_id:
        raise ExternalPluginLoadError("entry point does not implement the declared plugin")
    return plugin


async def serve_worker_requests(
    plugin: WorkloadPlugin,
    reader: BinaryIO,
    writer: BinaryIO,
    *,
    minimum_protocol: int = 1,
    maximum_protocol: int = 1,
    permissions: frozenset[str] = frozenset(),
) -> HandshakeAgreement:
    """Serve executable requests while continuing to receive cancellation."""
    offer = HandshakeOffer(
        plugin.plugin_id,
        minimum_protocol,
        maximum_protocol,
        WORKER_CAPABILITIES,
    )
    service = parse_handshake(await run_off_loop(_read_frame, reader))
    agreement = negotiate_handshake(service, offer)
    _write_frame(writer, handshake_frame(offer))
    await plugin.start(PluginContext(plugin.plugin_id, agreement.protocol_version, {}, permissions))
    _write_frame(writer, ready_frame(plugin.plugin_id))
    active: dict[str, tuple[asyncio.Task, CancellationController]] = {}
    # The loop holds only weak references to tasks, so a grace timer nobody
    # keeps can be collected while it sleeps — and the hard cancel it exists to
    # deliver never arrives. That is exactly the plugin the grace period is for:
    # one that ignores its cooperative token.
    deadlines: set[asyncio.Task] = set()
    try:
        while True:
            try:
                frame = await run_off_loop(_read_frame, reader)
            except EOFError:
                break
            if frame.version != agreement.protocol_version:
                # The other end of the same check the service makes: a frame
                # in a version this worker never agreed to was parsed as
                # though its fields meant what this version says they mean.
                _write_error(
                    writer,
                    agreement.protocol_version,
                    frame.request_id,
                    "frame-version-incompatible",
                )
                continue
            if not await _handle_request_frame(
                plugin,
                frame,
                writer,
                agreement.protocol_version,
                active,
                deadlines,
            ):
                break
    finally:
        for deadline in tuple(deadlines):
            deadline.cancel()
        await _cancel_active_requests(active)
        await plugin.stop()
    return agreement


async def _handle_request_frame(
    plugin: WorkloadPlugin,
    frame: IPCFrame,
    writer: BinaryIO,
    protocol_version: int,
    active: dict[str, tuple[asyncio.Task, CancellationController]],
    deadlines: set[asyncio.Task],
) -> bool:
    if frame.type is WorkerMessageType.CANCEL and frame.request_id is None:
        return False
    if frame.type is WorkerMessageType.EXECUTE:
        try:
            request = parse_execute(frame, protocol_version=protocol_version)
            if request.plugin_id != plugin.plugin_id:
                raise IPCProtocolError(
                    "plugin-identity-mismatch", "execute request names another plugin"
                )
            if request.job_id in active:
                raise IPCProtocolError("duplicate-request", "request is already active")
        except IPCProtocolError as error:
            _write_error(writer, protocol_version, frame.request_id, error.code)
            return True
        token = CancellationController()
        task = asyncio.create_task(
            _execute_request(plugin, request, token, writer, protocol_version),
            name=f"omnitensor-plugin-request-{request.job_id}",
        )
        active[request.job_id] = (task, token)
        task.add_done_callback(
            lambda _task, request_id=request.job_id: active.pop(request_id, None)
        )
        await asyncio.sleep(0)
        return True
    if frame.type is WorkerMessageType.CANCEL and frame.request_id is not None:
        current = active.get(frame.request_id)
        if current is not None and current[1].cancel("cancelled by service"):
            deadline = asyncio.create_task(
                _cancel_after_grace(current[0], WORKER_CANCEL_GRACE_SECONDS),
                name=f"omnitensor-plugin-cancel-{frame.request_id}",
            )
            deadlines.add(deadline)
            deadline.add_done_callback(deadlines.discard)
        return True
    _write_error(writer, protocol_version, frame.request_id, "unsupported-message")
    return True


async def _cancel_after_grace(task: asyncio.Task, grace_seconds: float) -> None:
    """Give cooperative cancellation a brief head start, then cancel the task."""
    await asyncio.sleep(grace_seconds)
    if not task.done():
        task.cancel()


async def _cancel_active_requests(
    active: dict[str, tuple[asyncio.Task, CancellationController]],
) -> None:
    """Bound worker shutdown even when a plugin ignores its cancellation token."""
    current = tuple(active.values())
    for _task, token in current:
        token.cancel("worker shutting down")
    pending = {task for task, _token in current if not task.done()}
    if not pending:
        return
    _done, pending = await asyncio.wait(pending, timeout=WORKER_CANCEL_GRACE_SECONDS)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.wait(pending, timeout=WORKER_SHUTDOWN_CANCEL_SECONDS)


class _WorkerProgress:
    def __init__(self, request_id: str, writer: BinaryIO, protocol_version: int) -> None:
        self._request_id = request_id
        self._writer = writer
        self._protocol_version = protocol_version

    async def report(self, progress: PluginProgress) -> None:
        if progress.job_id != self._request_id:
            raise IPCProtocolError("invalid-progress", "progress names another request")
        frame = progress_frame(progress)
        _write_frame(
            self._writer,
            IPCFrame(self._protocol_version, frame.type, frame.request_id, frame.payload),
        )


async def _execute_request(
    plugin: WorkloadPlugin,
    request: PluginRequest,
    token: CancellationController,
    writer: BinaryIO,
    protocol_version: int,
) -> None:
    try:
        result = await plugin.execute(
            request,
            token,
            _WorkerProgress(request.job_id, writer, protocol_version),
        )
        if result.job_id != request.job_id:
            raise IPCProtocolError("invalid-result", "result names another request")
        frames = result_frames(result)
    except asyncio.CancelledError:
        result = cancelled_result(
            request,
            "plugin request cancelled",
            completed_at_ms=max(1, time.time_ns() // 1_000_000),
        )
        frames = result_frames(result)
    except Exception:  # noqa: BLE001 - containment: this must not escape into the caller
        _write_error(writer, protocol_version, request.job_id, "plugin-execution-failed")
        return
    for frame in frames:
        _write_frame(
            writer,
            IPCFrame(protocol_version, frame.type, frame.request_id, frame.payload),
        )


def _write_error(
    writer: BinaryIO,
    protocol_version: int,
    request_id: str | None,
    code: str,
) -> None:
    _write_frame(
        writer,
        IPCFrame(
            protocol_version,
            WorkerMessageType.ERROR,
            request_id,
            {"code": code, "detail": "worker refused the message"},
        ),
    )


def _read_frame(reader: BinaryIO) -> IPCFrame:
    header = _read_exact(reader, 4)
    declared = int.from_bytes(header, "big")
    if declared > DEFAULT_MAX_FRAME_BYTES:
        raise IPCProtocolError(
            "frame-too-large",
            f"declared {declared} bytes; limit is {DEFAULT_MAX_FRAME_BYTES}",
        )
    return decode_frame(header + _read_exact(reader, declared))


def _read_exact(reader: BinaryIO, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = reader.read(size - len(chunks))
        if not chunk:
            if not chunks:
                raise EOFError
            raise IPCProtocolError(
                "truncated-frame",
                f"expected {size} bytes; received {len(chunks)}",
            )
        chunks.extend(chunk)
    return bytes(chunks)


def _write_frame(writer: BinaryIO, frame: IPCFrame) -> None:
    writer.write(encode_frame(frame))
    writer.flush()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin-id", required=True)
    parser.add_argument("--entry-point", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--distribution", required=True)
    parser.add_argument("--import-path", action="append", default=[])
    parser.add_argument("--permission", action="append", default=[])
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--state-path", default=None)
    parser.add_argument("--accelerator-lease-path", default=None)
    parser.add_argument("--model-choice", default="")
    parser.add_argument(
        "--no-seccomp",
        action="store_true",
        help="run without the exec/fork filter; only for kernels that cannot install one",
    )
    return parser


def claim_frame_channel() -> BinaryIO:
    """Take sole ownership of the framing stream before any plugin code runs.

    The IPC channel is the stdout this process inherited.  A single ``print``
    from a plugin's import or its ``start`` would be interleaved into the
    length-prefixed frames and desynchronise the service's reader for the rest
    of the worker's life.  Duplicating the descriptor keeps an exclusive handle
    on the real channel, and pointing file descriptor 1 at stderr keeps plugin
    output visible in the journal — including output written from C — without
    letting any of it reach the frames.
    """
    sys.stdout.flush()
    channel = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    return channel


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    channel = claim_frame_channel()
    for path in reversed(arguments.import_path):
        sys.path.insert(0, path)
    if not arguments.no_seccomp:
        # Before the plugin is imported, because importing it already runs its
        # code, and a filter applied afterwards would have arrived too late.
        #
        # Probed first so the refusal names its cause. Failing closed here is
        # right — a worker that believes it is confined and is not would run
        # plugin code under a guarantee that does not exist — but a stack trace
        # from deep inside ctypes told an operator nothing about their kernel.
        blocked = confinement_error()
        if blocked:
            raise SystemExit(
                f"{arguments.plugin_id}: cannot be confined and will not run unconfined: "
                f"{blocked}. Pass --no-seccomp to accept that deliberately."
            )
        install_filter()
    configure_plugin_bootstrap(_bootstrap(arguments))
    plugin = load_external_plugin(
        arguments.plugin_id,
        arguments.entry_point,
        arguments.target,
        arguments.distribution,
    )
    asyncio.run(
        serve_worker_requests(
            plugin,
            sys.stdin.buffer,
            channel,
            permissions=frozenset(arguments.permission),
        )
    )


# An artifact id as the manifests write them; the host has already refused
# anything a manifest does not pin, and this refuses anything shaped wrong.
_MODEL_CHOICE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


def _bootstrap(arguments) -> PluginBootstrap:
    artifacts = []
    for raw in arguments.artifact:
        try:
            item = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as error:
            raise SystemExit("plugin artifact bootstrap is invalid") from error
        if not isinstance(item, dict) or set(item) != {
            "id",
            "version",
            "format",
            "sha256",
            "companions",
            "path",
            "sourceUri",
            "licenseSpdx",
        }:
            raise SystemExit("plugin artifact bootstrap fields are invalid")
        companions = item["companions"]
        path = Path(item["path"]) if isinstance(item["path"], str) else Path()
        if not isinstance(companions, dict) or not path.is_absolute() or not path.is_file():
            raise SystemExit("plugin artifact bootstrap path is invalid")
        artifacts.append(
            BootstrapArtifact(
                item["id"],
                item["version"],
                item["format"],
                item["sha256"],
                path,
                tuple(sorted(companions.items())),
                item["sourceUri"],
                item["licenseSpdx"],
            )
        )
    state_path = Path(arguments.state_path) if arguments.state_path else None
    if state_path is not None and (not state_path.is_absolute() or not state_path.is_dir()):
        raise SystemExit("plugin state bootstrap path is invalid")
    lease_path = (
        Path(arguments.accelerator_lease_path) if arguments.accelerator_lease_path else None
    )
    if lease_path is not None and (not lease_path.is_absolute() or not lease_path.is_file()):
        raise SystemExit("plugin accelerator lease bootstrap path is invalid")
    model_choice = str(arguments.model_choice or "")
    if model_choice and _MODEL_CHOICE.fullmatch(model_choice) is None:
        raise SystemExit("plugin model choice bootstrap is invalid")
    return PluginBootstrap(
        arguments.plugin_id, tuple(artifacts), state_path, lease_path, model_choice
    )


if __name__ == "__main__":  # pragma: no cover - module process entry point
    main()
