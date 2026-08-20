"""Authenticated worker sessions and request/response correlation."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field

from ..stable_error import StableError
from .budgets import WorkerBudgetEnforcer
from .ipc import (
    HandshakeAgreement,
    IPCFrame,
    IPCProtocolError,
    WorkerMessageType,
    assemble_result_chunks,
    await_worker_ready,
    parse_progress,
    parse_result,
    parse_result_chunk,
    perform_service_handshake,
    read_frame,
    write_frame,
)
from .protocol import PluginResult, ProgressReporter
from .supervisor_process import (
    WorkerLauncher,
    WorkerProcess,
    WorkerSpec,
    drain_worker_diagnostics,
)


class PluginWorkerError(StableError, RuntimeError):
    """Stable executable-worker refusal safe to surface as a job failure."""


@dataclass(slots=True)
class WorkerSlot:
    spec: WorkerSpec
    process: WorkerProcess
    agreement: HandshakeAgreement
    monitor: asyncio.Task | None = None
    request_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    budget: WorkerBudgetEnforcer | None = None


class WorkerStartError(Exception):
    def __init__(self, detail: str, pid: int | None, *, charges_restart: bool = True) -> None:
        super().__init__(detail)
        self.detail = detail
        self.pid = pid
        self.charges_restart = charges_restart


def with_protocol(frame: IPCFrame, protocol_version: int) -> IPCFrame:
    return IPCFrame(protocol_version, frame.type, frame.request_id, frame.payload)


def handshake_failure(
    error: Exception,
    *,
    protocol_error_type=IPCProtocolError,
) -> str:
    if isinstance(error, TimeoutError):
        return "worker handshake timed out"
    if isinstance(error, protocol_error_type):
        return f"worker handshake rejected: {error.code}"
    return f"worker handshake failed: {type(error).__name__}"


def startup_failure(
    error: Exception,
    *,
    protocol_error_type=IPCProtocolError,
) -> str:
    if isinstance(error, TimeoutError):
        return "worker startup timed out"
    if isinstance(error, protocol_error_type):
        return f"worker startup rejected: {error.code}"
    return f"worker startup failed: {type(error).__name__}"


def with_worker_diagnostics(detail: str, output: str) -> str:
    """Carry what the worker said on stderr into the detail a person reads."""
    if not output:
        return detail
    return f"{detail}; worker stderr: {output}"


async def launch_authenticated(
    launcher: WorkerLauncher,
    spec: WorkerSpec,
    *,
    handshake_timeout: float,
    startup_timeout: float,
    stop_timeout: float,
    handshake: Callable[..., Awaitable[HandshakeAgreement]] = perform_service_handshake,
    await_ready: Callable[..., Awaitable[object]] = await_worker_ready,
    force_stop: Callable[[WorkerProcess, float], Awaitable[bool]],
    handshake_failure_of: Callable[[Exception], str] = handshake_failure,
    startup_failure_of: Callable[[Exception], str] = startup_failure,
    drain_diagnostics: Callable[[WorkerProcess], Awaitable[str]] = drain_worker_diagnostics,
) -> tuple[WorkerProcess, HandshakeAgreement]:
    try:
        process = await launcher.launch(spec)
    except Exception as error:
        raise WorkerStartError(f"worker launch failed: {type(error).__name__}", None) from error
    try:
        agreement = await asyncio.wait_for(
            handshake(process.reader, process.writer, spec.offer()),
            timeout=handshake_timeout,
        )
    except asyncio.CancelledError:
        # Shielded: every await inside force_stop would raise CancelledError
        # again at its first suspension point, so an unshielded cleanup never
        # reaches the SIGTERM and leaves the child running as a session leader.
        await asyncio.shield(force_stop(process, stop_timeout))
        raise
    except Exception as error:
        await force_stop(process, stop_timeout)
        raise WorkerStartError(
            with_worker_diagnostics(handshake_failure_of(error), await drain_diagnostics(process)),
            process.pid,
        ) from error
    try:
        await asyncio.wait_for(
            await_ready(process.reader, spec.plugin_id),
            timeout=startup_timeout,
        )
    except asyncio.CancelledError:
        await asyncio.shield(force_stop(process, stop_timeout))
        raise
    except Exception as error:
        await force_stop(process, stop_timeout)
        raise WorkerStartError(
            with_worker_diagnostics(startup_failure_of(error), await drain_diagnostics(process)),
            process.pid,
            charges_restart=not isinstance(error, TimeoutError),
        ) from error
    return process, agreement


async def read_result(
    slot: WorkerSlot,
    request_id: str,
    progress: ProgressReporter | None,
    *,
    read: Callable[..., Awaitable[IPCFrame]] = read_frame,
    progress_parser: Callable[[IPCFrame], object] = parse_progress,
    result_parser: Callable[[IPCFrame], PluginResult] = parse_result,
    chunk_parser: Callable[[IPCFrame, int], tuple[str, bool]] = parse_result_chunk,
    chunk_assembler: Callable[[str, Sequence[str]], PluginResult] = assemble_result_chunks,
) -> PluginResult:
    parts: list[str] = []
    while True:
        frame = _correlated(await read(slot.process.reader), slot, request_id)
        if frame.type is WorkerMessageType.PROGRESS:
            await _report_progress(progress, progress_parser(frame))
            continue
        if frame.type is WorkerMessageType.RESULT:
            return result_parser(frame)
        if frame.type is WorkerMessageType.RESULT_CHUNK:
            # An answer too large for one frame is still the answer: collect
            # the pieces in order rather than refusing the whole result.
            body, final = chunk_parser(frame, len(parts))
            parts.append(body)
            if final:
                return chunk_assembler(request_id, parts)
            continue
        if frame.type is WorkerMessageType.ERROR:
            _raise_worker_error(frame)
        raise PluginWorkerError(
            "worker-protocol-failed", f"unexpected worker message: {frame.type}"
        )


def _correlated(frame: IPCFrame, slot: WorkerSlot, request_id: str) -> IPCFrame:
    """The one place a worker frame is admitted: right version, right request.

    The version was checked only during the handshake, so every later frame
    was handed to a parser that reads fields without knowing which version
    settled what they mean.
    """
    negotiated = slot.agreement.protocol_version
    if frame.version != negotiated:
        raise PluginWorkerError(
            "worker-protocol-failed",
            f"worker frame version {frame.version} is not the negotiated {negotiated}",
        )
    if frame.request_id != request_id:
        raise PluginWorkerError("worker-protocol-failed", "worker response names another request")
    return frame


async def _report_progress(progress: ProgressReporter | None, observed: object) -> None:
    if progress is not None:
        await progress.report(observed)


def _raise_worker_error(frame: IPCFrame) -> None:
    """Re-raise a worker's own refusal, or refuse the refusal if it is malformed."""
    code = frame.payload.get("code")
    detail = frame.payload.get("detail")
    if not isinstance(code, str) or not code or not isinstance(detail, str):
        raise PluginWorkerError("worker-protocol-failed", "worker error response is invalid")
    raise PluginWorkerError(code, detail)


async def cancel_request(
    slot: WorkerSlot,
    request_id: str,
    *,
    cancel_timeout: float,
    stop_timeout: float,
    read_result_of: Callable[..., Awaitable[PluginResult]],
    write: Callable[..., Awaitable[object]] = write_frame,
    force_stop: Callable[[WorkerProcess, float], Awaitable[bool]],
) -> None:
    try:
        await write(
            slot.process.writer,
            IPCFrame(
                slot.agreement.protocol_version,
                WorkerMessageType.CANCEL,
                request_id,
                {"reason": "job cancelled"},
            ),
        )
        await asyncio.wait_for(
            read_result_of(slot, request_id, None),
            timeout=cancel_timeout,
        )
    except (Exception, asyncio.CancelledError):  # noqa: BLE001 - containment: this must not escape into the caller
        await asyncio.shield(force_stop(slot.process, stop_timeout))


async def stop_process(
    slot: WorkerSlot,
    *,
    timeout: float,
    write: Callable[..., Awaitable[object]] = write_frame,
    close_writer: Callable[..., Awaitable[object]],
    terminate: Callable[[WorkerProcess, float], Awaitable[bool]],
) -> bool:
    if slot.process.returncode is not None:
        return True
    protocol_error = IPCProtocolError
    # The version the handshake agreed, like every other service-to-worker
    # write: a worker speaking a negotiated version was told to shut down in
    # a dialect it never agreed to.
    frame_version = slot.agreement.protocol_version
    with suppress(ConnectionError, protocol_error, RuntimeError):
        await write(
            slot.process.writer,
            IPCFrame(
                frame_version,
                WorkerMessageType.CANCEL,
                None,
                {"reason": "shutdown"},
            ),
        )
    await close_writer(slot.process.writer)
    return await terminate(slot.process, timeout)


__all__ = [
    "PluginWorkerError",
    "WorkerSlot",
    "WorkerStartError",
    "cancel_request",
    "handshake_failure",
    "launch_authenticated",
    "read_result",
    "startup_failure",
    "stop_process",
    "with_protocol",
]
