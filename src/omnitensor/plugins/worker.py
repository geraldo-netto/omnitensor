"""Minimal stdio bootstrap for one separately installed workload plugin."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Callable, Iterable, Sequence
from importlib import metadata
from typing import BinaryIO

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
    parse_handshake,
)
from .protocol import PluginContext, WorkloadPlugin

WORKER_CAPABILITIES = frozenset({"cancel", "health"})


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
        raise ExternalPluginLoadError(
            f"entry-point load failed: {type(error).__name__}"
        ) from error
    if not isinstance(plugin, WorkloadPlugin) or plugin.plugin_id != plugin_id:
        raise ExternalPluginLoadError("entry point does not implement the declared plugin")
    return plugin


def serve_worker(
    plugin: WorkloadPlugin,
    reader: BinaryIO,
    writer: BinaryIO,
    *,
    minimum_protocol: int = 1,
    maximum_protocol: int = 1,
) -> HandshakeAgreement:
    """Handshake only after plugin startup, then serve bounded control frames."""
    offer = HandshakeOffer(
        plugin.plugin_id,
        minimum_protocol,
        maximum_protocol,
        WORKER_CAPABILITIES,
    )
    service = parse_handshake(_read_frame(reader))
    agreement = negotiate_handshake(service, offer)
    asyncio.run(
        plugin.start(
            PluginContext(plugin.plugin_id, agreement.protocol_version, {}, frozenset())
        )
    )
    _write_frame(writer, handshake_frame(offer))
    try:
        while True:
            try:
                frame = _read_frame(reader)
            except EOFError:
                break
            if frame.type is WorkerMessageType.CANCEL and frame.request_id is None:
                break
            _write_frame(
                writer,
                IPCFrame(
                    agreement.protocol_version,
                    WorkerMessageType.ERROR,
                    frame.request_id,
                    {"code": "unsupported-message", "detail": "message is not implemented"},
                ),
            )
    finally:
        asyncio.run(plugin.stop())
    return agreement


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
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    for path in reversed(arguments.import_path):
        sys.path.insert(0, path)
    plugin = load_external_plugin(
        arguments.plugin_id,
        arguments.entry_point,
        arguments.target,
        arguments.distribution,
    )
    serve_worker(plugin, sys.stdin.buffer, sys.stdout.buffer)


if __name__ == "__main__":  # pragma: no cover - module process entry point
    main()
