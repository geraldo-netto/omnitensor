"""Minimal stdio bootstrap for one separately installed workload plugin."""

from __future__ import annotations

import argparse
import asyncio
import os
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
    ready_frame,
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
    # WorkloadPlugin does not declare plugin_id, so isinstance() admits an
    # object without it; reading the attribute directly would surface a raw
    # AttributeError instead of the load error callers handle.
    if not isinstance(plugin, WorkloadPlugin) or getattr(plugin, "plugin_id", None) != plugin_id:
        raise ExternalPluginLoadError("entry point does not implement the declared plugin")
    return plugin


def serve_worker(
    plugin: WorkloadPlugin,
    reader: BinaryIO,
    writer: BinaryIO,
    *,
    minimum_protocol: int = 1,
    maximum_protocol: int = 1,
    permissions: frozenset[str] = frozenset(),
) -> HandshakeAgreement:
    """Acknowledge the handshake, start the plugin, then serve control frames.

    The handshake is answered before ``start`` runs so the service can bound
    protocol negotiation and plugin startup separately; a plugin that takes
    seconds to load a model no longer looks like a failed handshake.
    """
    offer = HandshakeOffer(
        plugin.plugin_id,
        minimum_protocol,
        maximum_protocol,
        WORKER_CAPABILITIES,
    )
    service = parse_handshake(_read_frame(reader))
    agreement = negotiate_handshake(service, offer)
    _write_frame(writer, handshake_frame(offer))
    asyncio.run(
        plugin.start(
            PluginContext(plugin.plugin_id, agreement.protocol_version, {}, permissions)
        )
    )
    _write_frame(writer, ready_frame(plugin.plugin_id))
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
    parser.add_argument("--permission", action="append", default=[])
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
    plugin = load_external_plugin(
        arguments.plugin_id,
        arguments.entry_point,
        arguments.target,
        arguments.distribution,
    )
    serve_worker(
        plugin,
        sys.stdin.buffer,
        channel,
        permissions=frozenset(arguments.permission),
    )


if __name__ == "__main__":  # pragma: no cover - module process entry point
    main()
