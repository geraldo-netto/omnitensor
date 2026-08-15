#!/usr/bin/env python3
"""Codec microbenchmark behind the control-plane transport decision.

Compares framed JSON against framed msgpack on this system's real payloads.
The measured claims:

  (a) at control-plane sizes (~0.5-6 KB) and rates (~1 message/second) the
      codec difference is microseconds and invisible;
  (b) msgpack pays enormously for inline tensors, through its ``bin`` type,
      which steps outside JSON's data model.

Recorded run (2026-08-15, this host, msgpack 1.0.3 C extension, live snapshot
and live describe-plugins payloads):

  apply-command ack  653 B    json 4.2/3.3 us     msgpack 1.6/2.0 us
  snapshot           5.5 KB   json 21.4/20.9 us   msgpack 9.8/16.1 us
  describe-plugins   6.2 KB   json 25.1/22.2 us   msgpack 11.8/17.5 us
  256x256x3 f32      json 52.7 ms enc / 3.7 MB
                     msgpack 2.2 ms / 1.8 MB
                     msgpack+bin 44.9 us / 786 KB   (~1,200x faster, 4.7x smaller)

Decision (operator's, 2026-08-15, superseding this script's earlier framed-JSON
default): msgpack everywhere, one codec, no negotiation. msgpack won every
codec race outright; the readable-wire and zero-dependency arguments for JSON
were judged not worth a second wire format, so ``msgpack`` is a project
dependency and the control socket (``socket_transport.py``) speaks framed
msgpack only, with the ``bin`` tensor headroom already in place.
"""

from __future__ import annotations

import json
import os
import struct
import timeit

import msgpack

REPS_SMALL = 2000
REPS_TENSOR = 20
DEFAULT_STATE_PATH = "~/.local/state/xpu-workload-manager/state.json"


def live_snapshot() -> dict | None:
    path = os.path.expanduser(os.environ.get("OMNITENSOR_STATE_PATH", DEFAULT_STATE_PATH))
    try:
        with open(path, "rb") as handle:
            return json.load(handle)
    except OSError:
        return None


def live_inventory() -> dict | None:
    try:
        import asyncio  # noqa: PLC0415

        from omnitensor.socket_transport import call_control  # noqa: PLC0415

        return asyncio.run(call_control("describe-plugins", {}))
    except Exception:  # noqa: BLE001 - live probe; the benchmark continues without it
        return None


def command_acknowledgement() -> dict:
    """The apply-command acknowledgement shape observed live on 2026-08-15."""
    profiles = [
        ("build-advisor", 2),
        ("desktop-context", 1),
        ("document-intelligence", 2),
        ("hardware-health", 3),
        ("low-light-enhancement", 2),
        ("network-peripherals", 2),
        ("resource-scheduler", 3),
        ("storage-intelligence", 3),
        ("visual-library", 2),
    ]
    return {
        "version": 2,
        "commandId": "invalid",
        "status": "rejected",
        "revision": 97,
        "appliedAt": 1786760956830,
        "message": "Command does not match the version 2 contract",
        "portfolio": {
            "paused": False,
            "profiles": {name: {"enabled": True, "weight": weight} for name, weight in profiles},
            "deviceChoices": {},
        },
    }


def tensor_values(count: int) -> list[float]:
    # Normalized pixel values with long decimal expansions: JSON's worst case,
    # and exactly what the tensor contracts carry (scale 1/255 and friends).
    return [(index % 256) / 255.0 for index in range(count)]


def frame(payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + payload


def measure(encode, reps: int) -> tuple[bytes, float, float]:
    wire = encode()
    payload = wire[4:]
    decode = (
        (lambda: json.loads(payload.decode("utf-8")))
        if payload[:1] in (b"{", b"[")
        else (lambda: msgpack.unpackb(payload))
    )
    encode_us = timeit.timeit(encode, number=reps) / reps * 1e6
    decode_us = timeit.timeit(decode, number=reps) / reps * 1e6
    return wire, encode_us, decode_us


def bench(label: str, document: dict, reps: int, tensor_key: str | None = None) -> None:
    rows = [
        ("json", lambda: frame(json.dumps(document, separators=(",", ":")).encode("utf-8"))),
        ("msgpack", lambda: frame(msgpack.packb(document))),
    ]
    if tensor_key is not None:
        binary = dict(document)
        values = document[tensor_key]
        binary[tensor_key] = struct.pack(f"<{len(values)}f", *values)
        rows.append(("msgpack+bin", lambda: frame(msgpack.packb(binary))))

    print(f"\n{label}")
    baseline: tuple[int, float, float] | None = None
    for name, encode in rows:
        wire, encode_us, decode_us = measure(encode, reps)
        print(
            f"  {name:12} wire={len(wire):>9,} B   "
            f"encode={encode_us:>10.1f} us   decode={decode_us:>10.1f} us"
        )
        if baseline is None:
            baseline = (len(wire), encode_us, decode_us)
        else:
            print(
                f"  {name:12} vs json: size x{len(wire) / baseline[0]:.2f}, "
                f"encode x{encode_us / baseline[1]:.2f}, decode x{decode_us / baseline[2]:.2f}"
            )


def main() -> None:
    bench(
        "1. apply-command acknowledgement (real shape, control plane)",
        command_acknowledgement(),
        REPS_SMALL,
    )

    snapshot = live_snapshot()
    if snapshot is None:
        print("\n2. snapshot skipped: no readable state file")
    else:
        bench("2. Live runtime snapshot (real file bytes)", snapshot, REPS_SMALL)

    inventory = live_inventory()
    if inventory is None:
        print("\n3. describe-plugins skipped: service not reachable on the control socket")
    else:
        bench(
            "3. Live describe-plugins reply (real, via the control socket)",
            inventory,
            REPS_SMALL,
        )

    bench(
        "4. Small inline tensor job (1x3x8x8 = 192 floats)",
        {
            "version": 1,
            "requestId": "xpuwlm-1-1",
            "workloadId": "visual-library",
            "payload": {"inputs": tensor_values(1 * 3 * 8 * 8)},
        },
        REPS_SMALL,
    )

    bench(
        "5. Full image tensor (256x256x3 = 196,608 float32)",
        {
            "version": 1,
            "requestId": "xpuwlm-1-2",
            "workloadId": "low-light-enhancement",
            "inputs": tensor_values(256 * 256 * 3),
        },
        REPS_TENSOR,
        tensor_key="inputs",
    )

    print("\nContext: the control plane runs at ~1 message/second, and the snapshot")
    print("travels as a file, not through the socket. Even 100 us per message is")
    print("0.01% of one core; the tensor rows are the only regime where the codec")
    print("choice is observable.")


if __name__ == "__main__":
    main()
