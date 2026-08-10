#!/usr/bin/env python3
"""Privileged eBPF helper: load the probes, export aggregates, hold nothing else.

Runs as a *system* unit with CAP_BPF and CAP_PERFMON because the OmniTensor
service cannot: it is an unprivileged user unit, and on a host with
``kernel.unprivileged_bpf_disabled=2`` there is no unprivileged load path.

The privilege split is the whole point, so this process is deliberately small.
It loads a pre-built CO-RE object, reads two histogram maps, and serves them as
JSON on a unix socket.  It never reads a pid, a comm, or a path out of the
kernel, because the maps do not contain any — the aggregation happens in BPF,
which is what makes the exported data safe for an unprivileged reader.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

AGGREGATE_VERSION = 1
DEFAULT_OBJECT = "/usr/lib/omnitensor/bpf/runq_latency.bpf.o"
DEFAULT_PIN_DIR = "/sys/fs/bpf/omnitensor"
DEFAULT_SOCKET = "/run/omnitensor/bpf-aggregate.sock"
HISTOGRAMS = (("runq_latency_us", "us"), ("block_latency_us", "us"))


def load_probes(object_path: Path, pin_dir: Path) -> None:
    """Load and pin the probes; already-pinned is success, not a failure."""
    if pin_dir.exists():
        return
    subprocess.run(
        ["bpftool", "prog", "loadall", str(object_path), str(pin_dir), "pinmaps", str(pin_dir)],
        check=True,
        capture_output=True,
    )
    for entry in sorted(pin_dir.iterdir()):
        if entry.name.startswith("sched_") or entry.name.startswith("block_rq"):
            subprocess.run(
                ["bpftool", "prog", "attach", "pinned", str(entry), "raw_tracepoint"],
                check=False,
                capture_output=True,
            )


def read_histogram(pin_dir: Path, name: str) -> list[int]:
    """One map's bucket counts, in slot order."""
    result = subprocess.run(
        ["bpftool", "map", "dump", "pinned", str(pin_dir / name), "-j"],
        check=True,
        capture_output=True,
        text=True,
    )
    buckets: dict[int, int] = {}
    for entry in json.loads(result.stdout):
        key = _packed(entry.get("key", []))
        buckets[key] = _packed(entry.get("value", []))
    return [buckets.get(slot, 0) for slot in range(max(buckets, default=-1) + 1)]


def _packed(words: list) -> int:
    value = 0
    for index, byte in enumerate(words):
        value |= int(byte, 16) << (8 * index) if isinstance(byte, str) else byte << (8 * index)
    return value


def aggregate(pin_dir: Path) -> dict:
    return {
        "version": AGGREGATE_VERSION,
        "collectedAtMs": int(time.time() * 1000),
        "histograms": [
            {"name": name, "unit": unit, "buckets": read_histogram(pin_dir, name)}
            for name, unit in HISTOGRAMS
        ],
        "counters": [],
    }


def serve(socket_path: Path, pin_dir: Path) -> None:
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        socket_path.unlink()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(socket_path))
        # World-readable: the consumer is an unprivileged user service, and the
        # exported data is aggregate-only by construction.
        os.chmod(socket_path, 0o666)
        server.listen(8)
        while True:
            connection, _ = server.accept()
            with connection:
                try:
                    payload = json.dumps(aggregate(pin_dir), separators=(",", ":"))
                except (subprocess.CalledProcessError, ValueError) as error:
                    payload = json.dumps({"version": AGGREGATE_VERSION, "error": str(error)[:200]})
                connection.sendall(payload.encode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", default=DEFAULT_OBJECT)
    parser.add_argument("--pin-dir", default=DEFAULT_PIN_DIR)
    parser.add_argument("--socket", default=DEFAULT_SOCKET)
    parser.add_argument("--print-once", action="store_true")
    arguments = parser.parse_args(argv)

    pin_dir = Path(arguments.pin_dir)
    try:
        load_probes(Path(arguments.object), pin_dir)
    except (subprocess.CalledProcessError, OSError) as error:
        # Fail loudly: a helper that runs without its probes would serve empty
        # histograms, which a reader cannot tell from an idle kernel.
        print(f"could not load BPF probes: {error}", file=sys.stderr)
        return 1
    if arguments.print_once:
        print(json.dumps(aggregate(pin_dir), indent=1))
        return 0
    serve(Path(arguments.socket), pin_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
