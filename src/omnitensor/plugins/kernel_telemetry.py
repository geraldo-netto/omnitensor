"""Kernel telemetry from eBPF, read from a privileged helper this service is not.

The service runs as an unprivileged systemd *user* unit with
``NoNewPrivileges=true``, and its workers additionally drop every capability.
Loading a BPF program needs ``CAP_BPF``/``CAP_PERFMON``, and on a host with
``kernel.unprivileged_bpf_disabled=2`` there is no unprivileged path at all —
verified here: ``bpftool prog load`` fails ``EPERM`` as this user.  So eBPF
cannot be imported into this process, however convenient that would be.  It
lives behind a separate privileged helper, and this module is only the reader.

That split is also the privacy boundary, and it is the reason for the strictest
rule here.  eBPF can see command lines, file paths, and packet bytes — exactly
what every collector in this service refuses to carry.  The helper therefore
aggregates *in kernel space*, in BPF maps, and exports only histograms and
counters.  An aggregate that arrives carrying a pid, a comm, a cgroup path, or
a filename is refused outright rather than stripped, because a field that
should be impossible is a contract break, not something to sanitise.

Absence is reported, never faked.  A host with no helper installed is a normal
host, and returning zeroes would be indistinguishable from a quiet kernel —
which is precisely the reading that would make a scheduler profile confident
and wrong.
"""

from __future__ import annotations

import json
import math
import socket
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

DEFAULT_SOCKET_PATH = "/run/omnitensor/bpf-aggregate.sock"
DEFAULT_MAX_AGGREGATE_BYTES = 256 * 1024
DEFAULT_TIMEOUT_SECONDS = 2.0
AGGREGATE_VERSION = 1
MAX_SERIES = 64
MAX_BUCKETS = 64
MAX_NAME_LENGTH = 64
MAX_DETAIL_LENGTH = 240
MAX_COUNT = 2**53 - 1

# Anything that could identify a process, a user, or a file. eBPF can read all
# of it, so the reader refuses it rather than trusting the helper not to send it.
FORBIDDEN_FIELDS = frozenset(
    {
        "pid",
        "tid",
        "ppid",
        "uid",
        "gid",
        "comm",
        "command",
        "cmdline",
        "argv",
        "exe",
        "path",
        "filename",
        "cgroup",
        "cgroupPath",
        "container",
        "saddr",
        "daddr",
        "payload",
    }
)


class KernelTelemetryState(StrEnum):
    """Why an aggregate holds what it holds — absence is never a zero."""

    READY = "ready"
    HELPER_ABSENT = "helper-absent"
    HELPER_UNREACHABLE = "helper-unreachable"
    HELPER_INVALID = "helper-invalid"


class KernelTelemetryError(ValueError):
    """Stable kernel-telemetry contract failure."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class LatencyHistogram:
    """One log2-bucketed latency distribution, as the BPF map holds it."""

    name: str
    unit: str
    buckets: tuple[int, ...]

    @property
    def total(self) -> int:
        return sum(self.buckets)

    def document(self) -> dict:
        return {"name": self.name, "unit": self.unit, "buckets": list(self.buckets)}


@dataclass(frozen=True, slots=True)
class KernelCounter:
    """One monotonic kernel counter."""

    name: str
    value: int

    def document(self) -> dict:
        return {"name": self.name, "value": self.value}


@dataclass(frozen=True, slots=True)
class KernelAggregate:
    """Everything the helper exports, and whether it can be trusted."""

    state: KernelTelemetryState
    detail: str = ""
    collected_at_ms: int = 0
    histograms: tuple[LatencyHistogram, ...] = ()
    counters: tuple[KernelCounter, ...] = field(default_factory=tuple)

    @property
    def usable(self) -> bool:
        return self.state is KernelTelemetryState.READY

    def document(self) -> dict:
        return {
            "version": AGGREGATE_VERSION,
            "state": str(self.state),
            "detail": self.detail[:MAX_DETAIL_LENGTH],
            "collectedAtMs": self.collected_at_ms,
            "histograms": [item.document() for item in self.histograms],
            "counters": [item.document() for item in self.counters],
        }

    def scheduler_features(self) -> dict[str, float]:
        """Bounded, aggregate-only latency features for resource scheduling."""
        return scheduler_features(self)


def scheduler_features(aggregate: KernelAggregate) -> dict[str, float]:
    """Summarize the two declared log2 histograms into stable model inputs."""
    if not aggregate.usable:
        return {}
    histograms = {item.name: item for item in aggregate.histograms}
    features: dict[str, float] = {}
    for name, prefix in (
        ("runq_latency_us", "kernelRunQueue"),
        ("block_latency_us", "kernelBlockIo"),
    ):
        histogram = histograms.get(name)
        if histogram is None:
            continue
        features[f"{prefix}Samples"] = float(min(histogram.total, MAX_COUNT))
        features[f"{prefix}P50UpperUs"] = float(_percentile_upper(histogram, 50))
        features[f"{prefix}P95UpperUs"] = float(_percentile_upper(histogram, 95))
    return features


def _bounded_name(value: object) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= MAX_NAME_LENGTH:
        raise KernelTelemetryError("aggregate-invalid", "series name is invalid")
    return value


def forbidden_field(document: Mapping[str, object]) -> str:
    """The first identity-bearing field a document carries, or ``""``.

    Checked recursively: a pid nested one level down is still a pid.
    """
    for key, value in document.items():
        if isinstance(key, str) and key in FORBIDDEN_FIELDS:
            return key
        nested = _nested_forbidden(value)
        if nested:
            return nested
    return ""


def _nested_forbidden(value: object) -> str:
    if isinstance(value, Mapping):
        return forbidden_field(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            if isinstance(item, Mapping):
                found = forbidden_field(item)
                if found:
                    return found
    return ""


def parse_aggregate(document: object) -> KernelAggregate:
    """Validate one helper document into an aggregate, or refuse it."""
    if not isinstance(document, Mapping):
        raise KernelTelemetryError("aggregate-invalid", "aggregate must be an object")
    if document.get("version") != AGGREGATE_VERSION:
        raise KernelTelemetryError(
            "aggregate-invalid", f"unsupported aggregate version: {document.get('version')}"
        )
    carried = forbidden_field(document)
    if carried:
        # The helper is supposed to aggregate in kernel space.  One that sends
        # a pid has stopped doing that, and stripping the field would hide it.
        raise KernelTelemetryError(
            "aggregate-identifying", f"kernel telemetry must not carry {carried}"
        )
    collected = document.get("collectedAtMs", 0)
    if isinstance(collected, bool) or not isinstance(collected, int) or collected < 0:
        raise KernelTelemetryError("aggregate-invalid", "collectedAtMs is invalid")
    return KernelAggregate(
        KernelTelemetryState.READY,
        "",
        collected,
        _histograms(document.get("histograms", [])),
        _counters(document.get("counters", [])),
    )


def _histograms(raw: object) -> tuple[LatencyHistogram, ...]:
    if not isinstance(raw, list) or len(raw) > MAX_SERIES:
        raise KernelTelemetryError("aggregate-invalid", "histograms are invalid")
    parsed = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise KernelTelemetryError("aggregate-invalid", "histogram must be an object")
        buckets = item.get("buckets")
        if not isinstance(buckets, list) or len(buckets) > MAX_BUCKETS:
            raise KernelTelemetryError("aggregate-invalid", "histogram buckets are invalid")
        for count in buckets:
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or not 0 <= count <= MAX_COUNT
            ):
                raise KernelTelemetryError("aggregate-invalid", "bucket counts must be counts")
        parsed.append(
            LatencyHistogram(
                _bounded_name(item.get("name")),
                _bounded_name(item.get("unit", "ns")),
                tuple(buckets),
            )
        )
    return tuple(parsed)


def _counters(raw: object) -> tuple[KernelCounter, ...]:
    if not isinstance(raw, list) or len(raw) > MAX_SERIES:
        raise KernelTelemetryError("aggregate-invalid", "counters are invalid")
    parsed = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise KernelTelemetryError("aggregate-invalid", "counter must be an object")
        value = item.get("value")
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= MAX_COUNT
        ):
            raise KernelTelemetryError("aggregate-invalid", "counter value must be a count")
        parsed.append(KernelCounter(_bounded_name(item.get("name")), value))
    return tuple(parsed)


def _percentile_upper(histogram: LatencyHistogram, percentile: int) -> int:
    """Exclusive upper bound of one log2 bucket percentile, in histogram units."""
    total = histogram.total
    if total < 1:
        return 0
    threshold = (total * percentile + 99) // 100
    cumulative = 0
    for index, count in enumerate(histogram.buckets):
        cumulative += count
        if cumulative >= threshold:
            return min(1 << (index + 1), MAX_COUNT)
    return 0


class UnixSocketAggregateSource:
    """Read one aggregate from the privileged helper's socket.

    Reading rather than connecting-and-holding: the helper is a separate unit
    that may restart independently, and a long-lived connection would make this
    service's health depend on the helper's uptime.
    """

    def __init__(
        self,
        socket_path: Path | str = DEFAULT_SOCKET_PATH,
        *,
        max_aggregate_bytes: int = DEFAULT_MAX_AGGREGATE_BYTES,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if (
            isinstance(max_aggregate_bytes, bool)
            or not isinstance(max_aggregate_bytes, int)
            or max_aggregate_bytes < 1
        ):
            raise KernelTelemetryError(
                "bounds-invalid", "max_aggregate_bytes must be a positive integer"
            )
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise KernelTelemetryError("bounds-invalid", "timeout_seconds must be positive")
        self._socket_path = Path(socket_path)
        self._max_aggregate_bytes = max_aggregate_bytes
        self._timeout_seconds = timeout_seconds

    @property
    def socket_path(self) -> Path:
        return self._socket_path

    def read(self) -> KernelAggregate:
        """One aggregate, or a stated reason there is none."""
        if not self._socket_path.exists():
            # A host without the helper is an ordinary host, not a fault.
            return KernelAggregate(
                KernelTelemetryState.HELPER_ABSENT,
                f"no kernel telemetry helper at {self._socket_path}",
            )
        try:
            payload = self._read_bytes()
        except OSError as error:
            return KernelAggregate(
                KernelTelemetryState.HELPER_UNREACHABLE,
                f"kernel telemetry helper is unreachable: {error}",
            )
        try:
            return parse_aggregate(json.loads(payload))
        except (ValueError, KernelTelemetryError) as error:
            return KernelAggregate(
                KernelTelemetryState.HELPER_INVALID,
                f"kernel telemetry helper sent an unusable aggregate: {error}",
            )

    def _read_bytes(self) -> bytes:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
            stream.settimeout(self._timeout_seconds)
            stream.connect(str(self._socket_path))
            chunks: list[bytes] = []
            remaining = self._max_aggregate_bytes + 1
            while remaining > 0:
                chunk = stream.recv(min(65536, remaining))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
        payload = b"".join(chunks)
        if len(payload) > self._max_aggregate_bytes:
            raise OSError(f"aggregate exceeds {self._max_aggregate_bytes} bytes")
        return payload


class AbsentAggregateSource:
    """The source used when no helper is configured, stating so plainly."""

    def read(self) -> KernelAggregate:
        return KernelAggregate(
            KernelTelemetryState.HELPER_ABSENT,
            "kernel telemetry is not configured for this service",
        )
