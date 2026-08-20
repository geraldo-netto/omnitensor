"""Neutral value contracts shared by live collectors and offline training."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from .stable_error import StableError

RECORD_VERSION = 1
DEFAULT_MAX_BUILD_RECORDS = 5_000
MAX_BUILD_DURATION_MS = 30 * 24 * 60 * 60 * 1000
MAX_NETWORK_LINKS = 256
MAX_NETWORK_COUNTER = 2**63 - 1
MAX_TEMPERATURE_MILLIDEGREES = 200_000
MAX_ERROR_COUNT = 2**53
MAX_PERCENT = 100
STABLE_ID = re.compile(r"[a-z0-9](?:[a-z0-9._:@/-]{0,78}[a-z0-9])?")
NETWORK_STABLE_ID = re.compile(r"[a-z0-9](?:[a-z0-9._-]{0,78}[a-z0-9])?")
PERIPHERAL_STABLE_ID = NETWORK_STABLE_ID


class RecorderError(StableError, ValueError):
    """Stable recording contract failure."""


@dataclass(frozen=True, slots=True)
class FeatureRow:
    """One observation: named numbers and when they were taken."""

    profile_id: str
    observed_at_ms: int
    features: Mapping[str, float]

    def document(self) -> dict:
        return {
            "v": RECORD_VERSION,
            "p": self.profile_id,
            "t": self.observed_at_ms,
            "f": dict(self.features),
        }


class ForecastError(StableError, ValueError):
    """Stable fitting or forecasting failure."""


@dataclass(frozen=True, slots=True)
class ForecastQuality:
    """How a fit scored against data it did not see."""

    samples: int
    mean_absolute_error: float
    baseline_error: float

    @property
    def skill(self) -> float:
        """Fraction of the naive baseline's error removed; <= 0 means none."""
        if self.baseline_error <= 0:
            return 0.0
        return 1.0 - (self.mean_absolute_error / self.baseline_error)

    @property
    def useful(self) -> bool:
        """Whether this beats predicting 'the same as last time' at all."""
        return self.skill > 0.0

    def document(self) -> dict:
        return {
            "samples": self.samples,
            "meanAbsoluteError": round(self.mean_absolute_error, 6),
            "baselineError": round(self.baseline_error, 6),
            "skill": round(self.skill, 6),
            "useful": self.useful,
        }


@dataclass(frozen=True, slots=True)
class LinearForecaster:
    """A fitted linear model, bound to the feature order it was fitted on."""

    feature_names: tuple[str, ...]
    window: int
    weights: tuple[float, ...]
    intercept: float
    quality: ForecastQuality

    @property
    def input_width(self) -> int:
        return len(self.feature_names) * self.window

    def predict(self, inputs: Sequence[float]) -> float:
        """One forecast, refusing any vector that is not the fitted shape."""
        if isinstance(inputs, (str, bytes)) or not isinstance(inputs, Sequence):
            raise ForecastError("input-invalid", "inputs must be a sequence of numbers")
        if len(inputs) != self.input_width:
            raise ForecastError(
                "input-invalid",
                f"expected {self.input_width} values, got {len(inputs)}",
            )
        total = self.intercept
        for weight, value in zip(self.weights, inputs, strict=True):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ForecastError("input-invalid", "inputs must be numbers")
            if not math.isfinite(value):
                raise ForecastError("input-invalid", "inputs must be finite")
            total += weight * float(value)
        return total

    def document(self) -> dict:
        return {
            "featureNames": list(self.feature_names),
            "window": self.window,
            "weights": [round(weight, 8) for weight in self.weights],
            "intercept": round(self.intercept, 8),
            "quality": self.quality.document(),
        }


class SourceStatus(StrEnum):
    READY = "ready"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class BuildOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class BuildRecord:
    """One exported build result; never a log body, never source content."""

    build_id: str
    outcome: BuildOutcome
    duration_ms: int
    started_at_ms: int
    changed_paths: tuple[str, ...]
    failed_checks: tuple[str, ...]


class SensorKind(StrEnum):
    """What a reading measures, which decides how it may be interpreted."""

    THERMAL = "thermal"
    FAN = "fan"
    VOLTAGE = "voltage"
    MEMORY_ERRORS = "memory-errors"
    POWER = "power"
    SERVICE = "service"


class SensorHealth(StrEnum):
    """Whether the reading itself can be trusted, separate from its value."""

    OK = "ok"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class HardwareSample:
    """One bounded reading with its unit stated rather than assumed."""

    stable_id: str
    kind: SensorKind
    health: SensorHealth
    value: int | None
    unit: str
    label: str
    observed_at_ms: int


class NetworkLinkKind(StrEnum):
    ETHERNET = "ethernet"
    WIFI = "wifi"
    CELLULAR = "cellular"
    VPN = "vpn"
    OTHER = "other"


class NetworkLinkState(StrEnum):
    UP = "up"
    DOWN = "down"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


class NetworkConnectivity(StrEnum):
    FULL = "full"
    LIMITED = "limited"
    PORTAL = "portal"
    NONE = "none"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class NetworkCounters:
    """Monotonic link counters; never packet contents."""

    received_bytes: int
    transmitted_bytes: int
    received_errors: int
    transmitted_errors: int
    received_drops: int
    transmitted_drops: int


@dataclass(frozen=True, slots=True)
class NetworkLinkSample:
    """One aggregate NetworkManager/link observation."""

    stable_id: str
    kind: NetworkLinkKind
    state: NetworkLinkState
    connectivity: NetworkConnectivity
    carrier: bool
    metered: bool
    default_route: bool
    signal_percent: int | None
    counters: NetworkCounters
    observed_at_ms: int


@dataclass(frozen=True, slots=True)
class NetworkSnapshot:
    """Bounded-source snapshot used by live adapters and replay tests."""

    status: SourceStatus
    observed_at_ms: int
    links: tuple[NetworkLinkSample, ...]


def build_record_error(record: object) -> str:
    if not isinstance(record, BuildRecord):
        return "build record has an invalid type"
    if not isinstance(record.outcome, BuildOutcome):
        return "build outcome is invalid"
    if not isinstance(record.build_id, str) or not 1 <= len(record.build_id) <= 120:
        return "build identity is invalid"
    timing = _build_timing_error(record)
    if timing:
        return timing
    for field, name in (
        (record.changed_paths, "changed paths"),
        (record.failed_checks, "failed checks"),
    ):
        if not isinstance(field, tuple) or any(not isinstance(item, str) for item in field):
            return f"build {name} are invalid"
    return ""


def _build_timing_error(record: BuildRecord) -> str:
    for value, name in ((record.duration_ms, "duration"), (record.started_at_ms, "start")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return f"build {name} is invalid"
    if record.duration_ms > MAX_BUILD_DURATION_MS:
        return "build duration is implausible"
    return ""


def hardware_sample_error(sample: object) -> str:
    shape = _hardware_sample_shape_error(sample)
    if shape:
        return shape
    assert isinstance(sample, HardwareSample)
    if sample.health is SensorHealth.MISSING:
        return "" if sample.value is None else "a missing hardware sample must not carry a value"
    if sample.value is None:
        return "a present hardware sample must carry a value"
    maximum = _hardware_ceiling(sample.kind)
    if isinstance(sample.value, bool) or not isinstance(sample.value, int):
        return f"hardware sample value must be an integer in [0, {maximum}]"
    if not 0 <= sample.value <= maximum:
        return f"hardware sample value must be an integer in [0, {maximum}]"
    return ""


def _hardware_sample_shape_error(sample: object) -> str:
    if not isinstance(sample, HardwareSample):
        return "hardware sample has an invalid type"
    if not isinstance(sample.kind, SensorKind):
        return "hardware sample kind is invalid"
    if not isinstance(sample.health, SensorHealth):
        return "hardware sample health is invalid"
    if not isinstance(sample.unit, str) or len(sample.unit) > 16:
        return "hardware sample unit is invalid"
    if not isinstance(sample.label, str) or len(sample.label) > 80:
        return "hardware sample label is invalid"
    return ""


def _hardware_ceiling(kind: SensorKind) -> int:
    if kind is SensorKind.THERMAL:
        return MAX_TEMPERATURE_MILLIDEGREES
    if kind is SensorKind.MEMORY_ERRORS:
        return MAX_ERROR_COUNT
    if kind in (SensorKind.POWER, SensorKind.SERVICE):
        return MAX_PERCENT
    return MAX_ERROR_COUNT


def network_snapshot_error(snapshot: object) -> str:
    error = _network_snapshot_shape_error(snapshot)
    if error:
        return error
    assert isinstance(snapshot, NetworkSnapshot)
    identities: set[str] = set()
    for link in snapshot.links:
        error = _network_link_error(link, snapshot.observed_at_ms)
        if error:
            return error
        if link.stable_id in identities:
            return f"duplicate network link identity: {link.stable_id}"
        identities.add(link.stable_id)
    return ""


def _network_snapshot_shape_error(snapshot: object) -> str:
    if not isinstance(snapshot, NetworkSnapshot):
        return "network snapshot is not typed"
    if not isinstance(snapshot.status, SourceStatus):
        return "network source status is invalid"
    if _strict_non_negative_integer_error(snapshot.observed_at_ms):
        return "network snapshot timestamp is invalid"
    if not isinstance(snapshot.links, tuple):
        return "network links must be a tuple"
    if len(snapshot.links) > MAX_NETWORK_LINKS:
        return f"network snapshot exceeds {MAX_NETWORK_LINKS} links"
    return ""


def _network_link_error(link: object, snapshot_time_ms: int) -> str:
    if not isinstance(link, NetworkLinkSample):
        return "network link sample is not typed"
    if not isinstance(link.stable_id, str) or not NETWORK_STABLE_ID.fullmatch(link.stable_id):
        return "network link stable identity is invalid"
    enum_error = _network_link_enum_error(link)
    if enum_error:
        return enum_error
    if any(type(value) is not bool for value in (link.carrier, link.metered, link.default_route)):
        return f"network link flags are invalid: {link.stable_id}"
    if link.signal_percent is not None and (
        type(link.signal_percent) is not int or not 0 <= link.signal_percent <= 100
    ):
        return f"network link signal is invalid: {link.stable_id}"
    counter_error = _network_counter_error(link.counters)
    if counter_error:
        return f"{counter_error}: {link.stable_id}"
    if (
        _strict_non_negative_integer_error(link.observed_at_ms)
        or link.observed_at_ms > snapshot_time_ms
    ):
        return f"network link timestamp is invalid: {link.stable_id}"
    return ""


def _network_link_enum_error(link: NetworkLinkSample) -> str:
    if not isinstance(link.kind, NetworkLinkKind):
        return f"network link kind is invalid: {link.stable_id}"
    if not isinstance(link.state, NetworkLinkState):
        return f"network link state is invalid: {link.stable_id}"
    if not isinstance(link.connectivity, NetworkConnectivity):
        return f"network link connectivity is invalid: {link.stable_id}"
    return ""


def _network_counter_error(counters: object) -> str:
    if not isinstance(counters, NetworkCounters):
        return "network counters are not typed"
    values = (
        counters.received_bytes,
        counters.transmitted_bytes,
        counters.received_errors,
        counters.transmitted_errors,
        counters.received_drops,
        counters.transmitted_drops,
    )
    if any(type(value) is not int or not 0 <= value <= MAX_NETWORK_COUNTER for value in values):
        return "network counters are invalid"
    return ""


def _strict_non_negative_integer_error(value: object) -> bool:
    return type(value) is not int or value < 0
