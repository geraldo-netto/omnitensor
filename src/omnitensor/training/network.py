"""Privacy-preserving local network anomaly source training.

The recipe borrows KitNET's small reconstruction-model idea, but it never
parses packets and does not vendor KitNET code or weights.  It reduces the
existing consent-gated link snapshots to identity-free aggregate deltas, fits
only operator-confirmed normal history, and exports one portable ONNX graph.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from omnitensor.preparation import file_digest
from omnitensor.telemetry_types import (
    MAX_NETWORK_COUNTER,
    NetworkConnectivity,
    NetworkCounters,
    NetworkLinkKind,
    NetworkLinkSample,
    NetworkLinkState,
    NetworkSnapshot,
    SourceStatus,
    network_snapshot_error,
)

from .contracts import TrainingError, write_training_report
from .tabular import (
    JsonlPolicy,
    checked_model,
    floor_chronological_split,
    load_bounded_jsonl,
    load_onnx_dependency,
    numeric_training_report,
    publish_numeric_training,
    save_onnx_atomic,
)

NETWORK_RECIPE = "aggregate-reconstruction-v1"
NORMAL_ONLY_CONFIRMATION = "I-confirm-this-replay-is-normal"
KITSUNE_PAPER_URI = "https://arxiv.org/abs/1802.09089"
KITSUNE_CODE_URI = "https://github.com/ymirsky/Kitsune-py"
KITSUNE_CITATION = "Mirsky, Doitshman, Elovici, and Shabtai, Kitsune, NDSS 2018"
KITSUNE_CODE_LICENSE = "MIT"
NETWORK_FEATURES = (
    "receivedBytesPerSecond",
    "transmittedBytesPerSecond",
    "receivedErrorsPerSecond",
    "transmittedErrorsPerSecond",
    "receivedDropsPerSecond",
    "transmittedDropsPerSecond",
    "linkCount",
    "upFraction",
    "degradedFraction",
    "constrainedConnectivityFraction",
    "meteredFraction",
    "carrierFraction",
    "defaultRouteFraction",
    "meanSignalFraction",
)
FEATURE_GROUPS = ((0, 1), (2, 3, 4, 5), (6, 7, 8, 9, 10, 11, 12, 13))
MAX_REPLAY_BYTES = 64 * 1024 * 1024
MAX_REPLAY_LINE_BYTES = 1024 * 1024
MAX_REPLAY_SNAPSHOTS = 200_000
DEFAULT_MIN_TRAINING_ROWS = 64
DEFAULT_MIN_HOLDOUT_ROWS = 16
DEFAULT_MAX_FALSE_POSITIVE_RATE = 0.1
_SNAPSHOT_KEYS = {
    "schemaVersion",
    "source",
    "sourceHealth",
    "observedAtMs",
    "links",
    "truncatedLinks",
}
_LINK_KEYS = {
    "stableId",
    "kind",
    "state",
    "connectivity",
    "carrier",
    "metered",
    "defaultRoute",
    "signalPercent",
    "counters",
    "observedAtMs",
}
_COUNTER_KEYS = {
    "receivedBytes",
    "transmittedBytes",
    "receivedErrors",
    "transmittedErrors",
    "receivedDrops",
    "transmittedDrops",
}


def _replay_policy() -> JsonlPolicy:
    return JsonlPolicy(
        MAX_REPLAY_BYTES,
        MAX_REPLAY_SNAPSHOTS,
        MAX_REPLAY_LINE_BYTES,
        "replay-invalid",
        "network replay is not a safe regular file",
        "replay-too-large",
        "network replay byte limit exceeded",
        "network replay snapshot limit exceeded",
        "replay-invalid",
        "network replay line {line_number} is invalid",
        "network replay line {line_number}: {detail}",
    )


@dataclass(frozen=True, slots=True)
class NetworkFeatureRow:
    observed_at_ms: int
    features: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class NetworkDataset:
    rows: tuple[NetworkFeatureRow, ...]
    replay_sha256: str
    snapshots: int


@dataclass(frozen=True, slots=True)
class NetworkReconstructionModel:
    means: tuple[float, ...]
    scales: tuple[float, ...]
    projection: tuple[tuple[float, ...], ...]

    def score(self, features: Sequence[float]) -> float:
        if len(features) != len(NETWORK_FEATURES):
            raise TrainingError("features-invalid", "network feature width disagrees")
        normalized = []
        for value, mean, scale in zip(features, self.means, self.scales, strict=True):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TrainingError("features-invalid", "network features must be numbers")
            if not math.isfinite(value):
                raise TrainingError("features-invalid", "network features must be finite")
            normalized.append((float(value) - mean) / scale)
        reconstruction = tuple(
            sum(
                normalized[column] * self.projection[column][output]
                for column in range(len(normalized))
            )
            for output in range(len(normalized))
        )
        return sum(
            (value - restored) ** 2
            for value, restored in zip(normalized, reconstruction, strict=True)
        ) / len(normalized)


class NetworkModelExporter(Protocol):
    def export(self, model: NetworkReconstructionModel, destination: Path) -> None: ...


def load_network_replay(path: Path | str) -> NetworkDataset:
    """Load closed collector documents without retaining link identities."""
    snapshots, digest = load_bounded_jsonl(
        path,
        _replay_policy(),
        _snapshot_from_document,
    )
    rows = network_feature_rows(snapshots)
    return NetworkDataset(rows, digest, len(snapshots))


def network_feature_rows(snapshots: Sequence[NetworkSnapshot]) -> tuple[NetworkFeatureRow, ...]:
    """Aggregate snapshots to fixed, identity-free counter and state features."""
    if isinstance(snapshots, (str, bytes)) or not isinstance(snapshots, Sequence):
        raise TrainingError("replay-invalid", "network snapshots must be a sequence")
    if len(snapshots) < 2:
        raise TrainingError("insufficient-history", "network replay needs at least two snapshots")
    rows = []
    previous = snapshots[0]
    _require_training_snapshot(previous)
    for current in snapshots[1:]:
        _require_training_snapshot(current)
        elapsed_ms = current.observed_at_ms - previous.observed_at_ms
        if elapsed_ms <= 0:
            raise TrainingError("observations-unordered", "network snapshot times must increase")
        rows.append(
            NetworkFeatureRow(
                current.observed_at_ms,
                _aggregate_features(previous, current),
            )
        )
        previous = current
    return tuple(rows)


class NetworkAnomalyTrainer:
    """Fit and gate one normal-only aggregate reconstruction source."""

    def __init__(
        self,
        exporter: NetworkModelExporter | None = None,
        *,
        confirm_normal: str,
        minimum_training_rows: int = DEFAULT_MIN_TRAINING_ROWS,
        minimum_holdout_rows: int = DEFAULT_MIN_HOLDOUT_ROWS,
        maximum_false_positive_rate: float = DEFAULT_MAX_FALSE_POSITIVE_RATE,
    ) -> None:
        bounds = (minimum_training_rows, minimum_holdout_rows)
        if confirm_normal != NORMAL_ONLY_CONFIRMATION:
            raise TrainingError(
                "normal-baseline-not-confirmed",
                f"review the replay and pass exactly {NORMAL_ONLY_CONFIRMATION}",
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in bounds
        ):
            raise ValueError("network split row bounds must be positive integers")
        if (
            isinstance(maximum_false_positive_rate, bool)
            or not isinstance(maximum_false_positive_rate, (int, float))
            or not math.isfinite(maximum_false_positive_rate)
            or not 0 <= maximum_false_positive_rate <= 1
        ):
            raise ValueError("maximum false-positive rate must be in [0, 1]")
        self._exporter = exporter or OnnxNetworkExporter()
        self._minimum_training_rows = minimum_training_rows
        self._minimum_holdout_rows = minimum_holdout_rows
        self._maximum_false_positive_rate = float(maximum_false_positive_rate)

    def train(self, dataset: NetworkDataset, output_dir: Path | str) -> dict:
        training, holdout = _chronological_split(
            dataset.rows,
            self._minimum_training_rows,
            self._minimum_holdout_rows,
        )
        model = _fit_reconstruction(training)
        training_scores = tuple(model.score(row.features) for row in training)
        holdout_scores = tuple(model.score(row.features) for row in holdout)
        threshold = max(1e-12, _percentile(training_scores, 0.99))
        false_positives = sum(score > threshold for score in holdout_scores)
        false_positive_rate = false_positives / len(holdout_scores)
        if false_positive_rate > self._maximum_false_positive_rate:
            raise TrainingError(
                "model-not-stable",
                "held-out normal replay exceeds the false-positive gate",
            )
        return publish_numeric_training(
            output_dir,
            model,
            self._exporter,
            report_name="network-training-report.json",
            report_prefix=".network-training-report-",
            report=lambda model_path: _training_report(
                dataset,
                training,
                holdout,
                training_scores,
                holdout_scores,
                threshold,
                false_positive_rate,
                self._maximum_false_positive_rate,
                model_path,
            ),
            writer=write_training_report,
        )


class OnnxNetworkExporter:
    """Export normalization plus block reconstruction error using common ops."""

    def export(self, model: NetworkReconstructionModel, destination: Path) -> None:
        onnx, tensor_proto, helper = load_onnx_dependency()
        width = len(NETWORK_FEATURES)
        flat_projection = [value for row in model.projection for value in row]
        graph = helper.make_graph(
            [
                helper.make_node("Sub", ["input", "means"], ["centered"]),
                helper.make_node("Div", ["centered", "scales"], ["normalized"]),
                helper.make_node("MatMul", ["normalized", "projection"], ["reconstruction"]),
                helper.make_node("Sub", ["normalized", "reconstruction"], ["residual"]),
                helper.make_node("Mul", ["residual", "residual"], ["squared"]),
                helper.make_node("ReduceMean", ["squared"], ["score"], axes=[1], keepdims=1),
            ],
            "omnitensor-network-aggregate-reconstruction",
            [helper.make_tensor_value_info("input", tensor_proto.FLOAT, [1, width])],
            [helper.make_tensor_value_info("score", tensor_proto.FLOAT, [1, 1])],
            [
                helper.make_tensor("means", tensor_proto.FLOAT, [width], model.means),
                helper.make_tensor("scales", tensor_proto.FLOAT, [width], model.scales),
                helper.make_tensor(
                    "projection", tensor_proto.FLOAT, [width, width], flat_projection
                ),
            ],
        )
        portable = checked_model(graph, onnx, helper)
        save_onnx_atomic(portable, destination, onnx, prefix=".network-model-")


def _training_report(
    dataset: NetworkDataset,
    training: Sequence[NetworkFeatureRow],
    holdout: Sequence[NetworkFeatureRow],
    training_scores: Sequence[float],
    holdout_scores: Sequence[float],
    threshold: float,
    false_positive_rate: float,
    maximum_false_positive_rate: float,
    model_path: Path,
) -> dict:
    family_fields = {
        "version": 1,
        "recipe": NETWORK_RECIPE,
        "inspiration": {
            "paper": KITSUNE_PAPER_URI,
            "code": KITSUNE_CODE_URI,
            "citation": KITSUNE_CITATION,
            "codeLicense": KITSUNE_CODE_LICENSE,
            "codeVendored": False,
        },
        "dataset": {
            "replaySha256": dataset.replay_sha256,
            "snapshots": dataset.snapshots,
            "identityPersisted": False,
            "packetContentIngested": False,
            "operatorConfirmedNormal": True,
        },
        "split": {
            "kind": "chronological",
            "training": len(training),
            "holdout": len(holdout),
            "holdoutFromMs": holdout[0].observed_at_ms,
        },
        "features": list(NETWORK_FEATURES),
        "quality": {
            "threshold": threshold,
            "trainingP95Score": _percentile(training_scores, 0.95),
            "holdoutP95Score": _percentile(holdout_scores, 0.95),
            "holdoutFalsePositiveRate": false_positive_rate,
            "maximumFalsePositiveRate": maximum_false_positive_rate,
            "anomalyRecall": None,
        },
    }
    return numeric_training_report(
        family_fields,
        model_path,
        len(NETWORK_FEATURES),
        digest=file_digest,
    )


def _snapshot_from_document(document: object) -> NetworkSnapshot:
    if not isinstance(document, dict) or set(document) != _SNAPSHOT_KEYS:
        raise TrainingError("snapshot-invalid", "network snapshot fields are invalid")
    schema_version = document.get("schemaVersion")
    if (
        type(schema_version) is not int
        or schema_version != 1
        or document.get("source") != "network-manager-link-metadata"
    ):
        raise TrainingError("snapshot-invalid", "network snapshot identity is invalid")
    truncated_links = document.get("truncatedLinks")
    if (
        type(truncated_links) is not int
        or truncated_links != 0
        or not isinstance(document.get("links"), list)
    ):
        raise TrainingError("snapshot-invalid", "network snapshot must contain every link")
    try:
        status = SourceStatus(document["sourceHealth"])
        links = tuple(_link_from_document(item) for item in document["links"])
        snapshot = NetworkSnapshot(status, document["observedAtMs"], links)
    except (TypeError, ValueError) as error:
        raise TrainingError("snapshot-invalid", "network snapshot values are invalid") from error
    detail = network_snapshot_error(snapshot)
    if detail:
        raise TrainingError("snapshot-invalid", detail)
    return snapshot


def _link_from_document(document: object) -> NetworkLinkSample:
    if not isinstance(document, dict) or set(document) != _LINK_KEYS:
        raise TrainingError("snapshot-invalid", "network link fields are invalid")
    counters = document.get("counters")
    if not isinstance(counters, dict) or set(counters) != _COUNTER_KEYS:
        raise TrainingError("snapshot-invalid", "network counter fields are invalid")
    try:
        return NetworkLinkSample(
            document["stableId"],
            NetworkLinkKind(document["kind"]),
            NetworkLinkState(document["state"]),
            NetworkConnectivity(document["connectivity"]),
            document["carrier"],
            document["metered"],
            document["defaultRoute"],
            document["signalPercent"],
            NetworkCounters(
                counters["receivedBytes"],
                counters["transmittedBytes"],
                counters["receivedErrors"],
                counters["transmittedErrors"],
                counters["receivedDrops"],
                counters["transmittedDrops"],
            ),
            document["observedAtMs"],
        )
    except (TypeError, ValueError) as error:
        raise TrainingError("snapshot-invalid", "network link values are invalid") from error


def _require_training_snapshot(snapshot: object) -> None:
    detail = network_snapshot_error(snapshot)
    if detail:
        raise TrainingError("snapshot-invalid", detail)
    assert isinstance(snapshot, NetworkSnapshot)
    if snapshot.status is not SourceStatus.READY:
        raise TrainingError(
            "snapshot-unhealthy", "normal training requires ready network snapshots"
        )


def _aggregate_features(previous: NetworkSnapshot, current: NetworkSnapshot) -> tuple[float, ...]:
    prior = {link.stable_id: link for link in previous.links}
    elapsed_seconds = (current.observed_at_ms - previous.observed_at_ms) / 1000
    deltas = [0] * 6
    for link in current.links:
        older = prior.get(link.stable_id)
        if older is None:
            continue
        before = _counter_values(older.counters)
        after = _counter_values(link.counters)
        if any(new < old for old, new in zip(before, after, strict=True)):
            raise TrainingError("counter-regression", "network counters decreased within replay")
        for index, (old, new) in enumerate(zip(before, after, strict=True)):
            deltas[index] += new - old
    count = len(current.links)
    states = [link.state for link in current.links]
    connectivity = [link.connectivity for link in current.links]
    signals = [
        link.signal_percent / 100 for link in current.links if link.signal_percent is not None
    ]
    constrained = {
        NetworkConnectivity.LIMITED,
        NetworkConnectivity.PORTAL,
        NetworkConnectivity.NONE,
    }
    features = (
        *(delta / elapsed_seconds for delta in deltas),
        float(count),
        _fraction(states, NetworkLinkState.UP),
        _fraction(states, NetworkLinkState.DEGRADED),
        sum(value in constrained for value in connectivity) / max(1, count),
        sum(link.metered for link in current.links) / max(1, count),
        sum(link.carrier for link in current.links) / max(1, count),
        sum(link.default_route for link in current.links) / max(1, count),
        sum(signals) / len(signals) if signals else 0.0,
    )
    if len(features) != len(NETWORK_FEATURES) or not all(
        math.isfinite(value) for value in features
    ):
        raise TrainingError("features-invalid", "aggregate network features are invalid")
    return tuple(float(value) for value in features)


def _counter_values(counters: NetworkCounters) -> tuple[int, ...]:
    values = (
        counters.received_bytes,
        counters.transmitted_bytes,
        counters.received_errors,
        counters.transmitted_errors,
        counters.received_drops,
        counters.transmitted_drops,
    )
    if any(value > MAX_NETWORK_COUNTER for value in values):
        raise TrainingError("snapshot-invalid", "network counters exceed the supported range")
    return values


def _fraction(values: Sequence[object], expected: object) -> float:
    return sum(value == expected for value in values) / max(1, len(values))


def _chronological_split(
    rows: Sequence[NetworkFeatureRow], minimum_training: int, minimum_holdout: int
) -> tuple[tuple[NetworkFeatureRow, ...], tuple[NetworkFeatureRow, ...]]:
    return floor_chronological_split(
        rows,
        minimum_training,
        minimum_holdout,
        insufficient_detail="network replay does not meet split row floors",
    )


def _fit_reconstruction(rows: Sequence[NetworkFeatureRow]) -> NetworkReconstructionModel:
    width = len(NETWORK_FEATURES)
    means = tuple(sum(row.features[index] for row in rows) / len(rows) for index in range(width))
    scales = tuple(
        max(
            1e-9,
            math.sqrt(sum((row.features[index] - means[index]) ** 2 for row in rows) / len(rows)),
        )
        for index in range(width)
    )
    normalized = tuple(
        tuple(
            (value - mean) / scale
            for value, mean, scale in zip(row.features, means, scales, strict=True)
        )
        for row in rows
    )
    projection = [[0.0] * width for _index in range(width)]
    for group in FEATURE_GROUPS:
        direction = _principal_direction(normalized, group)
        for local_row, row_index in enumerate(group):
            for local_column, column_index in enumerate(group):
                projection[row_index][column_index] = direction[local_row] * direction[local_column]
    return NetworkReconstructionModel(means, scales, tuple(tuple(row) for row in projection))


def _principal_direction(
    rows: Sequence[Sequence[float]], group: Sequence[int]
) -> tuple[float, ...]:
    covariance = tuple(
        tuple(sum(row[left] * row[right] for row in rows) / len(rows) for right in group)
        for left in group
    )
    direction = tuple(1 / math.sqrt(len(group)) for _index in group)
    for _step in range(50):
        candidate = tuple(
            sum(covariance[row][column] * direction[column] for column in range(len(group)))
            for row in range(len(group))
        )
        norm = math.sqrt(sum(value * value for value in candidate))
        if norm <= 1e-12:
            return (1.0, *(0.0 for _index in group[1:]))
        direction = tuple(value / norm for value in candidate)
    return direction


def _percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise TrainingError("insufficient-history", "cannot score an empty replay split")
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * quantile) - 1)
    return ordered[index]
