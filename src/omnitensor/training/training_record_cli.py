"""CLI ownership for explicit telemetry and runtime-snapshot recording."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from omnitensor.telemetry_recorder import TelemetryRecorder
from omnitensor.telemetry_types import RecorderError

from .contracts import TrainingError
from .desktop_history import DESKTOP_REVOCATION_CONFIRMATION, revoke_desktop_history
from .snapshot_recording import load_runtime_snapshot, record_runtime_snapshot

DEFAULT_RECORDS_ROOT = "~/.local/state/omnitensor/telemetry"
DEFAULT_SNAPSHOT_PATH = "~/.local/state/xpu-workload-manager/state.json"


@dataclass(frozen=True, slots=True)
class RecordCliDependencies:
    recorder_type: object = TelemetryRecorder
    snapshot_loader: object = load_runtime_snapshot
    snapshot_recorder: object = record_runtime_snapshot
    desktop_revoker: object = revoke_desktop_history


def record_main(
    argv: list[str] | None = None,
    *,
    dependencies: RecordCliDependencies | None = None,
    records_root_default: str = DEFAULT_RECORDS_ROOT,
    feature_parser=None,
) -> int:
    deps = dependencies or RecordCliDependencies()
    parse_features = feature_parser or _feature_values
    parser = argparse.ArgumentParser(
        prog="omnitensor-record-training-sample",
        description="Append one explicit numeric sample to bounded local training history",
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--feature",
        required=True,
        action="append",
        help="numeric NAME=VALUE; repeat for every feature",
    )
    parser.add_argument("--observed-at-ms", default=None, type=int)
    parser.add_argument("--records-root", default=records_root_default)
    arguments = parser.parse_args(argv)
    try:
        row = deps.recorder_type(Path(arguments.records_root).expanduser()).record(
            arguments.profile,
            parse_features(arguments.feature),
            arguments.observed_at_ms,
        )
    except (RecorderError, TrainingError, OSError, ValueError) as error:
        print(f"recording failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(row.document(), indent=2))
    return 0


def snapshot_record_main(
    argv: list[str] | None = None,
    *,
    dependencies: RecordCliDependencies | None = None,
    records_root_default: str = DEFAULT_RECORDS_ROOT,
    snapshot_default: str | None = None,
) -> int:
    deps = dependencies or RecordCliDependencies()
    parser = argparse.ArgumentParser(
        prog="omnitensor-record-runtime-snapshot",
        description="Append allowlisted aggregate runtime measurements to local training history",
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--selector",
        required=True,
        action="append",
        help=("queueDepth or runningProfiles; repeat to preserve the desired feature order"),
    )
    parser.add_argument(
        "--snapshot",
        default=(
            snapshot_default
            if snapshot_default is not None
            else os.environ.get("OMNITENSOR_STATE_PATH", DEFAULT_SNAPSHOT_PATH)
        ),
    )
    parser.add_argument("--records-root", default=records_root_default)
    arguments = parser.parse_args(argv)
    try:
        document = deps.snapshot_loader(Path(arguments.snapshot).expanduser())
        row = deps.snapshot_recorder(
            document,
            profile_id=arguments.profile,
            selectors=arguments.selector,
            recorder=deps.recorder_type(Path(arguments.records_root).expanduser()),
        )
    except (RecorderError, OSError, ValueError) as error:
        print(f"snapshot recording failed: {error}", file=sys.stderr)
        return 1
    if row is None:
        print(
            json.dumps(
                {
                    "status": "duplicate",
                    "profileId": arguments.profile,
                    "observedAtMs": document["generatedAt"],
                },
                indent=2,
            )
        )
        return 0
    output = row.document()
    output["status"] = "recorded"
    print(json.dumps(output, indent=2))
    return 0


def _feature_values(items: list[str]) -> dict[str, float]:
    values: dict[str, float] = {}
    for item in items:
        name, separator, raw = item.partition("=")
        if not separator or not name or name in values:
            raise TrainingError("features-invalid", "features must be unique NAME=VALUE pairs")
        try:
            values[name] = float(raw)
        except ValueError as error:
            raise TrainingError("features-invalid", f"feature {name} is not numeric") from error
    return values


_feature_values.__module__ = "omnitensor.training.cli"
feature_values = _feature_values


def desktop_revoke_main(
    argv: list[str] | None = None,
    *,
    dependencies: RecordCliDependencies | None = None,
) -> int:
    deps = dependencies or RecordCliDependencies()
    parser = argparse.ArgumentParser(
        prog="omnitensor-revoke-desktop-training-history",
        description="Delete one explicit desktop training history file safely",
    )
    parser.add_argument("--history", required=True)
    parser.add_argument(
        "--confirm-delete",
        required=True,
        help=f"pass exactly {DESKTOP_REVOCATION_CONFIRMATION}",
    )
    arguments = parser.parse_args(argv)
    try:
        removed = deps.desktop_revoker(arguments.history, arguments.confirm_delete)
    except (TrainingError, OSError, ValueError) as error:
        print(f"desktop history revocation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"removed": removed}, indent=2))
    return 0
