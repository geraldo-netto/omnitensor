"""Command-line surface for offline fit and native installation."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from ..conversion import ConversionError
from ..forecastresult import parse_forecast_reading
from ..plugins.artifact_installation import ArtifactInstallationError
from ..plugins.forecasting import ForecastError
from ..plugins.recorder import RecorderError, TelemetryRecorder
from ..preparation import PreparationError
from .contracts import TrainingError, TrainingSpec
from .forecast import ForecastTrainer
from .installation import available_targets, install_training
from .runner import (
    DbusForecastClient,
    ForecastRunError,
    TrustedForecastRunner,
    load_forecast_binding,
)
from .snapshot_recording import load_runtime_snapshot, record_runtime_snapshot

DEFAULT_RECORDS_ROOT = "~/.local/state/omnitensor/telemetry"
DEFAULT_SNAPSHOT_PATH = "~/.local/state/tpu-workload-manager/state.json"
DEFAULT_OUTPUT_ROOT = "~/.local/share/omnitensor/training"
DEFAULT_ARTIFACT_ROOT = "~/.local/share/omnitensor/artifacts"
DEFAULT_BINDINGS_ROOT = "~/.local/share/omnitensor/model-bindings"


def record_main(argv: list[str] | None = None) -> int:
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
    parser.add_argument("--records-root", default=DEFAULT_RECORDS_ROOT)
    arguments = parser.parse_args(argv)
    try:
        row = TelemetryRecorder(Path(arguments.records_root).expanduser()).record(
            arguments.profile,
            _feature_values(arguments.feature),
            arguments.observed_at_ms,
        )
    except (RecorderError, TrainingError, OSError, ValueError) as error:
        print(f"recording failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(row.document(), indent=2))
    return 0


def snapshot_record_main(argv: list[str] | None = None) -> int:
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
        "--snapshot", default=os.environ.get("OMNITENSOR_STATE_PATH", DEFAULT_SNAPSHOT_PATH)
    )
    parser.add_argument("--records-root", default=DEFAULT_RECORDS_ROOT)
    arguments = parser.parse_args(argv)
    try:
        document = load_runtime_snapshot(Path(arguments.snapshot).expanduser())
        row = record_runtime_snapshot(
            document,
            profile_id=arguments.profile,
            selectors=arguments.selector,
            recorder=TelemetryRecorder(Path(arguments.records_root).expanduser()),
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


def train_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="omnitensor-train-model",
        description="Fit a portable model from bounded local OmniTensor telemetry",
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--id", required=True, dest="artifact_id")
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--features",
        required=True,
        help="ordered comma-separated features; predicted feature first",
    )
    parser.add_argument("--target", required=True, dest="target_feature")
    parser.add_argument("--window", required=True, type=int)
    parser.add_argument("--horizon", default=1, type=int)
    parser.add_argument("--records-root", default=DEFAULT_RECORDS_ROOT)
    parser.add_argument("--output-dir", default=None)
    arguments = parser.parse_args(argv)
    output = (
        Path(arguments.output_dir).expanduser()
        if arguments.output_dir
        else Path(DEFAULT_OUTPUT_ROOT).expanduser() / arguments.profile / arguments.version
    )
    try:
        spec = TrainingSpec(
            arguments.profile,
            arguments.artifact_id,
            arguments.version,
            _features(arguments.features),
            arguments.target_feature,
            arguments.window,
            arguments.horizon,
        )
        report = ForecastTrainer().train(
            spec,
            Path(arguments.records_root).expanduser(),
            output,
        )
    except (TrainingError, ForecastError, OSError, ValueError) as error:
        print(f"training failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "report": str(output / "training-report.json"),
                "portableModel": str(output / "model.onnx"),
                "samples": report.samples,
                "quality": report.quality,
                "next": "omnitensor-install-trained-model "
                f"{output / 'training-report.json'} --targets auto",
            },
            indent=2,
        )
    )
    return 0


def install_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="omnitensor-install-trained-model",
        description="Compile and install a trained model for local accelerator lanes",
    )
    parser.add_argument("report")
    parser.add_argument(
        "--targets",
        default="auto",
        help="auto or comma-separated npu,gpu; TPU needs separate int8 export tooling",
    )
    parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--bindings-root", default=DEFAULT_BINDINGS_ROOT)
    parser.add_argument("--build-root", default=None)
    arguments = parser.parse_args(argv)
    try:
        targets = (
            available_targets() if arguments.targets == "auto" else _targets(arguments.targets)
        )
        if not targets:
            raise TrainingError(
                "compiler-missing", "no native compiler found; install [convert] or [convert-npu]"
            )
        installed = install_training(
            Path(arguments.report).expanduser(),
            targets=targets,
            artifact_root=Path(arguments.artifact_root).expanduser(),
            bindings_root=Path(arguments.bindings_root).expanduser(),
            build_root=(Path(arguments.build_root).expanduser() if arguments.build_root else None),
        )
    except (
        ArtifactInstallationError,
        ConversionError,
        ForecastError,
        PreparationError,
        TrainingError,
        OSError,
        ValueError,
    ) as error:
        print(f"installation failed: {error}", file=sys.stderr)
        return 1
    document = installed.document()
    document["next"] = "systemctl --user restart omnitensor.service"
    print(json.dumps(document, indent=2))
    return 0


def forecast_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="omnitensor-run-forecast",
        description="Submit one forecast assembled only from trusted local history",
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--records-root", default=DEFAULT_RECORDS_ROOT)
    parser.add_argument("--bindings-root", default=DEFAULT_BINDINGS_ROOT)
    parser.add_argument("--attempts", default=40, type=int)
    parser.add_argument("--poll-interval", default=0.1, type=float)
    arguments = parser.parse_args(argv)

    async def execute() -> dict:
        workload = load_forecast_binding(
            arguments.profile, Path(arguments.bindings_root).expanduser()
        )
        client = await DbusForecastClient.connect()
        try:
            return await TrustedForecastRunner(
                workload,
                TelemetryRecorder(Path(arguments.records_root).expanduser()),
                client,
                attempts=arguments.attempts,
                poll_interval=arguments.poll_interval,
            ).run()
        finally:
            client.close()

    try:
        output = asyncio.run(execute())
        reading = parse_forecast_reading(output.get("reading"))
        if reading is None:
            raise ForecastRunError(
                "runtime-response-invalid", "runtime returned no valid forecast reading"
            )
    except (ForecastRunError, OSError, ValueError) as error:
        print(f"forecast failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(reading, indent=2, allow_nan=False))
    return 0


def _features(text: str) -> tuple[str, ...]:
    features = tuple(part.strip() for part in text.split(",") if part.strip())
    if not features:
        raise TrainingError("features-invalid", "at least one feature is required")
    return features


def _targets(text: str) -> tuple[str, ...]:
    targets = tuple(part.strip() for part in text.split(",") if part.strip())
    if not targets:
        raise TrainingError("targets-invalid", "at least one target is required")
    return targets


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
