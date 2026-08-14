"""Command-line surface for offline fit and native installation."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from omnitensor.forecasting import ForecastError
from omnitensor.preparation import ArtifactInstallationError, PreparationError
from omnitensor.telemetry_recorder import TelemetryRecorder
from omnitensor.telemetry_types import RecorderError
from omnitensor.training.desktop_history import (
    DESKTOP_REVOCATION_CONFIRMATION,
    revoke_desktop_history,
)

from ..conversion import ConversionError
from ..forecastresult import parse_forecast_reading
from .build import (
    BUILD_PROVENANCE_CONFIRMATION,
    BuildAdvisorTrainer,
    load_build_history,
)
from .contracts import TrainingError, TrainingSpec
from .desktop import (
    DesktopSuggestionTrainer,
    load_desktop_history,
)
from .forecast import ForecastTrainer
from .hardware import (
    HARDWARE_LABEL_CONFIRMATION,
    HardwareHealthTrainer,
    load_hardware_history,
    parse_sensor_bindings,
)
from .installation import available_targets, install_training
from .network import (
    NORMAL_ONLY_CONFIRMATION,
    NetworkAnomalyTrainer,
    load_network_replay,
)
from .runner import (
    DbusForecastClient,
    ForecastRunError,
    TrustedForecastRunner,
    load_forecast_binding,
)
from .snapshot_recording import load_runtime_snapshot, record_runtime_snapshot
from .storage import (
    BACKBLAZE_TERMS,
    BackblazeDatasetBuilder,
    StorageTrainer,
)

DEFAULT_RECORDS_ROOT = "~/.local/state/omnitensor/telemetry"
DEFAULT_SNAPSHOT_PATH = "~/.local/state/xpu-workload-manager/state.json"
DEFAULT_OUTPUT_ROOT = "~/.local/share/omnitensor/training"
DEFAULT_ARTIFACT_ROOT = "~/.local/share/omnitensor/artifacts"
DEFAULT_BINDINGS_ROOT = "~/.local/share/omnitensor/model-bindings"
DEFAULT_STORAGE_OUTPUT = "~/.local/share/omnitensor/training/storage-intelligence/local"
DEFAULT_NETWORK_OUTPUT = "~/.local/share/omnitensor/training/network-peripherals/local"
DEFAULT_BUILD_OUTPUT = "~/.local/share/omnitensor/training/build-advisor/local"
DEFAULT_HARDWARE_OUTPUT = "~/.local/share/omnitensor/training/hardware-health/local"
DEFAULT_DESKTOP_OUTPUT = "~/.local/share/omnitensor/training/desktop-context/local"


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


def storage_train_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="omnitensor-train-storage-model",
        description="Fit a portable storage risk source from official Backblaze daily CSVs",
    )
    parser.add_argument("--csv-root", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_STORAGE_OUTPUT)
    parser.add_argument(
        "--accept-dataset-terms",
        required=True,
        help=f"after reviewing the source terms, pass exactly {BACKBLAZE_TERMS}",
    )
    parser.add_argument("--negative-keep-modulus", default=64, type=int)
    parser.add_argument("--minimum-positive-examples", default=25, type=int)
    parser.add_argument("--minimum-auc", default=0.6, type=float)
    arguments = parser.parse_args(argv)
    output = Path(arguments.output_dir).expanduser()
    try:
        dataset = BackblazeDatasetBuilder(
            accept_terms=arguments.accept_dataset_terms,
            negative_keep_modulus=arguments.negative_keep_modulus,
        ).build(Path(arguments.csv_root).expanduser())
        report = StorageTrainer(
            minimum_positive_examples=arguments.minimum_positive_examples,
            minimum_auc=arguments.minimum_auc,
        ).train(dataset, output)
    except (TrainingError, OSError, ValueError) as error:
        print(f"storage training failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "report": str(output / "storage-training-report.json"),
                "portableModel": str(output / "model.onnx"),
                "samples": report["samples"],
                "quality": report["quality"],
                "next": "compile and qualify this source separately for each target lane",
            },
            indent=2,
        )
    )
    return 0


def network_train_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="omnitensor-train-network-model",
        description="Fit an identity-free network anomaly source from normal aggregate replay",
    )
    parser.add_argument("--replay", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_NETWORK_OUTPUT)
    parser.add_argument(
        "--confirm-normal-only",
        required=True,
        help=f"after reviewing the replay, pass exactly {NORMAL_ONLY_CONFIRMATION}",
    )
    parser.add_argument("--minimum-training-rows", default=64, type=int)
    parser.add_argument("--minimum-holdout-rows", default=16, type=int)
    parser.add_argument("--maximum-false-positive-rate", default=0.1, type=float)
    arguments = parser.parse_args(argv)
    output = Path(arguments.output_dir).expanduser()
    try:
        dataset = load_network_replay(Path(arguments.replay).expanduser())
        report = NetworkAnomalyTrainer(
            confirm_normal=arguments.confirm_normal_only,
            minimum_training_rows=arguments.minimum_training_rows,
            minimum_holdout_rows=arguments.minimum_holdout_rows,
            maximum_false_positive_rate=arguments.maximum_false_positive_rate,
        ).train(dataset, output)
    except (TrainingError, OSError, ValueError) as error:
        print(f"network training failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "report": str(output / "network-training-report.json"),
                "portableModel": str(output / "model.onnx"),
                "samples": report["split"],
                "quality": report["quality"],
                "next": "compile and qualify this source separately for each target lane",
            },
            indent=2,
        )
    )
    return 0


def build_train_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="omnitensor-train-build-model",
        description="Fit portable build-risk and optional-check ranking scores",
    )
    parser.add_argument("--history", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_BUILD_OUTPUT)
    parser.add_argument("--mandatory-check", required=True, action="append")
    parser.add_argument("--optional-check", required=True, action="append")
    parser.add_argument(
        "--accept-provenance",
        required=True,
        help=f"after auditing the metadata export, pass exactly {BUILD_PROVENANCE_CONFIRMATION}",
    )
    parser.add_argument("--minimum-class-examples", default=8, type=int)
    parser.add_argument("--minimum-check-class-examples", default=4, type=int)
    parser.add_argument("--minimum-risk-auc", default=0.55, type=float)
    parser.add_argument("--minimum-ranking-mrr", default=0.25, type=float)
    arguments = parser.parse_args(argv)
    output = Path(arguments.output_dir).expanduser()
    try:
        dataset = load_build_history(
            Path(arguments.history).expanduser(),
            accept_provenance=arguments.accept_provenance,
        )
        report = BuildAdvisorTrainer(
            mandatory_checks=arguments.mandatory_check,
            optional_checks=arguments.optional_check,
            minimum_class_examples=arguments.minimum_class_examples,
            minimum_check_class_examples=arguments.minimum_check_class_examples,
            minimum_risk_auc=arguments.minimum_risk_auc,
            minimum_ranking_mrr=arguments.minimum_ranking_mrr,
        ).train(dataset, output)
    except (TrainingError, OSError, ValueError) as error:
        print(f"build training failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "report": str(output / "build-training-report.json"),
                "portableModel": str(output / "model.onnx"),
                "samples": report["split"],
                "quality": report["quality"],
                "next": "compile and qualify this source separately for each target lane",
            },
            indent=2,
        )
    )
    return 0


def hardware_train_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="omnitensor-train-hardware-model",
        description="Fit a portable hardware-health risk source from reviewed labels",
    )
    parser.add_argument("--history", required=True)
    parser.add_argument("--sensor", required=True, action="append", help="ROLE=STABLE_ID")
    parser.add_argument("--output-dir", default=DEFAULT_HARDWARE_OUTPUT)
    parser.add_argument(
        "--confirm-labels",
        required=True,
        help=f"after reviewing baseline/fault labels, pass exactly {HARDWARE_LABEL_CONFIRMATION}",
    )
    parser.add_argument("--window", default=8, type=int)
    parser.add_argument("--minimum-class-examples", default=8, type=int)
    parser.add_argument("--minimum-auc", default=0.6, type=float)
    parser.add_argument("--minimum-recall", default=0.5, type=float)
    parser.add_argument("--maximum-false-positive-rate", default=0.1, type=float)
    arguments = parser.parse_args(argv)
    output = Path(arguments.output_dir).expanduser()
    try:
        bindings = parse_sensor_bindings(arguments.sensor)
        dataset = load_hardware_history(
            Path(arguments.history).expanduser(),
            bindings=bindings,
            confirm_labels=arguments.confirm_labels,
        )
        report = HardwareHealthTrainer(
            window=arguments.window,
            minimum_class_examples=arguments.minimum_class_examples,
            minimum_auc=arguments.minimum_auc,
            minimum_recall=arguments.minimum_recall,
            maximum_false_positive_rate=arguments.maximum_false_positive_rate,
        ).train(dataset, output)
    except (TrainingError, OSError, ValueError) as error:
        print(f"hardware training failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "report": str(output / "hardware-training-report.json"),
                "portableModel": str(output / "model.onnx"),
                "samples": report["split"],
                "quality": report["quality"],
                "next": "compile and qualify this source separately for each target lane",
            },
            indent=2,
        )
    )
    return 0


def desktop_train_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="omnitensor-train-desktop-model",
        description="Fit content-free personalized desktop suggestion scores",
    )
    parser.add_argument("--history", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_DESKTOP_OUTPUT)
    parser.add_argument("--minimum-class-examples", default=8, type=int)
    parser.add_argument("--minimum-accuracy", default=0.6, type=float)
    parser.add_argument("--minimum-macro-recall", default=0.5, type=float)
    arguments = parser.parse_args(argv)
    output = Path(arguments.output_dir).expanduser()
    try:
        dataset = load_desktop_history(Path(arguments.history).expanduser())
        report = DesktopSuggestionTrainer(
            minimum_class_examples=arguments.minimum_class_examples,
            minimum_accuracy=arguments.minimum_accuracy,
            minimum_macro_recall=arguments.minimum_macro_recall,
        ).train(dataset, output)
    except (TrainingError, OSError, ValueError) as error:
        print(f"desktop training failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "report": str(output / "desktop-training-report.json"),
                "portableModel": str(output / "model.onnx"),
                "samples": report["split"],
                "quality": report["quality"],
                "next": "compile and qualify separately; never apply a suggestion automatically",
            },
            indent=2,
        )
    )
    return 0


def desktop_revoke_main(argv: list[str] | None = None) -> int:
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
        removed = revoke_desktop_history(arguments.history, arguments.confirm_delete)
    except (TrainingError, OSError, ValueError) as error:
        print(f"desktop history revocation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"removed": removed}, indent=2))
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
        help=(
            "auto or comma-separated tpu,npu,gpu; each lane needs a compatible source and compiler"
        ),
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
                "compiler-missing",
                "no compatible native compiler found; install the producer tool for a target lane",
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
