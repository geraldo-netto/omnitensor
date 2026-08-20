"""Table-driven CLI dispatch for forecast and numeric model trainers."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from omnitensor.forecasting import ForecastError

from .. import paths
from .build import (
    BUILD_PROVENANCE_CONFIRMATION,
    BuildAdvisorTrainer,
    load_build_history,
)
from .contracts import TrainingError, TrainingSpec
from .desktop import DesktopSuggestionTrainer, load_desktop_history
from .forecast import ForecastTrainer
from .hardware import (
    HARDWARE_LABEL_CONFIRMATION,
    HardwareHealthTrainer,
    load_hardware_history,
    parse_sensor_bindings,
)
from .network import NORMAL_ONLY_CONFIRMATION, NetworkAnomalyTrainer, load_network_replay
from .storage import BACKBLAZE_TERMS, BackblazeDatasetBuilder, StorageTrainer

DEFAULT_RECORDS_ROOT = paths.TELEMETRY_RECORDS_ROOT
DEFAULT_OUTPUT_ROOT = paths.TRAINING_OUTPUT_ROOT
DEFAULT_STORAGE_OUTPUT = "~/.local/share/omnitensor/training/storage-intelligence/local"
DEFAULT_NETWORK_OUTPUT = "~/.local/share/omnitensor/training/network-peripherals/local"
DEFAULT_BUILD_OUTPUT = "~/.local/share/omnitensor/training/build-advisor/local"
DEFAULT_HARDWARE_OUTPUT = "~/.local/share/omnitensor/training/hardware-health/local"
DEFAULT_DESKTOP_OUTPUT = "~/.local/share/omnitensor/training/desktop-context/local"


def _features(text: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in text.split(",") if part.strip())
    if not values:
        raise TrainingError("features-invalid", "at least one feature is required")
    return values


_features.__module__ = "omnitensor.training.cli"
features = _features


def _default_outputs() -> dict[str, str]:
    return {
        "forecast": DEFAULT_OUTPUT_ROOT,
        "storage": DEFAULT_STORAGE_OUTPUT,
        "network": DEFAULT_NETWORK_OUTPUT,
        "build": DEFAULT_BUILD_OUTPUT,
        "hardware": DEFAULT_HARDWARE_OUTPUT,
        "desktop": DEFAULT_DESKTOP_OUTPUT,
    }


@dataclass(frozen=True, slots=True)
class TrainingCliDependencies:
    forecast_trainer: object = ForecastTrainer
    storage_builder: object = BackblazeDatasetBuilder
    storage_trainer: object = StorageTrainer
    network_loader: object = load_network_replay
    network_trainer: object = NetworkAnomalyTrainer
    build_loader: object = load_build_history
    build_trainer: object = BuildAdvisorTrainer
    sensor_parser: object = parse_sensor_bindings
    hardware_loader: object = load_hardware_history
    hardware_trainer: object = HardwareHealthTrainer
    desktop_loader: object = load_desktop_history
    desktop_trainer: object = DesktopSuggestionTrainer
    feature_parser: object = _features
    outputs: Mapping[str, str] = field(default_factory=_default_outputs)
    records_root: str = DEFAULT_RECORDS_ROOT


def _forecast_train_main(argv, deps: TrainingCliDependencies) -> int:
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
    parser.add_argument("--records-root", default=deps.records_root)
    parser.add_argument("--output-dir", default=None)
    arguments = parser.parse_args(argv)
    output = (
        Path(arguments.output_dir).expanduser()
        if arguments.output_dir
        else Path(deps.outputs["forecast"]).expanduser() / arguments.profile / arguments.version
    )
    try:
        spec = TrainingSpec(
            arguments.profile,
            arguments.artifact_id,
            arguments.version,
            deps.feature_parser(arguments.features),
            arguments.target_feature,
            arguments.window,
            arguments.horizon,
        )
        report = deps.forecast_trainer().train(
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


def _storage_train_main(argv, deps: TrainingCliDependencies) -> int:
    parser = argparse.ArgumentParser(
        prog="omnitensor-train-storage-model",
        description="Fit a portable storage risk source from official Backblaze daily CSVs",
    )
    parser.add_argument("--csv-root", required=True)
    parser.add_argument("--output-dir", default=deps.outputs["storage"])
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
        dataset = deps.storage_builder(
            accept_terms=arguments.accept_dataset_terms,
            negative_keep_modulus=arguments.negative_keep_modulus,
        ).build(Path(arguments.csv_root).expanduser())
        report = deps.storage_trainer(
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


def _network_train_main(argv, deps: TrainingCliDependencies) -> int:
    parser = argparse.ArgumentParser(
        prog="omnitensor-train-network-model",
        description="Fit an identity-free network anomaly source from normal aggregate replay",
    )
    parser.add_argument("--replay", required=True)
    parser.add_argument("--output-dir", default=deps.outputs["network"])
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
        dataset = deps.network_loader(Path(arguments.replay).expanduser())
        report = deps.network_trainer(
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


def _build_train_main(argv, deps: TrainingCliDependencies) -> int:
    parser = argparse.ArgumentParser(
        prog="omnitensor-train-build-model",
        description="Fit portable build-risk and optional-check ranking scores",
    )
    parser.add_argument("--history", required=True)
    parser.add_argument("--output-dir", default=deps.outputs["build"])
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
        dataset = deps.build_loader(
            Path(arguments.history).expanduser(),
            accept_provenance=arguments.accept_provenance,
        )
        report = deps.build_trainer(
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


def _hardware_train_main(argv, deps: TrainingCliDependencies) -> int:
    parser = argparse.ArgumentParser(
        prog="omnitensor-train-hardware-model",
        description="Fit a portable hardware-health risk source from reviewed labels",
    )
    parser.add_argument("--history", required=True)
    parser.add_argument("--sensor", required=True, action="append", help="ROLE=STABLE_ID")
    parser.add_argument("--output-dir", default=deps.outputs["hardware"])
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
        bindings = deps.sensor_parser(arguments.sensor)
        dataset = deps.hardware_loader(
            Path(arguments.history).expanduser(),
            bindings=bindings,
            confirm_labels=arguments.confirm_labels,
        )
        report = deps.hardware_trainer(
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


def _desktop_train_main(argv, deps: TrainingCliDependencies) -> int:
    parser = argparse.ArgumentParser(
        prog="omnitensor-train-desktop-model",
        description="Fit content-free personalized desktop suggestion scores",
    )
    parser.add_argument("--history", required=True)
    parser.add_argument("--output-dir", default=deps.outputs["desktop"])
    parser.add_argument("--minimum-class-examples", default=8, type=int)
    parser.add_argument("--minimum-accuracy", default=0.6, type=float)
    parser.add_argument("--minimum-macro-recall", default=0.5, type=float)
    arguments = parser.parse_args(argv)
    output = Path(arguments.output_dir).expanduser()
    try:
        dataset = deps.desktop_loader(Path(arguments.history).expanduser())
        report = deps.desktop_trainer(
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


TRAINER_DISPATCH: Mapping[str, Callable] = MappingProxyType(
    {
        "forecast": _forecast_train_main,
        "storage": _storage_train_main,
        "network": _network_train_main,
        "build": _build_train_main,
        "hardware": _hardware_train_main,
        "desktop": _desktop_train_main,
    }
)


def dispatch_trainer(
    command: str,
    argv: list[str] | None = None,
    *,
    dependencies: TrainingCliDependencies | None = None,
) -> int:
    try:
        handler = TRAINER_DISPATCH[command]
    except KeyError as error:
        raise TrainingError("trainer-unknown", f"no trainer command is named {command}") from error
    return handler(argv, dependencies or TrainingCliDependencies())


def train_main(argv=None, *, dependencies=None) -> int:
    return dispatch_trainer("forecast", argv, dependencies=dependencies)


def storage_train_main(argv=None, *, dependencies=None) -> int:
    return dispatch_trainer("storage", argv, dependencies=dependencies)


def network_train_main(argv=None, *, dependencies=None) -> int:
    return dispatch_trainer("network", argv, dependencies=dependencies)


def build_train_main(argv=None, *, dependencies=None) -> int:
    return dispatch_trainer("build", argv, dependencies=dependencies)


def hardware_train_main(argv=None, *, dependencies=None) -> int:
    return dispatch_trainer("hardware", argv, dependencies=dependencies)


def desktop_train_main(argv=None, *, dependencies=None) -> int:
    return dispatch_trainer("desktop", argv, dependencies=dependencies)
