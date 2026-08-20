"""Stable facade for offline training command-line entry points."""

# ruff: noqa: F401 - compatibility facade deliberately re-exports old names

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

from .. import paths
from ..conversion import ConversionError
from ..forecastresult import parse_forecast_reading
from .build import (
    BUILD_PROVENANCE_CONFIRMATION,
    BuildAdvisorTrainer,
    load_build_history,
)
from .contracts import TrainingError, TrainingSpec
from .desktop import DesktopSuggestionTrainer, load_desktop_history
from .desktop_history import DESKTOP_REVOCATION_CONFIRMATION, revoke_desktop_history
from .forecast import ForecastTrainer
from .forecast_cli import ForecastCliDependencies
from .forecast_cli import forecast_main as _forecast_main
from .hardware import (
    HARDWARE_LABEL_CONFIRMATION,
    HardwareHealthTrainer,
    load_hardware_history,
    parse_sensor_bindings,
)
from .installation import available_targets, install_training
from .network import NORMAL_ONLY_CONFIRMATION, NetworkAnomalyTrainer, load_network_replay
from .runner import (
    ForecastRunError,
    SocketForecastClient,
    TrustedForecastRunner,
    load_forecast_binding,
)
from .snapshot_recording import load_runtime_snapshot, record_runtime_snapshot
from .storage import BACKBLAZE_TERMS, BackblazeDatasetBuilder, StorageTrainer
from .training_install_cli import InstallCliDependencies
from .training_install_cli import install_main as _install_main
from .training_install_cli import targets as _targets
from .training_record_cli import RecordCliDependencies
from .training_record_cli import desktop_revoke_main as _desktop_revoke_main
from .training_record_cli import feature_values as _feature_values
from .training_record_cli import record_main as _record_main
from .training_record_cli import snapshot_record_main as _snapshot_record_main
from .training_train_cli import TRAINER_DISPATCH, TrainingCliDependencies
from .training_train_cli import build_train_main as _build_train_main
from .training_train_cli import desktop_train_main as _desktop_train_main
from .training_train_cli import features as _features
from .training_train_cli import hardware_train_main as _hardware_train_main
from .training_train_cli import network_train_main as _network_train_main
from .training_train_cli import storage_train_main as _storage_train_main
from .training_train_cli import train_main as _train_main

DEFAULT_RECORDS_ROOT = paths.TELEMETRY_RECORDS_ROOT
DEFAULT_SNAPSHOT_PATH = paths.SNAPSHOT_PATH
DEFAULT_OUTPUT_ROOT = paths.TRAINING_OUTPUT_ROOT
DEFAULT_ARTIFACT_ROOT = paths.ARTIFACT_ROOT
DEFAULT_BINDINGS_ROOT = paths.MODEL_BINDINGS_ROOT
DEFAULT_STORAGE_OUTPUT = f"{paths.TRAINING_OUTPUT_ROOT}/storage-intelligence/local"
DEFAULT_NETWORK_OUTPUT = f"{paths.TRAINING_OUTPUT_ROOT}/network-peripherals/local"
DEFAULT_BUILD_OUTPUT = f"{paths.TRAINING_OUTPUT_ROOT}/build-advisor/local"
DEFAULT_HARDWARE_OUTPUT = "~/.local/share/omnitensor/training/hardware-health/local"
DEFAULT_DESKTOP_OUTPUT = "~/.local/share/omnitensor/training/desktop-context/local"


def _record_dependencies() -> RecordCliDependencies:
    return RecordCliDependencies(
        recorder_type=TelemetryRecorder,
        snapshot_loader=load_runtime_snapshot,
        snapshot_recorder=record_runtime_snapshot,
        desktop_revoker=revoke_desktop_history,
    )


def _training_dependencies() -> TrainingCliDependencies:
    return TrainingCliDependencies(
        forecast_trainer=ForecastTrainer,
        storage_builder=BackblazeDatasetBuilder,
        storage_trainer=StorageTrainer,
        network_loader=load_network_replay,
        network_trainer=NetworkAnomalyTrainer,
        build_loader=load_build_history,
        build_trainer=BuildAdvisorTrainer,
        sensor_parser=parse_sensor_bindings,
        hardware_loader=load_hardware_history,
        hardware_trainer=HardwareHealthTrainer,
        desktop_loader=load_desktop_history,
        desktop_trainer=DesktopSuggestionTrainer,
        feature_parser=_features,
        outputs={
            "forecast": DEFAULT_OUTPUT_ROOT,
            "storage": DEFAULT_STORAGE_OUTPUT,
            "network": DEFAULT_NETWORK_OUTPUT,
            "build": DEFAULT_BUILD_OUTPUT,
            "hardware": DEFAULT_HARDWARE_OUTPUT,
            "desktop": DEFAULT_DESKTOP_OUTPUT,
        },
        records_root=DEFAULT_RECORDS_ROOT,
    )


def record_main(argv: list[str] | None = None) -> int:
    return _record_main(
        argv,
        dependencies=_record_dependencies(),
        records_root_default=DEFAULT_RECORDS_ROOT,
        feature_parser=_feature_values,
    )


def snapshot_record_main(argv: list[str] | None = None) -> int:
    return _snapshot_record_main(
        argv,
        dependencies=_record_dependencies(),
        records_root_default=DEFAULT_RECORDS_ROOT,
        snapshot_default=os.environ.get("OMNITENSOR_STATE_PATH", DEFAULT_SNAPSHOT_PATH),
    )


def train_main(argv: list[str] | None = None) -> int:
    return _train_main(argv, dependencies=_training_dependencies())


def storage_train_main(argv: list[str] | None = None) -> int:
    return _storage_train_main(argv, dependencies=_training_dependencies())


def network_train_main(argv: list[str] | None = None) -> int:
    return _network_train_main(argv, dependencies=_training_dependencies())


def build_train_main(argv: list[str] | None = None) -> int:
    return _build_train_main(argv, dependencies=_training_dependencies())


def hardware_train_main(argv: list[str] | None = None) -> int:
    return _hardware_train_main(argv, dependencies=_training_dependencies())


def desktop_train_main(argv: list[str] | None = None) -> int:
    return _desktop_train_main(argv, dependencies=_training_dependencies())


def desktop_revoke_main(argv: list[str] | None = None) -> int:
    return _desktop_revoke_main(argv, dependencies=_record_dependencies())


def install_main(argv: list[str] | None = None) -> int:
    return _install_main(
        argv,
        dependencies=InstallCliDependencies(
            target_provider=available_targets,
            installer=install_training,
        ),
        artifact_root_default=DEFAULT_ARTIFACT_ROOT,
        bindings_root_default=DEFAULT_BINDINGS_ROOT,
        target_parser=_targets,
    )


def forecast_main(argv: list[str] | None = None) -> int:
    return _forecast_main(
        argv,
        dependencies=ForecastCliDependencies(
            binding_loader=load_forecast_binding,
            client_type=SocketForecastClient,
            runner_type=TrustedForecastRunner,
            recorder_type=TelemetryRecorder,
            reading_parser=parse_forecast_reading,
        ),
        records_root_default=DEFAULT_RECORDS_ROOT,
        bindings_root_default=DEFAULT_BINDINGS_ROOT,
    )
