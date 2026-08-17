"""Forecast-v1 compiler and installation CLI ownership."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from omnitensor.forecasting import ForecastError
from omnitensor.preparation import ArtifactInstallationError, PreparationError

from ..conversion import ConversionError
from .contracts import TrainingError
from .installation import available_targets, install_training

DEFAULT_ARTIFACT_ROOT = "~/.local/share/omnitensor/artifacts"
DEFAULT_BINDINGS_ROOT = "~/.local/share/omnitensor/model-bindings"


@dataclass(frozen=True, slots=True)
class InstallCliDependencies:
    target_provider: object = available_targets
    installer: object = install_training


def install_main(
    argv: list[str] | None = None,
    *,
    dependencies: InstallCliDependencies | None = None,
    artifact_root_default: str = DEFAULT_ARTIFACT_ROOT,
    bindings_root_default: str = DEFAULT_BINDINGS_ROOT,
    target_parser=None,
) -> int:
    deps = dependencies or InstallCliDependencies()
    parse_targets = target_parser or _targets
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
    parser.add_argument("--artifact-root", default=artifact_root_default)
    parser.add_argument("--bindings-root", default=bindings_root_default)
    parser.add_argument("--build-root", default=None)
    arguments = parser.parse_args(argv)
    try:
        selected = (
            deps.target_provider()
            if arguments.targets == "auto"
            else parse_targets(arguments.targets)
        )
        if not selected:
            raise TrainingError(
                "compiler-missing",
                "no compatible native compiler found; install the producer tool for a target lane",
            )
        installed = deps.installer(
            Path(arguments.report).expanduser(),
            targets=selected,
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


def _targets(text: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in text.split(",") if part.strip())
    if not values:
        raise TrainingError("targets-invalid", "at least one target is required")
    return values


_targets.__module__ = "omnitensor.training.cli"
targets = _targets
