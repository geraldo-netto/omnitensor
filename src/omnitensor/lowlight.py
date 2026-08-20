"""Low-light workspace configuration and non-destructive output publication.

This module deliberately does not decode images or run a model. It owns the
filesystem safety boundary that those separate stages use: inputs and
outputs live in disjoint, explicitly configured roots, and publishing creates
a new output without ever replacing either an input or an earlier result.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

from .atomicio import fsync_directory as _fsync_directory
from .atomicio import write_bytes_atomic as _write_bytes_atomic
from .lowlight_settings import LOW_LIGHT_CONFIGURATION_SPEC as LOW_LIGHT_CONFIGURATION_SPEC
from .lowlight_settings import (
    LOW_LIGHT_CONFIGURATION_VERSION as LOW_LIGHT_CONFIGURATION_VERSION,
)
from .lowlight_settings import LOW_LIGHT_WORKLOAD_ID as LOW_LIGHT_WORKLOAD_ID
from .lowlight_settings import MAX_PATH_CHARACTERS as MAX_PATH_CHARACTERS
from .lowlight_settings import (
    _migrate_low_light_0_1_to_0_2 as _migrate_low_light_0_1_to_0_2,
)
from .plugins.settings import PluginSettingsError, PluginSettingsStore
from .stable_error import StableError

DEFAULT_LOW_LIGHT_SETTINGS_ROOT = "~/.config/omnitensor/workload-settings"
DEFAULT_MAX_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_OUTPUT_NAME_CHARACTERS = 255


class LowLightWorkspaceError(StableError, ValueError):
    """Stable refusal at the low-light filesystem boundary."""


@dataclass(frozen=True, slots=True)
class LowLightWorkspace:
    """Canonical, disjoint roots selected by the user."""

    input_folder: Path
    output_folder: Path

    def document(self) -> dict[str, str]:
        return {
            "inputFolder": str(self.input_folder),
            "outputFolder": str(self.output_folder),
        }


@dataclass(frozen=True, slots=True)
class ConfiguredLowLightWorkspace:
    """A validated workspace and the persisted revision that supplied it."""

    workspace: LowLightWorkspace
    revision: int


def validate_low_light_workspace(
    input_folder: Path | str,
    output_folder: Path | str,
) -> LowLightWorkspace:
    """Resolve and validate distinct roots before any image is processed."""
    source = _canonical_directory(input_folder, "input", os.R_OK | os.X_OK)
    destination = _canonical_directory(output_folder, "output", os.W_OK | os.X_OK)
    if source == destination or source in destination.parents or destination in source.parents:
        raise LowLightWorkspaceError(
            "folders-overlap",
            "input and output folders must be disjoint; neither may contain the other",
        )
    return LowLightWorkspace(source, destination)


def configure_low_light_workspace(
    store: PluginSettingsStore,
    input_folder: Path | str,
    output_folder: Path | str,
) -> ConfiguredLowLightWorkspace:
    """Validate and compare-and-swap one canonical workspace configuration."""
    workspace = validate_low_light_workspace(input_folder, output_folder)
    current = store.load(LOW_LIGHT_CONFIGURATION_SPEC)
    updated = store.update(
        LOW_LIGHT_CONFIGURATION_SPEC,
        expected_revision=current.revision,
        configuration=workspace.document(),
    )
    return ConfiguredLowLightWorkspace(workspace, updated.revision)


def load_low_light_workspace(
    store: PluginSettingsStore,
) -> ConfiguredLowLightWorkspace | None:
    """Load and semantically revalidate persisted roots, or report unconfigured."""
    settings = store.load(LOW_LIGHT_CONFIGURATION_SPEC)
    source = settings.configuration["inputFolder"]
    destination = settings.configuration["outputFolder"]
    if source is None and destination is None:
        return None
    if not isinstance(source, str) or not isinstance(destination, str):
        raise LowLightWorkspaceError(
            "configuration-incomplete",
            "input and output folders must both be configured",
        )
    workspace = validate_low_light_workspace(source, destination)
    if workspace.document() != dict(settings.configuration):
        raise LowLightWorkspaceError(
            "configuration-not-canonical",
            "stored input and output folders must be canonical absolute paths",
        )
    return ConfiguredLowLightWorkspace(workspace, settings.revision)


def publish_low_light_output(
    workspace: LowLightWorkspace,
    source: Path | str,
    encoded: bytes,
    *,
    output_name: str | None = None,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> Path:
    """Publish bytes atomically without modifying the source or replacing output.

    The final path is created with a hard link from a synced, same-directory
    staging file. ``link(2)`` fails when the destination exists, making the
    no-overwrite rule atomic even when two workers publish concurrently.
    """
    if (
        isinstance(max_output_bytes, bool)
        or not isinstance(max_output_bytes, int)
        or max_output_bytes < 1
    ):
        raise ValueError("max_output_bytes must be a positive integer")
    if not isinstance(encoded, bytes) or not encoded:
        raise LowLightWorkspaceError("output-invalid", "encoded output must be non-empty bytes")
    if len(encoded) > max_output_bytes:
        raise LowLightWorkspaceError(
            "output-too-large",
            f"encoded output exceeds the {max_output_bytes} byte limit",
        )

    current = validate_low_light_workspace(workspace.input_folder, workspace.output_folder)
    source_path = _source_file(current.input_folder, source)
    name = _output_name(output_name if output_name is not None else source_path.name)
    destination = current.output_folder / name
    try:
        _write_bytes_atomic(
            destination,
            encoded,
            0o600,
            replace=False,
            prefix=".low-light-output-",
        )
        _fsync_directory(current.output_folder)
    except FileExistsError as error:
        raise LowLightWorkspaceError(
            "output-exists",
            f"refusing to overwrite existing output: {destination}",
        ) from error
    except OSError as error:
        raise LowLightWorkspaceError(
            "output-unavailable",
            f"cannot publish output in {current.output_folder}",
        ) from error
    return destination


def resolve_low_light_source(workspace: LowLightWorkspace, source: Path | str) -> Path:
    """Resolve one regular input under a freshly revalidated workspace."""
    current = validate_low_light_workspace(workspace.input_folder, workspace.output_folder)
    return _source_file(current.input_folder, source)


def configure_main(argv: list[str] | None = None) -> int:
    """Configure the low-light workspace without starting image processing."""
    parser = argparse.ArgumentParser(
        prog="omnitensor-configure-low-light",
        description="Configure disjoint input and output folders for low-light enhancement",
    )
    parser.add_argument("--input-folder", required=True)
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--settings-root", default=DEFAULT_LOW_LIGHT_SETTINGS_ROOT)
    arguments = parser.parse_args(argv)
    try:
        configured = configure_low_light_workspace(
            PluginSettingsStore(Path(arguments.settings_root).expanduser()),
            Path(arguments.input_folder).expanduser(),
            Path(arguments.output_folder).expanduser(),
        )
    except (LowLightWorkspaceError, PluginSettingsError, OSError, ValueError) as error:
        print(f"low-light configuration failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "workloadId": LOW_LIGHT_WORKLOAD_ID,
                "revision": configured.revision,
                **configured.workspace.document(),
                "originalsPreserved": True,
                "existingOutputsOverwritten": False,
            },
            indent=2,
        )
    )
    return 0


def _canonical_directory(path: Path | str, label: str, access: int) -> Path:
    if not isinstance(path, (str, os.PathLike)):
        raise LowLightWorkspaceError(f"{label}-folder-invalid", f"{label} folder is invalid")
    candidate = Path(path).expanduser()
    if len(str(candidate)) > MAX_PATH_CHARACTERS:
        raise LowLightWorkspaceError(
            f"{label}-folder-invalid",
            f"{label} folder path exceeds {MAX_PATH_CHARACTERS} characters",
        )
    try:
        resolved = candidate.resolve(strict=True)
        mode = resolved.stat().st_mode
    except (OSError, RuntimeError, ValueError) as error:
        raise LowLightWorkspaceError(
            f"{label}-folder-invalid",
            f"{label} folder does not resolve to an existing directory",
        ) from error
    if not stat.S_ISDIR(mode):
        raise LowLightWorkspaceError(
            f"{label}-folder-invalid",
            f"{label} folder is not a directory",
        )
    if not os.access(resolved, access):
        required = "readable and searchable" if label == "input" else "writable and searchable"
        raise LowLightWorkspaceError(
            f"{label}-folder-unavailable",
            f"{label} folder must be {required}",
        )
    return resolved


def _source_file(input_folder: Path, source: Path | str) -> Path:
    if not isinstance(source, (str, os.PathLike)):
        raise LowLightWorkspaceError("input-invalid", "input path is invalid")
    try:
        resolved = Path(source).resolve(strict=True)
        mode = resolved.stat().st_mode
    except (OSError, RuntimeError, ValueError) as error:
        raise LowLightWorkspaceError(
            "input-invalid",
            "input does not resolve to an existing regular file",
        ) from error
    if input_folder not in resolved.parents or not stat.S_ISREG(mode):
        raise LowLightWorkspaceError(
            "input-invalid",
            "input must be a regular file contained by the configured input folder",
        )
    return resolved


def _output_name(name: str) -> str:
    if (
        not isinstance(name, str)
        or not name
        or len(name) > MAX_OUTPUT_NAME_CHARACTERS
        or name in {".", ".."}
        or Path(name).name != name
        or "\x00" in name
    ):
        raise LowLightWorkspaceError(
            "output-name-invalid",
            "output name must be one non-empty filename",
        )
    return name


if __name__ == "__main__":  # pragma: no cover - console entry point owns this path
    raise SystemExit(configure_main())
