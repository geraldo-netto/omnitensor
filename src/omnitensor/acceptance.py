"""Stable facade for installed-service acceptance checks and probes."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import jsonschema

from .acceptance_checks import (
    BACKEND_RUNTIMES,
    DEFAULT_SNAPSHOT_MAX_AGE_MS,
    FUTURE_TOLERANCE_MS,
    REQUIRED_SCHEMAS,
    check_applet,
    check_applet_contract,
    check_backends,
    check_bus,
    check_confinement,
    check_executable,
    check_isolation,
    check_plugin_discovery,
    check_schemas,
    check_service,
    check_snapshot,
    check_workload_catalog,
    verify_installation,
)
from .acceptance_cli import (
    DEFAULT_APPLET_ROOT,
    build_default_report,
    load_applet_checksums,
    main,
)
from .acceptance_contracts import BusProbe, Check, InstallationReport, ServiceProbe
from .acceptance_probes import (
    DbusApplyCommandProbe,
    SystemdUserServiceProbe,
    _executor_for,
    _run_command,
    _runtime_verdict,
)
from .registry import (
    ManifestError,
    _schema_path,
    bundled_workloads_path,
    load_schema,
    load_workloads,
    validate_document,
)

_PRIVATE_COMPAT = (_schema_path, _executor_for, _runtime_verdict, _run_command)

# The previous implementation had no ``__all__``. Its import-star surface was
# therefore every non-private imported helper plus every public definition;
# state that exact compatibility surface explicitly now that owners are split.
__all__ = [
    "json",
    "shutil",
    "Callable",
    "Iterable",
    "Sequence",
    "dataclass",
    "Path",
    "Protocol",
    "runtime_checkable",
    "jsonschema",
    "ManifestError",
    "bundled_workloads_path",
    "load_schema",
    "load_workloads",
    "validate_document",
    "REQUIRED_SCHEMAS",
    "DEFAULT_SNAPSHOT_MAX_AGE_MS",
    "FUTURE_TOLERANCE_MS",
    "Check",
    "InstallationReport",
    "ServiceProbe",
    "BusProbe",
    "check_executable",
    "check_service",
    "check_schemas",
    "check_workload_catalog",
    "check_plugin_discovery",
    "check_isolation",
    "check_bus",
    "check_snapshot",
    "check_applet",
    "DEFAULT_APPLET_ROOT",
    "check_confinement",
    "check_applet_contract",
    "BACKEND_RUNTIMES",
    "check_backends",
    "verify_installation",
    "SystemdUserServiceProbe",
    "DbusApplyCommandProbe",
    "load_applet_checksums",
    "build_default_report",
    "main",
]
