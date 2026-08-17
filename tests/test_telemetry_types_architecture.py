from __future__ import annotations

import ast
import pickle
import subprocess
import sys
from pathlib import Path

import omnitensor.forecasting as forecasting
import omnitensor.telemetry_recorder as recorder
import omnitensor.telemetry_types as types
from omnitensor.plugins import build_ingestion, hardware_collection, network_collection, triggers
from omnitensor.plugins import forecasting as legacy_forecasting
from omnitensor.plugins import recorder as legacy_recorder

ROOT = Path(__file__).parents[1]


def test_neutral_telemetry_owners_load_without_the_plugin_package():
    script = """
import sys
import omnitensor.forecasting
import omnitensor.telemetry_recorder
import omnitensor.telemetry_types
loaded = [
    name for name in sys.modules
    if name == 'omnitensor.plugins' or name.startswith('omnitensor.plugins.')
]
assert loaded == []
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_training_has_no_direct_plugin_dependency():
    training_root = ROOT / "src/omnitensor/training"
    violations = []
    for path in sorted(training_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                plugin_import = any(
                    alias.name == "omnitensor.plugins"
                    or alias.name.startswith("omnitensor.plugins.")
                    for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                plugin_import = (
                    module == "plugins"
                    or module.startswith("plugins.")
                    or module.startswith("omnitensor.plugins")
                )
                plugin_import = plugin_import or (
                    module in {"", "omnitensor"}
                    and any(alias.name == "plugins" for alias in node.names)
                )
            else:
                plugin_import = False
            if plugin_import:
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert violations == []


def test_neutral_telemetry_types_are_exact_legacy_reexports():
    pairs = (
        (types.SourceStatus, triggers.SourceStatus),
        (types.BuildOutcome, build_ingestion.BuildOutcome),
        (types.BuildRecord, build_ingestion.BuildRecord),
        (types.SensorKind, hardware_collection.SensorKind),
        (types.SensorHealth, hardware_collection.SensorHealth),
        (types.HardwareSample, hardware_collection.HardwareSample),
        (types.NetworkLinkKind, network_collection.NetworkLinkKind),
        (types.NetworkLinkState, network_collection.NetworkLinkState),
        (types.NetworkConnectivity, network_collection.NetworkConnectivity),
        (types.NetworkCounters, network_collection.NetworkCounters),
        (types.NetworkLinkSample, network_collection.NetworkLinkSample),
        (types.NetworkSnapshot, network_collection.NetworkSnapshot),
        (types.FeatureRow, legacy_recorder.FeatureRow),
        (types.RecorderError, legacy_recorder.RecorderError),
        (types.ForecastError, legacy_forecasting.ForecastError),
        (types.ForecastQuality, legacy_forecasting.ForecastQuality),
        (types.LinearForecaster, legacy_forecasting.LinearForecaster),
    )
    assert all(neutral is legacy for neutral, legacy in pairs)


def test_neutral_implementations_preserve_legacy_facade_identity():
    assert recorder.TelemetryRecorder is legacy_recorder.TelemetryRecorder
    assert recorder.validated_features is legacy_recorder.validated_features
    assert recorder.feature_matrix is legacy_recorder.feature_matrix
    assert forecasting.fit_forecaster is legacy_forecasting.fit_forecaster
    assert (
        forecasting.evaluate_forecast_predictions
        is legacy_forecasting.evaluate_forecast_predictions
    )


def test_moved_values_pickle_through_the_module_that_defines_them():
    values = (
        types.SourceStatus.READY,
        types.BuildRecord("build", types.BuildOutcome.SUCCEEDED, 1, 2, (), ()),
        types.HardwareSample(
            "sensor", types.SensorKind.THERMAL, types.SensorHealth.OK, 1, "mC", "CPU", 2
        ),
        types.NetworkSnapshot(types.SourceStatus.READY, 2, ()),
        types.FeatureRow("profile", 2, {"value": 1.0}),
        types.ForecastQuality(2, 0.1, 0.2),
    )
    for value in values:
        assert pickle.loads(pickle.dumps(value)) == value

    # Every one of these said it lived in a `plugins.*` module that merely
    # re-exports it. Nothing read that claim except a traceback, which it
    # misdirected, and a pickle, which named a file the definition is not in.
    for value in (
        types.SourceStatus,
        types.BuildRecord,
        types.HardwareSample,
        types.NetworkSnapshot,
        types.FeatureRow,
        types.ForecastQuality,
        types.build_record_error,
        types.hardware_sample_error,
        types.network_snapshot_error,
    ):
        assert value.__module__ == "omnitensor.telemetry_types"
