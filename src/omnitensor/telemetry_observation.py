"""Job, inventory, summary, and snapshot telemetry observation."""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence

from .forecastresult import parse_forecast_reading
from .inspection import build_plugin_inventory
from .plugins.secrets import SecretRedactor
from .plugins.summaries import ResultSummaryRegistry, SummaryError
from .plugins.telemetry import PluginTelemetryRegistry
from .ports import (
    PluginCatalogSnapshot,
    PluginPermissionSource,
    PluginRuntimeSnapshot,
    PluginSnapshotSource,
)
from .profile_selection import profile_statuses
from .runtime_api import no_inventory
from .snapshot import build_snapshot, input_roots_document

LOGGER = logging.getLogger("omnitensor.service")


class TelemetryJobObserver:
    """Adapter from the job lifecycle onto per-profile telemetry counters."""

    def __init__(self, telemetry: PluginTelemetryRegistry, clock_ms=None) -> None:
        self._telemetry = telemetry
        self._clock_ms = clock_ms or (lambda: max(1, int(time.time() * 1000)))

    def job_started(self, workload_id: str) -> None:
        self._guarded(workload_id, self._telemetry.queue_job)
        self._guarded(workload_id, self._telemetry.start_job)

    def job_finished(self, workload_id: str, status: str, detail: str) -> None:
        if status == "succeeded":
            self._guarded(
                workload_id,
                lambda plugin_id: self._telemetry.succeed(
                    plugin_id, completed_at_ms=self._clock_ms()
                ),
            )
        elif status == "cancelled":
            self._guarded(workload_id, self._telemetry.cancel)
        else:
            self._guarded(
                workload_id,
                lambda plugin_id: self._telemetry.fail(
                    plugin_id,
                    "job-failed",
                    detail or "the job failed",
                    completed_at_ms=self._clock_ms(),
                ),
            )

    def _guarded(self, workload_id: str, call) -> None:
        try:
            call(workload_id)
        except (KeyError, ValueError) as error:
            LOGGER.debug("plugin telemetry not updated for %s: %s", workload_id, error)


TelemetryJobObserver.__module__ = "omnitensor.service"


class ResultSummaryObserver:
    """Convert readable results into bounded desktop summaries."""

    def __init__(
        self,
        summaries: Callable[[], ResultSummaryRegistry],
        *,
        clock_ms: Callable[[], int] | None = None,
        logger: logging.Logger = LOGGER,
    ) -> None:
        self._summaries = summaries
        self._clock_ms = clock_ms or (lambda: max(1, int(time.time() * 1000)))
        self._logger = logger

    def summarize(self, job_id: str, output: dict, *, publish_forecast=None) -> None:
        reading = output.get("reading") if isinstance(output, dict) else None
        profile_id = output.get("profileId") if isinstance(output, dict) else None
        if not isinstance(reading, dict) or not isinstance(profile_id, str):
            return
        forecast = parse_forecast_reading(reading)
        if forecast is not None:
            (publish_forecast or self.publish_forecast)(job_id, profile_id, forecast)
            return
        top = reading.get("top") or []
        if not top:
            return
        best = top[0]
        named = best.get("label") or f"class {best.get('index')}"
        score = best.get("score")
        try:
            self._summaries().publish(
                plugin_id=profile_id,
                title=str(named)[:120],
                summary=(
                    f"{len(top)} candidates, best {score:.3f}"
                    if isinstance(score, (int, float))
                    else f"{len(top)} candidates"
                ),
                timestamp_ms=self._clock_ms(),
                confidence=None,
                risk_score=None,
                result_reference=f"result-{job_id}",
                redactor=SecretRedactor(()),
            )
        except (SummaryError, TypeError, ValueError) as error:
            self._logger.debug("result summary not published for %s: %s", job_id, error)

    def publish_forecast(self, job_id: str, profile_id: str, reading: dict) -> None:
        try:
            self._summaries().publish(
                plugin_id=profile_id,
                title=f"{reading['targetFeature']} forecast",
                summary=f"{reading['horizon']} observations ahead: {reading['value']:g}",
                timestamp_ms=self._clock_ms(),
                confidence=None,
                risk_score=None,
                result_reference=f"result-{job_id}",
                redactor=SecretRedactor(()),
            )
        except (SummaryError, TypeError, ValueError) as error:
            self._logger.debug("forecast summary not published for %s: %s", job_id, error)


def describe_plugins(
    plugin_runtime,
    resolve_artifact,
    *,
    clock_ms: Callable[[], int],
) -> str:
    if not isinstance(plugin_runtime, PluginSnapshotSource):
        return no_inventory()
    retained = plugin_runtime.snapshot
    if not isinstance(retained, PluginRuntimeSnapshot) or not isinstance(
        retained.catalog, PluginCatalogSnapshot
    ):
        return no_inventory()
    states = {status.plugin_id: str(status.state) for status in retained.workers}
    # Why a worker is not running. Dropping it made every failed launch read
    # on the desktop as an unqualified provider, which sent people to install
    # artifacts that were already installed and correct.
    details = {
        status.plugin_id: getattr(status, "detail", "")
        for status in retained.workers
        if getattr(status, "detail", "")
    }
    granted_permissions = (
        plugin_runtime.granted_permissions
        if isinstance(plugin_runtime, PluginPermissionSource)
        else lambda _plugin_id: ()
    )
    document = build_plugin_inventory(
        retained.catalog.plugins,
        resolve_artifact=resolve_artifact,
        granted_permissions=granted_permissions,
        worker_states=states.get,
        worker_details=details.get,
        generated_at_ms=clock_ms(),
    )
    return json.dumps(document, separators=(",", ":"))


def device_load(discovery, device, stats) -> float | None:
    utilization = discovery.utilization(device)
    return (
        utilization
        if utilization is not None
        else stats["loads"].get(device.id, stats["loads"].get(device.backend))
    )


def runtime_snapshot(
    *,
    devices: Sequence[object],
    device_load_of,
    workloads: Mapping,
    executors: dict,
    scheduler,
    policy,
    artifact_ready,
    permissions_missing,
    result_summaries,
    plugin_telemetry,
    input_roots,
    kernel_telemetry_source,
    profile_statuses_of=None,
) -> dict:
    scheduler.tick()
    stats = scheduler.stats()
    observed_devices = [
        dataclasses.replace(device, load=device_load_of(device, stats)) for device in devices
    ]
    return build_snapshot(
        devices=observed_devices,
        metrics=stats,
        profiles=(profile_statuses_of or profile_statuses)(
            workloads,
            executors,
            scheduler,
            policy,
            artifact_ready,
            permissions_missing,
        ),
        alerts=result_summaries.documents(),
        plugin_telemetry=plugin_telemetry.documents(),
        inputs=input_roots_document(input_roots),
        kernel_telemetry=kernel_telemetry_source.read().document(),
    )


def register_plugin_telemetry(snapshot, telemetry: PluginTelemetryRegistry) -> None:
    if not isinstance(snapshot, PluginRuntimeSnapshot) or not isinstance(
        snapshot.catalog, PluginCatalogSnapshot
    ):
        return
    for plugin in snapshot.catalog.plugins:
        telemetry.register(plugin.plugin_id)


__all__ = [
    "ResultSummaryObserver",
    "TelemetryJobObserver",
    "describe_plugins",
    "device_load",
    "register_plugin_telemetry",
    "runtime_snapshot",
]
