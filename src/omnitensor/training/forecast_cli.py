"""Trusted live-forecast command-line orchestration."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from omnitensor.telemetry_recorder import TelemetryRecorder

from .. import paths
from ..forecastresult import parse_forecast_reading
from .forecast_client import SocketForecastClient
from .forecast_contracts import ForecastRunError
from .runner import TrustedForecastRunner, load_forecast_binding

DEFAULT_RECORDS_ROOT = paths.TELEMETRY_RECORDS_ROOT
DEFAULT_BINDINGS_ROOT = paths.MODEL_BINDINGS_ROOT


@dataclass(frozen=True, slots=True)
class ForecastCliDependencies:
    binding_loader: object = load_forecast_binding
    client_type: object = SocketForecastClient
    runner_type: object = TrustedForecastRunner
    recorder_type: object = TelemetryRecorder
    reading_parser: object = parse_forecast_reading


def forecast_main(
    argv: list[str] | None = None,
    *,
    dependencies: ForecastCliDependencies | None = None,
    records_root_default: str = DEFAULT_RECORDS_ROOT,
    bindings_root_default: str = DEFAULT_BINDINGS_ROOT,
) -> int:
    deps = dependencies or ForecastCliDependencies()
    parser = argparse.ArgumentParser(
        prog="omnitensor-run-forecast",
        description="Submit one forecast assembled only from trusted local history",
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--records-root", default=records_root_default)
    parser.add_argument("--bindings-root", default=bindings_root_default)
    parser.add_argument("--attempts", default=40, type=int)
    parser.add_argument("--poll-interval", default=0.1, type=float)
    arguments = parser.parse_args(argv)

    async def execute() -> dict:
        workload = deps.binding_loader(
            arguments.profile, Path(arguments.bindings_root).expanduser()
        )
        client = await deps.client_type.connect()
        try:
            return await deps.runner_type(
                workload,
                deps.recorder_type(Path(arguments.records_root).expanduser()),
                client,
                attempts=arguments.attempts,
                poll_interval=arguments.poll_interval,
            ).run()
        finally:
            client.close()

    try:
        output = asyncio.run(execute())
        reading = deps.reading_parser(output.get("reading"))
        if reading is None:
            raise ForecastRunError(
                "runtime-response-invalid", "runtime returned no valid forecast reading"
            )
    except (ForecastRunError, OSError, ValueError) as error:
        print(f"forecast failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(reading, indent=2, allow_nan=False))
    return 0
