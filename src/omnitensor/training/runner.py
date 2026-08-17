"""Turn trusted local history into one bounded forecast job.

The public runner deliberately accepts a recorder, not a tensor.  A tensor of
the right shape does not prove which measurements it contains or their order;
the installed feature contract and retained recorder rows do.
"""

from __future__ import annotations

import asyncio
import json
import math
import uuid
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Protocol  # noqa: F401

from omnitensor.telemetry_recorder import TelemetryRecorder
from omnitensor.telemetry_types import FeatureRow

from ..contract import schema_versions
from ..registry import (
    ManifestError,
    Workload,
    apply_model_bindings,
    bundled_workloads_path,
    load_workloads,
    validate_document,
)
from .forecast_client import SocketForecastClient  # noqa: F401
from .forecast_contracts import ForecastClient, ForecastRunError

MAX_WIRE_BYTES = 1024 * 1024
REQUIRED_METHODS = frozenset({"describe-contract", "submit-job", "get-job-result"})
REQUIRED_SCHEMAS = (
    "runtime-job-submit",
    "runtime-job-acknowledgement",
    "runtime-job-result-request",
    "runtime-job-result",
)


def load_forecast_binding(
    profile_id: str,
    bindings_root: Path,
    *,
    bundled_root: Path | None = None,
) -> Workload:
    """Load a binding only when it is a trusted overlay of a bundled profile."""
    bundled = _load_catalog(bundled_root or bundled_workloads_path())
    if profile_id not in bundled:
        raise ForecastRunError("profile-unknown", f"no bundled profile is named {profile_id}")
    bindings = _load_catalog(bindings_root)
    binding = bindings.get(profile_id)
    if binding is None:
        raise ForecastRunError(
            "binding-unavailable", f"no local binding is installed for {profile_id}"
        )
    if apply_model_bindings(bundled, {profile_id: binding})[profile_id] is not binding:
        raise ForecastRunError("binding-untrusted", "binding changes bundled profile policy")
    _validate_binding_models(binding)
    return binding


def _load_catalog(root: Path) -> dict[str, Workload]:
    try:
        return load_workloads(root)
    except (FileNotFoundError, ManifestError, OSError, ValueError) as error:
        raise ForecastRunError("binding-invalid", str(error)) from error


def _validate_binding_models(binding: Workload) -> None:
    if not binding.models:
        raise ForecastRunError("binding-invalid", "binding declares no model")
    contracts = [model.get("featureContract") for model in binding.models]
    if any(contract is None for contract in contracts):
        raise ForecastRunError(
            "feature-contract-missing", "every model must declare featureContract"
        )
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise ForecastRunError("feature-contract-mismatch", "model feature contracts disagree")
    for model in binding.models:
        if "sha256" not in model:
            raise ForecastRunError("model-unpinned", f"{model['format']} model has no digest")


def latest_forecast_payload(
    workload: Workload,
    rows: Sequence[FeatureRow],
) -> dict:
    """Build the sole tensor shape accepted by the trusted forecast path."""
    model = workload.model
    contract = model.get("featureContract") if model else None
    if contract is None:
        raise ForecastRunError("feature-contract-missing", "workload has no feature contract")
    _validate_history(workload.id, rows)
    window = contract["window"]
    if len(rows) < window:
        raise ForecastRunError(
            "insufficient-history", f"forecast needs {window} rows, got {len(rows)}"
        )
    return {"inputs": [[_flatten_window(rows[-window:], contract["featureNames"])]]}


def _validate_history(profile_id: str, rows: Sequence[FeatureRow]) -> None:
    previous = -1
    for row in rows:
        if row.profile_id != profile_id:
            raise ForecastRunError("history-profile-mismatch", "history contains another profile")
        if row.observed_at_ms <= previous:
            raise ForecastRunError(
                "observations-unordered", "history timestamps must increase strictly"
            )
        previous = row.observed_at_ms


def _flatten_window(rows: Sequence[FeatureRow], feature_names: Sequence[str]) -> list[float]:
    values: list[float] = []
    for row in rows:
        for name in feature_names:
            value = row.features.get(name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ForecastRunError("features-missing", f"latest window has no finite {name}")
            values.append(float(value))
    return values


def wire_document(text: str, schema: str) -> dict:
    """Parse one bounded, canonical runtime reply."""
    if not isinstance(text, str):
        raise ForecastRunError("runtime-response-invalid", "runtime reply is not text")
    try:
        encoded = text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ForecastRunError("runtime-response-invalid", "runtime reply is not UTF-8") from error
    if len(encoded) > MAX_WIRE_BYTES:
        raise ForecastRunError("runtime-response-invalid", "runtime reply exceeds 1 MiB")
    try:
        document = json.loads(text)
    except (TypeError, ValueError) as error:
        raise ForecastRunError("runtime-response-invalid", "runtime reply is not JSON") from error
    violations = validate_document(schema, document)
    if violations or not isinstance(document, dict):
        detail = "; ".join(violations) if violations else "reply must be an object"
        raise ForecastRunError("runtime-response-invalid", detail)
    return document


def _runtime_job_schema_versions() -> dict[str, int]:
    """Snapshot the four job contracts this client actually exchanges."""
    available = schema_versions()
    try:
        return {name: available[name] for name in REQUIRED_SCHEMAS}
    except KeyError as error:
        raise ForecastRunError(
            "runtime-contract-mismatch", "local runtime job schema versions are incomplete"
        ) from error


class TrustedForecastRunner:
    """Handshake, assemble measured history, submit, and poll one forecast."""

    def __init__(
        self,
        workload: Workload,
        recorder: TelemetryRecorder,
        client: ForecastClient,
        *,
        attempts: int = 40,
        poll_interval: float = 0.1,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        request_id: Callable[[], str] = lambda: uuid.uuid4().hex,
    ) -> None:
        if isinstance(attempts, bool) or not isinstance(attempts, int) or not 1 <= attempts <= 120:
            raise ForecastRunError("bounds-invalid", "attempts must be from 1 to 120")
        if (
            isinstance(poll_interval, bool)
            or not isinstance(poll_interval, (int, float))
            or not 0 <= poll_interval <= 5
        ):
            raise ForecastRunError("bounds-invalid", "poll interval must be from 0 to 5 seconds")
        self._workload = workload
        self._recorder = recorder
        self._client = client
        self._attempts = attempts
        self._poll_interval = float(poll_interval)
        self._sleep = sleep
        self._request_id = request_id

    async def run(self) -> dict:
        required_schemas = _runtime_job_schema_versions()
        contract = wire_document(
            await self._client.describe_contract(), "runtime-contract.schema.json"
        )
        if not REQUIRED_METHODS.issubset(contract["methods"]):
            raise ForecastRunError(
                "runtime-contract-mismatch", "runtime lacks forecast job methods"
            )
        if any(
            contract["schemas"].get(name) != version for name, version in required_schemas.items()
        ):
            raise ForecastRunError(
                "runtime-contract-mismatch", "runtime job schema versions differ"
            )
        payload = latest_forecast_payload(self._workload, self._recorder.rows(self._workload.id))
        request_id = self._request_id()
        request = json.dumps(
            {
                "version": required_schemas["runtime-job-submit"],
                "requestId": request_id,
                "workloadId": self._workload.id,
                "payload": payload,
            },
            separators=(",", ":"),
            allow_nan=False,
        )
        reply = wire_document(
            await self._client.submit_job(request),
            "runtime-job-acknowledgement.schema.json",
        )
        if reply["requestId"] != request_id:
            raise ForecastRunError("runtime-response-invalid", "submission request id differs")
        if reply["status"] != "accepted" or reply["jobId"] is None:
            raise ForecastRunError(reply["code"], reply["message"])
        return await self._poll(reply["jobId"], required_schemas)

    async def _poll(self, job_id: str, schema_snapshot: dict[str, int]) -> dict:
        for attempt in range(self._attempts):
            request_id = self._request_id()
            request = json.dumps(
                {
                    "version": schema_snapshot["runtime-job-result-request"],
                    "requestId": request_id,
                    "jobId": job_id,
                },
                separators=(",", ":"),
            )
            reply = wire_document(
                await self._client.get_job_result(request),
                "runtime-job-result.schema.json",
            )
            if reply["requestId"] != request_id or reply["jobId"] != job_id:
                raise ForecastRunError("runtime-response-invalid", "result identity differs")
            if reply["state"] == "succeeded":
                if reply["output"] is None:
                    raise ForecastRunError(
                        "runtime-response-invalid", "successful result has no output"
                    )
                return reply["output"]
            if reply["state"] != "running":
                raise ForecastRunError(reply["code"], reply["message"])
            if attempt + 1 < self._attempts:
                await self._sleep(self._poll_interval)
        raise ForecastRunError("forecast-timeout", "forecast did not finish within the poll limit")
