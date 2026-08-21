#!/usr/bin/env python3
"""Collect real control-socket selected-text observations from one installed GPU worker."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from omnitensor.atomicio import read_json_bounded, write_json_atomic
from omnitensor.plugins.selected_text_acceptance import (
    SelectedTextWorkerLoadReceipt,
    load_selected_text_corpus,
    load_selected_text_worker_load_receipt,
)
from omnitensor.preparation import file_digest
from omnitensor.registry import MAX_MANIFEST_BYTES, validate_workload_document
from omnitensor.socket_transport import call_control

SELECTED_TEXT_MANIFEST = (
    Path(__file__).resolve().parents[1] / "plugin-manifests/selected-text-tools.json"
)


def _arguments(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Run the frozen selected-text corpus through the installed service"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--qwen-model", type=Path, required=True)
    parser.add_argument("--hebrew-model", type=Path, required=True)
    parser.add_argument("--worker-load-receipt", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    return parser.parse_args(argv)


async def _terminal(call, job_id: str, timeout: float, prefix: str):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    poll = 0
    while loop.time() < deadline:
        result = await call(
            "get-job-result",
            {"version": 1, "requestId": f"{prefix}-result-{poll}", "jobId": job_id},
        )
        if result["state"] in {"succeeded", "failed", "cancelled", "unknown"}:
            return result
        poll += 1
        await asyncio.sleep(0.5)
    raise RuntimeError(f"job {job_id} did not terminate within {timeout} seconds")


async def _run_case(call, case, index: int, timeout: float):
    payload = {"selection": case.selection, "operation": case.operation}
    if case.language is not None:
        payload["language"] = case.language
    request_id = f"selected-acceptance-{index}"
    started = time.monotonic_ns()
    acknowledgement = await call(
        "submit-job",
        {
            "version": 1,
            "requestId": request_id,
            "workloadId": "selected-text-tools",
            "payload": payload,
        },
    )
    if acknowledgement["status"] != "accepted" or not acknowledgement["jobId"]:
        raise RuntimeError(f"{case.case_id} was not accepted: {acknowledgement}")
    terminal = await _terminal(call, acknowledgement["jobId"], timeout, request_id)
    latency_ms = (time.monotonic_ns() - started) // 1_000_000
    if terminal["state"] != "succeeded" or not isinstance(terminal.get("output"), dict):
        raise RuntimeError(f"{case.case_id} failed: {terminal}")
    return {"caseId": case.case_id, "result": terminal["output"], "latencyMs": latency_ms}


async def _cancellation_latency(call, timeout: float) -> int:
    request_id = "selected-acceptance-cancel"
    acknowledgement = await call(
        "submit-job",
        {
            "version": 1,
            "requestId": request_id,
            "workloadId": "selected-text-tools",
            "payload": {
                "selection": "Explain this checksum behavior. " * 900,
                "operation": "explain",
            },
        },
    )
    job_id = acknowledgement.get("jobId")
    if acknowledgement.get("status") != "accepted" or not job_id:
        raise RuntimeError(f"cancellation probe was not accepted: {acknowledgement}")
    started = time.monotonic_ns()
    cancelled = await call(
        "cancel-job",
        {"version": 1, "requestId": f"{request_id}-request", "jobId": job_id},
    )
    if cancelled["status"] not in {"cancelled", "accepted"}:
        raise RuntimeError(f"cancellation was refused: {cancelled}")
    terminal = await _terminal(call, job_id, timeout, request_id)
    if terminal["state"] != "cancelled":
        raise RuntimeError(f"cancellation did not reach a cancelled terminal: {terminal}")
    return (time.monotonic_ns() - started) // 1_000_000


def _load_selected_text_manifest() -> dict:
    manifest = read_json_bounded(SELECTED_TEXT_MANIFEST, MAX_MANIFEST_BYTES)
    if (
        not isinstance(manifest, dict)
        or manifest.get("id") != "selected-text-tools"
        or validate_workload_document(manifest)
    ):
        raise RuntimeError("selected-text manifest is invalid")
    return manifest


def _receipt_artifact_ids(
    receipt: SelectedTextWorkerLoadReceipt,
    manifest: dict,
) -> tuple[str, str]:
    declared = manifest["plugin"]["artifacts"]

    def match(digest: str, *, selectable: bool) -> str:
        matches = [
            item["id"]
            for item in declared
            if item["sha256"] == digest and (item.get("selectable", True) is selectable)
        ]
        if len(matches) != 1:
            role = "selectable primary" if selectable else "fixed Hebrew"
            raise RuntimeError(f"worker load receipt does not name one declared {role} artifact")
        return matches[0]

    return (
        match(receipt.primary.model_sha256, selectable=True),
        match(receipt.hebrew.model_sha256, selectable=False),
    )


def _require_ready(inventory, receipt: SelectedTextWorkerLoadReceipt, manifest: dict):
    matches = [
        item for item in inventory.get("plugins", []) if item.get("id") == "selected-text-tools"
    ]
    if len(matches) != 1:
        raise RuntimeError("installed selected-text worker identity is missing")
    plugin = matches[0]
    artifacts = {item["id"]: item for item in plugin.get("artifacts", [])}
    if plugin.get("version") != manifest["version"] or plugin.get("workerState") != "ready":
        raise RuntimeError(f"selected-text {manifest['version']} worker is not ready")
    expected = set(_receipt_artifact_ids(receipt, manifest))
    # The manifest may declare more artifacts than this corpus exercises, so
    # require the exact two the live receipt records and never refuse a worker
    # for having the rest of what the manifest asked for.
    missing = sorted(identifier for identifier in expected if identifier not in artifacts)
    if missing:
        raise RuntimeError(f"selected-text model artifacts are not installed: {', '.join(missing)}")
    unready = sorted(
        identifier for identifier in expected if not artifacts[identifier].get("ready")
    )
    if unready:
        raise RuntimeError(f"selected-text model artifacts are not ready: {', '.join(unready)}")


async def _collect(arguments):
    inventory = await call_control("describe-plugins", {})
    receipt = load_selected_text_worker_load_receipt(arguments.worker_load_receipt)
    manifest = _load_selected_text_manifest()
    _require_ready(inventory, receipt, manifest)
    if file_digest(arguments.qwen_model) != receipt.primary.model_sha256:
        raise RuntimeError("Qwen model path differs from the worker load receipt")
    if file_digest(arguments.hebrew_model) != receipt.hebrew.model_sha256:
        raise RuntimeError("Hebrew model path differs from the worker load receipt")
    corpus = load_selected_text_corpus(arguments.corpus)
    observations = []
    for index, case in enumerate(corpus.cases):
        observations.append(await _run_case(call_control, case, index, arguments.timeout_seconds))
    cancellation = await _cancellation_latency(call_control, arguments.timeout_seconds)
    public = json.dumps(observations, ensure_ascii=False)
    route_preserved = all(
        observation["result"]["providerId"] == case.expected_provider_id
        for observation, case in zip(observations, corpus.cases, strict=True)
    )
    return {
        "evidenceVersion": 1,
        "corpusSha256": corpus.sha256,
        "models": receipt.models_document(),
        "observations": observations,
        "safety": {
            "cancellationLatencyMs": cancellation,
            "privateFragmentsDiscarded": "private:" not in public,
            "defaultRoutePreserved": route_preserved,
        },
    }


def main(argv: list[str] | None = None):
    arguments = _arguments(argv)
    document = asyncio.run(_collect(arguments))
    write_json_atomic(arguments.output, document, prefix=".selected-text-evidence-")
    print(json.dumps(document, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
