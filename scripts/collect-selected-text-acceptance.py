#!/usr/bin/env python3
"""Collect real control-socket selected-text observations from one installed GPU worker."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from omnitensor.atomicio import write_json_atomic
from omnitensor.plugins.selected_text_acceptance import (
    load_selected_text_corpus,
    load_selected_text_worker_load_receipt,
)
from omnitensor.preparation import file_digest
from omnitensor.socket_transport import call_control


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


def _require_ready(inventory):
    matches = [
        item for item in inventory.get("plugins", []) if item.get("id") == "selected-text-tools"
    ]
    if len(matches) != 1:
        raise RuntimeError("installed selected-text worker identity is missing")
    plugin = matches[0]
    artifacts = {item["id"]: item for item in plugin.get("artifacts", [])}
    if plugin.get("version") != "1.1.0" or plugin.get("workerState") != "ready":
        raise RuntimeError("selected-text 1.1.0 worker is not ready")
    expected = {"qwen3-8b-q4-k-m", "dictalm2-hebrew-q4-k-m"}
    if set(artifacts) != expected or not all(item.get("ready") for item in artifacts.values()):
        raise RuntimeError("selected-text model artifacts are not ready")


async def _collect(arguments):
    _require_ready(await call_control("describe-plugins", {}))
    receipt = load_selected_text_worker_load_receipt(arguments.worker_load_receipt)
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
