"""Real-hardware Vulkan smoke test.

Skipped automatically when ncnn or a non-software Vulkan device is absent,
so CI on machines without a GPU stays green while a developer machine
proves the actual Vulkan compute path end to end.
"""

from __future__ import annotations

import os

import pytest

from omnitensor.executors.vulkan import VulkanGpuExecutor

if os.environ.get("MUTANT_UNDER_TEST"):
    pytest.skip(
        "real-hardware smoke test segfaults native code under mutated inputs",
        allow_module_level=True,
    )

ncnn = pytest.importorskip("ncnn")

ABSVAL_PARAM = """7767517
2 2
Input            in0      0 1 in0
AbsVal           abs0     1 1 in0 out0
"""


@pytest.fixture
def absval_model(tmp_path):
    param = tmp_path / "absval.param"
    param.write_text(ABSVAL_PARAM)
    (tmp_path / "absval.bin").write_bytes(b"")
    return str(param)


def test_absval_runs_on_a_real_vulkan_device(absval_model):
    executor = VulkanGpuExecutor(device_present=True)
    availability = executor.availability()
    if not availability.available:
        pytest.skip(f"no usable Vulkan device: {availability.reason}")
    result = executor.run(absval_model, [[-1.5, 2.0, -3.0]])
    assert result.outputs == [[1.5, 2.0, 3.0]]
    assert result.duration_ms >= 0


ABSVAL_MODEL = {
    "id": "absval-model",
    "version": "1.0.0",
    "format": "ncnn",
    "fullyQuantized": True,
    "minimumCompilerVersion": "1.0.0",
    "minimumRuntimeVersion": "1.0.0",
}


def test_a_dispatched_job_runs_on_a_real_vulkan_device(absval_model):
    """The whole join — catalog, artifact, routing, scheduler, GPU — end to end."""
    import asyncio
    from pathlib import Path

    from conftest import sample_manifest

    from omnitensor.dispatch import InferenceJobDispatcher, inference_result_payload
    from omnitensor.plugins.artifacts import ArtifactResolution
    from omnitensor.registry import Workload
    from omnitensor.scheduler import Scheduler

    executor = VulkanGpuExecutor(device_present=True)
    availability = executor.availability()
    if not availability.available:
        pytest.skip(f"no usable Vulkan device: {availability.reason}")

    manifest = sample_manifest(
        "absval-workload", accelerator="gpu", acceleratorPreference=["gpu"]
    )
    manifest["requirements"]["model"] = dict(ABSVAL_MODEL)
    workload = Workload(id="absval-workload", manifest=manifest)

    class ActiveArtifact:
        def resolve(self, reference):  # pragma: no cover - v1 manifest has no digest
            raise AssertionError("a v1 manifest declares no digest")

        def resolve_active(self, artifact_id):
            assert artifact_id == "absval-model"
            return ArtifactResolution(True, Path(absval_model), "", 1)

    async def scenario():
        executors = {"gpu": executor}
        scheduler = Scheduler(executors, lambda _profile: 1)
        scheduler.start()
        dispatcher = InferenceJobDispatcher(
            {workload.id: workload}, scheduler, executors, ActiveArtifact()
        )
        try:
            result = await asyncio.wait_for(
                dispatcher.dispatch(
                    "job-1", "absval-workload", {"inputs": [[-1.5, 2.0, -3.0]]}
                ),
                timeout=30,
            )
        finally:
            await scheduler.stop()
        return inference_result_payload(result)

    payload = asyncio.run(scenario())

    assert payload["outputs"] == [[1.5, 2.0, 3.0]]
    assert payload["durationMs"] >= 0
