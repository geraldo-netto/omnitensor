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
    # Pinned, because an unpinned model is refused before it reaches a device —
    # and ncnn keeps its weights in the companion, so that is pinned too.
    "sha256": "a" * 64,
    "companions": {"model.bin": "b" * 64},
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

    class DeclaredArtifact:
        def resolve(self, reference):
            assert reference.id == "absval-model"
            assert reference.sha256 == ABSVAL_MODEL["sha256"]
            return ArtifactResolution(True, Path(absval_model), "", 1)

        def resolve_active(self, artifact_id):  # pragma: no cover - never consulted
            raise AssertionError("a pinned model is resolved by its digest")

    async def scenario():
        executors = {"gpu": executor}
        scheduler = Scheduler(executors, lambda _profile: 1)
        scheduler.start()
        dispatcher = InferenceJobDispatcher(
            {workload.id: workload}, scheduler, executors, DeclaredArtifact()
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


def test_a_declared_batch_axis_does_not_become_a_one_channel_mat(absval_model):
    """An ncnn Mat is CHW and carries no batch axis.

    A contract states its input as NCHW, so handing ``[1, C, H, W]`` straight
    to ``Mat`` builds a four-dimensional Mat with one channel. The network runs
    on it and returns a full set of scores, so the failure looks exactly like a
    working inference — every classification produced this way was of a tensor
    assembled from the wrong axes.
    """
    executor = VulkanGpuExecutor(device_present=True)
    availability = executor.availability()
    if not availability.available:
        pytest.skip(f"no usable Vulkan device: {availability.reason}")

    planar = [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]], [[9.0, 10.0], [11.0, 12.0]]]

    batched = executor._to_mat([planar])
    bare = executor._to_mat(planar)

    assert (batched.dims, batched.c) == (3, 3), "the batch axis is dropped, not carried"
    assert (bare.dims, bare.c) == (3, 3)
    assert batched.w == bare.w and batched.h == bare.h


def test_a_real_batch_is_refused_rather_than_silently_truncated():
    """Answering for the first image would discard the rest without saying so."""
    executor = VulkanGpuExecutor(device_present=True)
    if not executor.availability().available:
        pytest.skip("no usable Vulkan device")

    with pytest.raises(RuntimeError, match="one image at a time"):
        executor._to_mat([[[[1.0]]], [[[2.0]]]])


def test_the_pixels_survive_being_handed_to_ncnn(absval_model):
    """A Mat must own what it points at.

    ``ncnn.Mat(array)`` wraps the numpy buffer and does not keep the object
    that owns it alive, so the Mat the executor built died with the array the
    moment it was constructed — and referenced tensors arrive as nested lists,
    so that array was always a fresh allocation. The GPU upload happens later,
    during extraction, and read whatever the allocator had since put in those
    pages. Usually nothing had, which is why this presented as an intermittent
    reproducibility problem rather than as a use-after-free.

    Allocation churn between building the Mat and running the model is what
    makes the difference visible, so this test creates it deliberately.
    """
    numpy = pytest.importorskip("numpy")
    executor = VulkanGpuExecutor(device_present=True)
    availability = executor.availability()
    if not availability.available:
        pytest.skip(f"no usable Vulkan device: {availability.reason}")

    # Big enough that the array is its own allocation rather than a few bytes
    # in an arena, which is what a real input is and what gets reused.
    values = [float(index % 17) - 8.0 for index in range(3 * 227 * 227)]
    mat = executor._to_mat(values)

    # Nothing else holds the source array now. Reuse the pages hard.
    churn = [numpy.full(len(values), float(index), dtype=numpy.float32) for index in range(64)]
    del churn

    assert list(mat.numpy().ravel()) == values


def test_the_same_input_gives_the_same_answer_every_time(absval_model):
    """Reproducibility, asserted rather than assumed.

    Until this held, no accuracy number measured on the GPU lane meant
    anything: the same bytes scored differently from one submission to the
    next.
    """
    executor = VulkanGpuExecutor(device_present=True)
    if not executor.availability().available:
        pytest.skip("no usable Vulkan device")

    values = [float(index % 17) - 8.0 for index in range(3 * 32 * 32)]
    answers = {tuple(executor.run(absval_model, [values]).outputs[0]) for _ in range(8)}

    assert len(answers) == 1, "the same input produced more than one answer"
