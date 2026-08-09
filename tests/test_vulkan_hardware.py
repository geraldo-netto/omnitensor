"""Real-hardware Vulkan smoke test.

Skipped automatically when ncnn or a non-software Vulkan device is absent,
so CI on machines without a GPU stays green while a developer machine
proves the actual Vulkan compute path end to end.
"""

from __future__ import annotations

import pytest

from omnitensor.executors.vulkan import VulkanGpuExecutor

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
