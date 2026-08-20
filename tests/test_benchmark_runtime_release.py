"""A benchmark run that fails must still give the card back.

The runtime holds VRAM and a cross-worker lease from `load()` until
`terminate()`. `measure()` treats one model failing as a result and moves on to
the next model, so any failure between the two — a device confirmation, an
unreadable case file, a case run that raises — used to leave the previous model
resident and the lease held while the next `load()` started.
"""

from __future__ import annotations

import sys
import types

import pytest

from omnitensor.benchmark import bench_cli, harness, prompt_trial
from omnitensor.benchmark.vulkan_devices import DeviceError, VulkanDevice

DEVICE = VulkanDevice(index=0, name="AMD Radeon RX 6600 XT", integrated=False)


class Report:
    accelerator_layers = 33
    total_model_layers = 33
    device = "AMD Radeon RX 6600 XT"


class FakeRuntime:
    """Records what a real `LlamaVulkanRuntime` would have held and released."""

    def __init__(self, store, lease_path):
        self.terminated: list[str] = []
        self.physical_device = DEVICE.name

    async def load(self, paths, lane):
        assert lane == "gpu"
        return Report()

    async def terminate(self, request_id: str) -> None:
        self.terminated.append(request_id)


@pytest.fixture
def provider(monkeypatch):
    """Stand in for the optional provider distribution, which is not installed."""
    built: list[FakeRuntime] = []

    def make(store, lease_path):
        runtime = FakeRuntime(store, lease_path)
        built.append(runtime)
        return runtime

    runtime_module = types.ModuleType("omnitensor_vulkan_runtime.runtime")
    runtime_module.MAX_RUNTIME_CONTEXT_TOKENS = 32_768
    runtime_module.LlamaVulkanRuntime = make
    package = types.ModuleType("omnitensor_vulkan_runtime")
    package.runtime = runtime_module
    package.workload_tasks = dict
    monkeypatch.setitem(sys.modules, "omnitensor_vulkan_runtime", package)
    monkeypatch.setitem(sys.modules, "omnitensor_vulkan_runtime.runtime", runtime_module)
    return built


def test_a_refused_device_confirmation_still_releases_the_model(provider, monkeypatch, tmp_path):
    monkeypatch.setattr(bench_cli, "model_path", lambda root, model_id: tmp_path / "m.gguf")

    def refuse(device, answered):
        raise DeviceError("asked for the integrated card, the discrete one answered")

    monkeypatch.setattr(harness, "confirm", refuse)

    with pytest.raises(DeviceError):
        bench_cli.run_model(
            "qwen3-8b-q4-k-m",
            ("event-extraction",),
            artifact_root=tmp_path,
            case_root=tmp_path,
            device=DEVICE,
            lease_root=tmp_path / "leases",
            say=lambda _message: None,
        )

    assert [runtime.terminated for runtime in provider] == [["__startup__"]]


def test_unreadable_cases_still_release_the_model(provider, monkeypatch, tmp_path):
    monkeypatch.setattr(bench_cli, "model_path", lambda root, model_id: tmp_path / "m.gguf")
    monkeypatch.setattr(harness, "confirm", lambda device, answered: None)
    monkeypatch.setattr(bench_cli, "tasks", lambda: {"event-extraction": object})

    def unreadable(workload, root):
        raise OSError("cases are gone")

    monkeypatch.setattr(bench_cli.case_files, "load", unreadable)

    with pytest.raises(OSError, match="cases are gone"):
        bench_cli.run_model(
            "qwen3-8b-q4-k-m",
            ("event-extraction",),
            artifact_root=tmp_path,
            case_root=tmp_path,
            device=DEVICE,
            lease_root=tmp_path / "leases",
            say=lambda _message: None,
        )

    assert [runtime.terminated for runtime in provider] == [["__startup__"]]


def test_a_prompt_trial_that_fails_still_releases_the_model(provider, monkeypatch, tmp_path):
    monkeypatch.setattr(prompt_trial, "model_path", lambda root, model_id: tmp_path / "m.gguf")
    monkeypatch.setattr(harness, "confirm", lambda device, answered: None)

    def unreadable(workload, root):
        raise OSError("cases are gone")

    monkeypatch.setattr(prompt_trial.case_files, "load", unreadable)

    with pytest.raises(OSError, match="cases are gone"):
        prompt_trial.trial(
            "event-extraction",
            object(),
            model_id="qwen3-4b-q4-k-m",
            device=DEVICE,
            artifact_root=tmp_path,
            case_root=tmp_path,
            lease_root=tmp_path / "leases",
            answerable_only=False,
            say=lambda _message: None,
        )

    assert [runtime.terminated for runtime in provider] == [["__startup__"]]
