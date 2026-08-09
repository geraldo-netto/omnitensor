from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from hypothesis import HealthCheck, settings

from omnitensor.discovery import DiscoveryPaths

# Deterministic, faster hypothesis runs under mutation testing so kills are
# reproducible and the mutmut wall clock stays bounded.
settings.register_profile(
    "mutmut",
    deadline=None,
    derandomize=True,
    max_examples=20,
    suppress_health_check=list(HealthCheck),
)
if os.environ.get("MUTANT_UNDER_TEST"):
    settings.load_profile("mutmut")


@pytest.fixture
def fake_nodes(tmp_path: Path) -> DiscoveryPaths:
    dev = tmp_path / "dev"
    sys = tmp_path / "sys"
    dev.mkdir()
    sys.mkdir()
    return DiscoveryPaths(dev=dev, sys=sys)


def add_pcie_tpu(paths: DiscoveryPaths, index: int = 0) -> None:
    (paths.dev / f"apex_{index}").touch()


def add_usb_tpu(paths: DiscoveryPaths, vendor: str = "18d1", product: str = "9302") -> None:
    device = paths.sys / "bus/usb/devices/1-1"
    device.mkdir(parents=True)
    (device / "idVendor").write_text(f"{vendor}\n")
    (device / "idProduct").write_text(f"{product}\n")


def add_npu(paths: DiscoveryPaths, index: int = 0, vendor: str | None = "0x8086") -> None:
    (paths.dev / "accel").mkdir(exist_ok=True)
    (paths.dev / f"accel/accel{index}").touch()
    if vendor is not None:
        vendor_dir = paths.sys / f"class/accel/accel{index}/device"
        vendor_dir.mkdir(parents=True)
        (vendor_dir / "vendor").write_text(f"{vendor}\n")


def add_gpu(paths: DiscoveryPaths, node: int = 128, vendor: str | None = "0x10de") -> None:
    (paths.dev / "dri").mkdir(exist_ok=True)
    (paths.dev / f"dri/renderD{node}").touch()
    if vendor is not None:
        vendor_dir = paths.sys / f"class/drm/renderD{node}/device"
        vendor_dir.mkdir(parents=True)
        (vendor_dir / "vendor").write_text(f"{vendor}\n")


def sample_manifest(workload_id: str = "sample-workload", **requirement_overrides) -> dict:
    requirements = {
        "runtimeApi": 1,
        "accelerator": "tpu",
        "acceleratorPreference": ["tpu", "npu", "gpu"],
        "minimumDevices": 1,
        "model": None,
    }
    requirements.update(requirement_overrides)
    return {
        "manifestVersion": 1,
        "id": workload_id,
        "version": "1.0.0",
        "capabilities": ["classify"],
        "requirements": requirements,
        "ui": {
            "title": "Sample workload",
            "group": "Examples",
            "description": "Classifies bounded sample inputs",
            "icon": "applications-science-symbolic",
            "order": 10,
        },
        "defaults": {"enabled": True, "weight": 2},
        "pipeline": {"hostResponsibilities": ["Decode bounded inputs"]},
        "acceptance": [],
    }


def write_workload(root: Path, manifest: dict) -> None:
    directory = root / manifest["id"]
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps(manifest))
