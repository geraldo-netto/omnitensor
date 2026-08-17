from __future__ import annotations

from conftest import add_gpu, add_npu, add_pcie_tpu, add_usb_tpu

from omnitensor import discovery as discovery_module
from omnitensor.discovery import (
    detect_devices,
    detect_gpu,
    detect_gpus,
    detect_npu,
    detect_tpu,
)


def test_absence_reports_empty_list(fake_nodes):
    assert detect_devices(fake_nodes) == []


def test_pcie_tpu_wins_over_usb(fake_nodes):
    add_pcie_tpu(fake_nodes)
    add_usb_tpu(fake_nodes)
    device = detect_tpu(fake_nodes)
    assert device.id == "tpu-pcie-0"
    assert device.kind == "pcie"
    assert device.backend == "tpu"


def test_usb_tpu_identity_is_exact(fake_nodes):
    add_usb_tpu(fake_nodes, vendor="18d1", product="ffff")
    assert detect_tpu(fake_nodes) is None
    add_usb_tpu2 = fake_nodes.sys / "bus/usb/devices/2-1"
    add_usb_tpu2.mkdir(parents=True)
    (add_usb_tpu2 / "idVendor").write_text("18d1")
    (add_usb_tpu2 / "idProduct").write_text("9302")
    device = detect_tpu(fake_nodes)
    assert device.id == "tpu-usb"
    assert device.name == "Coral USB Accelerator"


def test_npu_vendor_mapping(fake_nodes):
    add_npu(fake_nodes, vendor="0x8086")
    assert detect_npu(fake_nodes).name == "Intel NPU"


def test_npu_unreadable_vendor_never_fails(fake_nodes):
    add_npu(fake_nodes, vendor=None)
    device = detect_npu(fake_nodes)
    assert device.available is True
    assert device.name == "NPU accelerator"
    assert device.vendor == ""


def test_gpu_vendor_mapping(fake_nodes):
    add_gpu(fake_nodes, node=130, vendor="0x1002")
    device = detect_gpu(fake_nodes)
    assert device.id == "gpu-renderD130"
    assert device.name == "AMD GPU"
    assert device.kind == "dri"


def test_runtime_device_ids_are_enumerated_and_can_be_selected(fake_nodes):
    add_pcie_tpu(fake_nodes, index=0)
    add_pcie_tpu(fake_nodes, index=42)
    add_npu(fake_nodes, index=0, vendor="0x8086")
    add_npu(fake_nodes, index=23, vendor="0x1022")
    add_gpu(fake_nodes, node=128, vendor="0x8086")
    add_gpu(fake_nodes, node=2048, vendor="0x1002")

    devices = detect_devices(
        fake_nodes,
        {
            "tpu": "tpu-pcie-42",
            "npu": "npu-accel23",
            "gpu": "gpu-renderD2048",
        },
    )

    assert [device.id for device in devices] == [
        "tpu-pcie-42",
        "npu-accel23",
        "gpu-renderD2048",
        "gpu-renderD128",
    ]


def test_explicit_device_selection_chooses_a_non_default_gpu(fake_nodes):
    add_gpu(fake_nodes, node=301, vendor="0x8086")
    add_gpu(fake_nodes, node=902, vendor="0x1002")

    assert detect_gpu(fake_nodes).id == "gpu-renderD301"
    assert detect_gpu(fake_nodes, "gpu-renderD902").id == "gpu-renderD902"
    assert detect_gpu(fake_nodes, "gpu-renderD999") is None


def test_all_gpu_identities_are_published_and_identical_hardware_is_disambiguated(
    fake_nodes,
):
    add_gpu(fake_nodes, node=128, vendor="0x1002", device="0x73ff")
    add_gpu(fake_nodes, node=129, vendor="0x1002", device="0x73ff")
    add_gpu(fake_nodes, node=130, vendor="0x8086", device="0x46a6")

    gpus = detect_gpus(fake_nodes)

    assert [gpu.id for gpu in gpus] == [
        "gpu-renderD128",
        "gpu-renderD129",
        "gpu-renderD130",
    ]
    assert [(gpu.vendor, gpu.hardware_id, gpu.identity_index) for gpu in gpus] == [
        ("0x1002", "0x73ff", 0),
        ("0x1002", "0x73ff", 1),
        ("0x8086", "0x46a6", 0),
    ]
    assert [gpu.id for gpu in detect_devices(fake_nodes)] == [
        "gpu-renderD128",
        "gpu-renderD129",
        "gpu-renderD130",
    ]


def test_device_discovery_ignores_names_that_only_resemble_accelerators(fake_nodes):
    (fake_nodes.dev / "dri").mkdir()
    for name in ("renderD", "renderD-1", "renderDabc", "card7", "renderD1234567"):
        (fake_nodes.dev / "dri" / name).touch()

    assert detect_gpu(fake_nodes) is None


def test_a_node_removed_between_enumeration_and_inspection_is_ignored(fake_nodes):
    assert (
        discovery_module._detect_node(
            fake_nodes.dev / "dri/renderD777",
            fake_nodes.sys / "class/drm/renderD777/device/vendor",
            discovery_module.GPU_VENDOR_NAMES,
            "GPU (render node)",
            "gpu-renderD777",
            "gpu",
            "dri",
        )
        is None
    )


def test_combined_order_is_tpu_npu_gpu(fake_nodes):
    add_gpu(fake_nodes)
    add_npu(fake_nodes)
    add_pcie_tpu(fake_nodes)
    backends = [device.backend for device in detect_devices(fake_nodes)]
    assert backends == ["tpu", "npu", "gpu"]


def test_snapshot_entry_shape(fake_nodes):
    add_npu(fake_nodes)
    entry = detect_npu(fake_nodes).snapshot_entry()
    assert entry == {
        "id": "npu-accel0",
        "backend": "npu",
        "available": True,
        "name": "Intel NPU",
        "kind": "accel",
        "vendor": "0x8086",
        "load": None,
        "reason": "",
    }


def test_gpu_utilization_reads_sysfs_busy_percent(fake_nodes):
    add_gpu(fake_nodes, node=128, vendor="0x1002")
    from omnitensor.discovery import device_utilization

    device = detect_gpu(fake_nodes)
    busy = fake_nodes.sys / "class/drm/renderD128/device/gpu_busy_percent"
    busy.write_text("37\n")
    assert device_utilization(fake_nodes, device) == 37.0
    busy.write_text("999\n")
    assert device_utilization(fake_nodes, device) == 100.0
    busy.write_text("garbage\n")
    assert device_utilization(fake_nodes, device) is None
    for non_finite in ("nan", "inf", "-inf"):
        busy.write_text(f"{non_finite}\n")
        assert device_utilization(fake_nodes, device) is None
    busy.unlink()
    assert device_utilization(fake_nodes, device) is None


def test_utilization_only_applies_to_render_nodes(fake_nodes):
    add_npu(fake_nodes)
    from omnitensor.discovery import device_utilization

    assert device_utilization(fake_nodes, detect_npu(fake_nodes)) is None
