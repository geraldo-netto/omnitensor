"""Vulkan GPU executor via ncnn.

Runs ``ncnn`` models (``.param`` + sibling ``.bin``) with Vulkan compute,
so any Mesa/RADV/ANV/NVIDIA Vulkan driver works without CUDA or ROCm.
Device selection honours the no-CPU rule: software Vulkan devices
(llvmpipe, type ``cpu``) are never used — if only a software device is
present the backend reports unavailable instead of silently running on
the host CPU.  Discrete devices are preferred over integrated ones.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from .base import (
    DEFAULT_MAX_CACHED_MODELS,
    DEVICE_ABSENT,
    RUNTIME_MISSING,
    RUNTIME_UNUSABLE,
    Availability,
    InferenceResult,
    ModelStore,
)

# ncnn VkGpuInfo.type(): 0 discrete, 1 integrated, 2 virtual, 3 cpu (software).
_DEVICE_PREFERENCE = {0: 0, 1: 1, 2: 2}


@dataclass(frozen=True, slots=True)
class VulkanDevice:
    """One hardware Vulkan device selected by the shared runtime policy."""

    index: int
    name: str
    kind: int
    # None means ncnn reported no id for this device, not "id zero" and not
    # the unmatchable sentinel a request carries: an unknown identity must
    # match nothing rather than match everything else that is also unknown.
    vendor_id: int | None = None
    device_id: int | None = None


# A request whose ids could not be read from sysfs. It is deliberately
# unmatchable: a selected render node that cannot be tied to a Vulkan identity
# must be refused, never quietly served by ncnn's preferred GPU.
UNMATCHABLE_DEVICE_ID = -1


@dataclass(frozen=True, slots=True)
class VulkanDeviceRequest:
    """One sysfs hardware identity and its occurrence among identical GPUs."""

    vendor_id: int
    device_id: int
    occurrence: int = 0

    @property
    def unmatchable(self) -> bool:
        return UNMATCHABLE_DEVICE_ID in (self.vendor_id, self.device_id)


class VulkanSelectionError(RuntimeError):
    """A stable Vulkan selection refusal with a caller-specific rendering."""

    def __init__(self, kind: str, reason: str):
        self.kind = kind
        self.reason = reason
        super().__init__(reason)


def _numeric_info(info, name: str) -> int | None:
    getter = getattr(info, name, None)
    if not callable(getter):
        return None
    value = getter()
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _requested_device(candidates, requested):
    if isinstance(requested, VulkanDeviceRequest):
        if requested.unmatchable:
            raise VulkanSelectionError(
                "requested",
                "Selected DRM GPU has no hardware identity to match against",
            )
        matching = [
            item[2]
            for item in candidates
            if (item[2].vendor_id, item[2].device_id) == (requested.vendor_id, requested.device_id)
        ]
        if requested.occurrence < len(matching):
            return matching[requested.occurrence]
        raise VulkanSelectionError(
            "requested",
            "Selected DRM GPU is absent from the hardware Vulkan inventory",
        )
    matching = [item for item in candidates if item[1] == requested]
    if not matching:
        raise VulkanSelectionError(
            "requested",
            f"Vulkan device {requested} is absent or software-only",
        )
    return matching[0][2]


def hardware_vulkan_devices(runtime) -> tuple[int, tuple[tuple[int, int, VulkanDevice], ...]]:
    """Every hardware Vulkan device the runtime reports, and how many it saw.

    The count is returned alongside because "no hardware device" and "no
    device at all" are different answers: the first means a software (CPU)
    device was present and refused, which is the no-CPU rule doing its job and
    has to be said differently from absent hardware.

    Anything that enumerates Vulkan has to come through here.  A second,
    weaker enumeration elsewhere is how a caller ends up on llvmpipe: it sees
    one device, calls it the obvious answer, and never looks at its type.
    """
    try:
        count = runtime.get_gpu_count()
        candidates = []
        for index in range(count):
            info = runtime.get_gpu_info(index)
            kind = info.type()
            if kind in _DEVICE_PREFERENCE:
                name_getter = getattr(info, "device_name", None)
                name = name_getter() if callable(name_getter) else ""
                vendor_id = _numeric_info(info, "vendor_id")
                device_id = _numeric_info(info, "device_id")
                candidates.append(
                    (
                        _DEVICE_PREFERENCE[kind],
                        index,
                        VulkanDevice(index, name, kind, vendor_id, device_id),
                    )
                )
    except Exception as error:  # noqa: BLE001 - native loader failures are arbitrary
        raise VulkanSelectionError("enumeration", str(error)) from error
    return count, tuple(candidates)


def select_vulkan_device(
    runtime,
    requested: int | VulkanDeviceRequest | None = None,
) -> VulkanDevice:
    """Choose the preferred hardware Vulkan device, never a CPU device."""
    count, candidates = hardware_vulkan_devices(runtime)
    if requested is not None:
        return _requested_device(candidates, requested)
    if not candidates:
        if count > 0:
            raise VulkanSelectionError(
                "software-only",
                "Only a software (CPU) Vulkan device is present; CPU execution is not used",
            )
        raise VulkanSelectionError("absent", "No Vulkan device is available")
    return min(candidates)[2]


def _import_ncnn():  # pragma: no cover - trivial import shim
    try:
        import ncnn  # noqa: PLC0415

        return ncnn
    except ImportError:
        return None


class VulkanGpuExecutor:
    backend = "gpu"
    model_formats = frozenset({"ncnn"})

    def __init__(
        self,
        device_present: bool,
        runtime=None,
        *,
        requested_device: int | VulkanDeviceRequest | None = None,
        max_cached_models: int = DEFAULT_MAX_CACHED_MODELS,
    ):
        self._device_present = device_present
        self._runtime = runtime if runtime is not None else _import_ncnn()
        self._requested_device = requested_device
        self._selected: tuple[int | None, str] | None = None
        self._nets = ModelStore(max_cached_models)

    def _select_device(self) -> tuple[int | None, str]:
        """Choose a device once and keep the answer.

        Enumerating Vulkan is not a read-only question: it initialises the
        loader and queries every driver.  The snapshot publisher asks whether
        this backend is available on every tick — twice a second — from a
        different thread from the one running inference, which is a great deal
        of driver work to repeat under a running job.

        Which device is present changes only when hardware does, and discovery
        rebuilds these executors when it does, so the answer is cached for the
        life of the executor rather than recomputed under a running job.

        This is hygiene, and it is credited with nothing.  It was once
        suspected of causing the same image to score differently from one
        submission to the next; measurement did not support that, and the
        cause turned out to be a Mat pointing at freed pixels (see
        ``_to_mat``).
        """
        if self._selected is None:
            self._selected = self._enumerate_device()
        return self._selected

    def _enumerate_device(self) -> tuple[int | None, str]:
        try:
            return select_vulkan_device(self._runtime, self._requested_device).index, ""
        except VulkanSelectionError as error:
            if error.kind == "enumeration":
                return None, f"Vulkan device enumeration failed: {error.reason}"
            return None, error.reason

    def availability(self) -> Availability:
        device, reason, code = self._available_device()
        return Availability(device is not None, reason, code)

    def _available_device(self) -> tuple[int | None, str, str]:
        if not self._device_present:
            return None, "No GPU render node detected", DEVICE_ABSENT
        if self._runtime is None:
            return None, "ncnn is not installed", RUNTIME_MISSING
        device, reason = self._select_device()
        return device, reason, "" if device is not None else RUNTIME_UNUSABLE

    def run(self, model_path: str, inputs: list) -> InferenceResult:
        # Resolve availability and device index in one enumeration.  Calling
        # availability() first would enumerate twice and let a hotplug event
        # shift the chosen index before set_vulkan_device().
        device, reason, _code = self._available_device()
        if device is None:
            raise RuntimeError(f"{self.backend} executor unavailable: {reason}")
        # The store counts this job in flight for the whole block, so an
        # eviction or a close from another thread parks the Net's teardown
        # until after the extractor below has been released.
        with self._nets.running():
            return self._extract(self._loaded(model_path, device), inputs)

    def _extract(self, net, inputs: list) -> InferenceResult:
        extractor = net.create_extractor()
        try:
            input_names = net.input_names()
            output_names = net.output_names()
            for name, value in zip(input_names, inputs, strict=True):
                extractor.input(name, self._to_mat(value))
            started = time.monotonic()
            outputs = []
            for name in output_names:
                code, mat = extractor.extract(name)
                if code != 0:
                    raise RuntimeError(f"ncnn extraction failed for output {name}")
                # Copied out of the Mat here, while the Net that owns its
                # memory is still alive.
                outputs.append(self._from_mat(mat))
                del mat
            duration_ms = (time.monotonic() - started) * 1000
        finally:
            # The extractor holds Vulkan memory the Net's allocators own, and
            # nothing ordered their teardown: letting both fall out of scope
            # released the allocators while the extractor still referenced
            # them, which ncnn reports as "pool allocator destroyed too early"
            # when it notices — and which otherwise leaves freed device memory
            # to be handed to a later job. The Net now outlives the job, so
            # this orders the extractor's release before anything else can
            # drop the last reference to the Net that owns those allocators.
            del extractor
        return InferenceResult(outputs=outputs, duration_ms=duration_ms)

    def close(self) -> None:
        """Release every loaded network; safe to call more than once.

        A network still under a running extraction is released once that job
        leaves, never underneath it.
        """
        self._nets.close()

    def _loaded(self, model_path: str, device: int):
        """The network for this model on this device, built at most once.

        Building it costs 60–190 ms on this machine — reading the weights,
        compiling every layer's shaders, and warming the driver's pipeline
        cache — against roughly 1 ms to run an image through it once it
        exists. Rebuilding per job spent between a tenth and a fifth of a
        second producing exactly the state the previous job had just
        discarded, and made `docs/installation.md`'s "roughly 2 ms once the
        model is cached" a claim about a cache that was not in this path.

        The entry is revalidated against both graph and weights on every
        lookup, so replacing an artifact in place serves the new model rather
        than the retired one, and the cache is bounded so a long-running
        service does not accumulate one network per model it has ever been
        asked for.

        Eviction releases the network through the store rather than dropping
        the reference for the collector to notice later — but never while a
        job is running on it: a job that is mid-extraction still holds the
        network it is running on, and tearing down its allocators from
        another thread is the failure this executor already had once, so the
        store parks that teardown until the last job has left.
        """
        param_path = Path(model_path)
        bin_path = param_path.with_suffix(".bin")

        def build():
            net = self._runtime.Net()
            net.opt.use_vulkan_compute = True
            # Transformer attention is not semantically stable under ncnn's
            # fp16 storage/arithmetic defaults on the Vulkan devices we
            # qualify.  Artifacts can still store explicitly converted fp16
            # weights, but runtime precision is kept at fp32 for every model.
            net.opt.use_fp16_packed = False
            net.opt.use_fp16_storage = False
            net.opt.use_fp16_arithmetic = False
            net.set_vulkan_device(device)
            if net.load_param(str(param_path)) != 0:
                raise RuntimeError(f"Could not load ncnn param: {param_path}")
            if net.load_model(str(bin_path)) != 0:
                raise RuntimeError(f"Could not load ncnn model: {bin_path}")
            return net

        # Device selection is immutable for this executor, so its cache and
        # the parameter path together identify the model/device pair. Jobs run
        # one at a time per backend, but the store's lock also prevents direct
        # callers from building the same network twice concurrently.
        return self._nets.get_or_build(
            str(param_path),
            build,
            companion_paths=(str(bin_path),),
        )

    def _to_mat(self, value):
        """Build the Mat ncnn expects from the tensor a caller declared.

        An ncnn Mat is channels-height-width and carries no batch axis, while
        a model contract states its input as NCHW — so handing ``[1, 3, H, W]``
        straight to ``Mat`` produces a *four-dimensional* Mat with a single
        channel.  The network still runs on it and returns a full set of
        scores, which is the worst possible failure: every classification this
        executor produced was of a one-channel tensor assembled from the wrong
        axes, and looked exactly like a working one.

        The clone is the other half of that.  ``Mat(array)`` *wraps* the numpy
        buffer — it shares the memory and does not take a reference to the
        object that owns it — so the Mat built here outlived its pixels the
        moment this function returned.  A referenced tensor arrives as nested
        lists, so the array is always freshly allocated and always freed on
        return, and the upload to the GPU does not happen until ``extract()``.
        Whatever the allocator had put in those pages by then was what the
        model classified.  Most of the time nothing had reused them and the
        answer was right, which is why this read as an intermittent
        reproducibility problem rather than as the use-after-free it was.
        """
        if isinstance(value, self._runtime.Mat):
            # Cloned like any other, because a Mat handed in from outside was
            # built the same way and carries the same borrowed memory.
            return self._owned(value)
        import numpy  # noqa: PLC0415 - shipped with the ncnn wheel

        observed = numpy.asarray(value)
        if observed.dtype.kind in "iu" and observed.dtype.kind != "b":
            limits = numpy.iinfo(numpy.int32)
            if observed.size and (observed.min() < limits.min or observed.max() > limits.max):
                raise RuntimeError("ncnn integer input exceeds int32 range")
            dtype = numpy.int32
        else:
            dtype = numpy.float32
        array = numpy.ascontiguousarray(value, dtype=dtype)
        while array.ndim > 3 and array.shape[0] == 1:
            array = array[0]
        if array.ndim > 3:
            # Answering for the first image of several would report a confident
            # result about one input while silently discarding the rest.
            raise RuntimeError(
                f"ncnn runs one image at a time; received a batch of {array.shape[0]}"
            )
        return self._owned(self._runtime.Mat(array))

    @staticmethod
    def _owned(mat):
        """A Mat that owns its pixels, so nothing can free them underneath it.

        Keeping the numpy array alive until extraction would also work and
        would save one copy of the input — 0.1 ms against a job that takes
        tens of milliseconds — but it makes correctness depend on a lifetime
        the reader has to reconstruct from two functions. This makes the Mat
        self-sufficient at the point it is built.
        """
        return mat.clone()

    @staticmethod
    def _from_mat(mat):
        return mat.numpy().tolist() if hasattr(mat, "numpy") else list(mat)
