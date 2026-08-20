"""The one failure this provider raises before it can do any work."""


class MediaGpuError(RuntimeError):
    """The provider cannot run this model on the GPU.

    There is deliberately no CPU path, so a missing accelerator grant, an
    absent runtime, or a load that did not end up on a Vulkan device is a
    refusal rather than something to fall back from.
    """


__all__ = ["MediaGpuError"]
