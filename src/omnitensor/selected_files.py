"""Shared policy for files explicitly selected through the broker."""

# Large videos routinely exceed the old 128 MiB document-oriented default.
# One GiB is already the canonical snapshot contract's accepted ceiling, and
# keeping one owner makes the published promise equal the staging enforcement.
MAX_SELECTED_SOURCE_BYTES = 1024 * 1024 * 1024


__all__ = ["MAX_SELECTED_SOURCE_BYTES"]
