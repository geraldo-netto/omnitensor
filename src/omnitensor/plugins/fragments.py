"""Worker-local private prompt material, shared by every generation workload.

The fragment and its store used to live in the event modules — the dataclass in
`events`, the store in `event_workload` — because event extraction was the first
workload that needed them. All four workloads publish fragments, and so the
native runtime adapter imported the event workload to name the type of its own
constructor argument: a GPU backend that could not be built without the
calendar. Nothing here knows what an event is.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..stable_error import StableError

_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,120}$")


class FragmentStoreError(StableError, ValueError):
    """Stable refusal with no private content in its detail."""


@dataclass(frozen=True, slots=True)
class SourceFragment:
    """Private prompt material retained inside the isolated worker."""

    reference: str
    source_sha256: str
    page: int
    text: str
    text_sha256: str


@runtime_checkable
class PrivateFragmentStore(Protocol):
    """Worker-local prompt material shared with the native generation port.

    ``resolve`` is part of the port because the runtime adapter calls it to
    assemble a prompt. It was omitted while the adapter named the concrete
    class instead, which made the Protocol describe less than the only
    collaboration it existed to describe.
    """

    async def publish(self, request_id: str, fragments: Sequence[SourceFragment]) -> None: ...

    async def discard(self, request_id: str) -> None: ...

    def resolve(self, request_id: str, reference: str) -> SourceFragment: ...


class MemoryFragmentStore:
    """Bounded process-local store suitable for native runtime adapters."""

    def __init__(self) -> None:
        self._requests: dict[str, dict[str, SourceFragment]] = {}

    async def publish(self, request_id: str, fragments: Sequence[SourceFragment]) -> None:
        _valid_request_id(request_id)
        if not fragments:
            raise FragmentStoreError("source-empty", "source produced no private fragments")
        current = dict(self._requests.get(request_id, {}))
        for fragment in fragments:
            if not isinstance(fragment, SourceFragment):
                raise FragmentStoreError("source-invalid", "private fragment has invalid type")
            if fragment.reference in current:
                raise FragmentStoreError("source-invalid", "private fragment reference repeats")
            current[fragment.reference] = fragment
        self._requests[request_id] = current

    async def discard(self, request_id: str) -> None:
        _valid_request_id(request_id)
        self._requests.pop(request_id, None)

    def resolve(self, request_id: str, reference: str) -> SourceFragment:
        """Resolve only inside the worker; callers never receive the mapping."""
        _valid_request_id(request_id)
        try:
            return self._requests[request_id][reference]
        except KeyError as error:
            raise FragmentStoreError(
                "source-unavailable", "private fragment is unavailable"
            ) from error


def _valid_request_id(value: str) -> None:
    if not isinstance(value, str) or _REQUEST_ID.fullmatch(value) is None:
        raise FragmentStoreError("request-invalid", "request id is invalid")


__all__ = [
    "FragmentStoreError",
    "MemoryFragmentStore",
    "PrivateFragmentStore",
    "SourceFragment",
]
