"""The private fragment store, which no workload owns."""

import asyncio

import pytest
from omnitensor.plugins.fragments import (
    FragmentStoreError,
    MemoryFragmentStore,
    PrivateFragmentStore,
    SourceFragment,
)


def test_memory_store_rejects_invalid_ids_and_missing_fragments():
    store = MemoryFragmentStore()
    with pytest.raises(FragmentStoreError, match="request id"):
        asyncio.run(store.discard("bad/id"))
    with pytest.raises(FragmentStoreError) as empty:
        asyncio.run(store.publish("job", ()))
    assert empty.value.code == "source-empty"

    fragment = SourceFragment("private:job:page:1", "a" * 64, 1, "text", "b" * 64)
    asyncio.run(store.publish("job", (fragment,)))
    assert store.resolve("job", fragment.reference) == fragment
    with pytest.raises(FragmentStoreError) as duplicate:
        asyncio.run(store.publish("job", (fragment,)))
    assert str(duplicate.value) == "source-invalid: private fragment reference repeats"
    with pytest.raises(FragmentStoreError) as invalid:
        asyncio.run(store.publish("other", (object(),)))
    assert invalid.value.detail == "private fragment has invalid type"
    asyncio.run(store.discard("job"))


def test_the_store_satisfies_the_port_the_runtime_depends_on():
    """The adapter type-checks the Protocol, so an incomplete one must fail it."""

    class Partial:
        async def publish(self, request_id, fragments): ...

        async def discard(self, request_id): ...

    assert isinstance(MemoryFragmentStore(), PrivateFragmentStore)
    assert not isinstance(Partial(), PrivateFragmentStore)
