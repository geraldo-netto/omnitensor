"""Deficit-round-robin over per-key queues, in one place.

Two schedulers need the same rule: the pending key with the smallest pass value
runs next, and its pass advances by ``1 / weight``, so a weight-5 profile is
served about five times per weight-1 serve while both have work waiting. It was
implemented twice, character for character - once for accelerator lanes in
:mod:`omnitensor.scheduler`, once for plugin worker slots in
:mod:`omnitensor.plugin_admission` - and had already drifted: the stride
ceiling was a bare literal in one and a named constant in the other, and only
one of them defended against a non-integer weight.

Three properties travel with the rule and are easy to lose in a re-derivation:

* **A newcomer joins at virtual time**, never at zero, so it cannot monopolise
  the device catching up on service it never queued for.
* **A held-out key is clamped up to virtual time**, never past it, so being
  refused by policy neither banks credit nor is punished for waiting.
* **A drained key is pruned**, so the three tables cannot grow without bound.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from typing import Generic, TypeVar

# Policy weights are 1..5; a wider stride ceiling would let one profile starve
# every other for longer than the policy vocabulary can express.
MAX_STRIDE_WEIGHT = 5

Item = TypeVar("Item")


class StrideQueue(Generic[Item]):
    """Per-key FIFO queues served by stride order."""

    __slots__ = ("items", "order", "passes")

    def __init__(self) -> None:
        self.items: dict[str, deque[Item]] = {}
        self.order: list[str] = []
        self.passes: dict[str, float] = {}

    def push(self, key: str, item: Item) -> None:
        if key not in self.items:
            self.items[key] = deque()
            self.order.append(key)
            self.passes[key] = min(self.passes.values(), default=0.0)
        self.items[key].append(item)

    def depth(self) -> int:
        # Over a snapshot of the values: a publisher reads this from a worker
        # thread while the event loop pops and prunes, and a live view raises
        # "dictionary changed size during iteration" there.
        return sum(len(queue) for queue in list(self.items.values()))

    def choose(
        self,
        weight_of: Callable[[str], object],
        admits: Callable[[str], bool] | None = None,
    ) -> str | None:
        """The key to serve next, with its pass advanced, or ``None``."""
        pending = [key for key in self.order if self.items[key] and (admits is None or admits(key))]
        if not pending:
            return None
        chosen = min(pending, key=lambda key: (self.passes[key], self.order.index(key)))
        self.passes[chosen] += 1.0 / _stride_weight(weight_of(chosen))
        self.clamp_held_out(admits)
        return chosen

    def pop(
        self,
        weight_of: Callable[[str], object],
        admits: Callable[[str], bool] | None = None,
    ) -> tuple[str, Item] | None:
        """Take the next item in stride order, pruning a drained key."""
        chosen = self.choose(weight_of, admits)
        if chosen is None:
            return None
        item = self.items[chosen].popleft()
        if not self.items[chosen]:
            self.prune(chosen)
        return chosen, item

    def discard(self, key: str, item: Item) -> bool:
        """Remove one queued item, pruning the key when it empties."""
        queue = self.items.get(key)
        if queue is None:
            return False
        try:
            queue.remove(item)
        except ValueError:
            return False
        if not queue:
            self.prune(key)
        return True

    def prune(self, key: str) -> None:
        self.items.pop(key, None)
        if key in self.order:
            self.order.remove(key)
        self.passes.pop(key, None)

    def clamp_held_out(self, admits: Callable[[str], bool] | None) -> None:
        """Keep a policy-held key level with virtual time, never past it."""
        if admits is None:
            return
        admitted = [key for key, queue in self.items.items() if queue and admits(key)]
        if not admitted:
            return
        virtual_time = min(self.passes[key] for key in admitted)
        for key, queue in self.items.items():
            if queue and not admits(key):
                self.passes[key] = max(self.passes[key], virtual_time)


def _stride_weight(weight: object) -> int:
    """A usable stride divisor from whatever policy supplied."""
    if isinstance(weight, bool) or not isinstance(weight, int):
        return 1
    return max(1, min(MAX_STRIDE_WEIGHT, weight))


__all__ = ["MAX_STRIDE_WEIGHT", "StrideQueue"]
