"""The one deficit-round-robin implementation both schedulers now share."""

from __future__ import annotations

import pytest

from omnitensor.plugin_admission import PluginAdmissionQueue
from omnitensor.scheduler import _BackendQueue
from omnitensor.stride import MAX_STRIDE_WEIGHT, StrideQueue


def served(queue: StrideQueue, weights: dict, count: int, admits=None) -> list[str]:
    order = []
    for _index in range(count):
        item = queue.pop(lambda key: weights.get(key, 1), admits)
        if item is None:
            break
        order.append(item[0])
    return order


def test_weight_decides_how_often_a_key_is_served():
    queue: StrideQueue[str] = StrideQueue()
    for index in range(10):
        queue.push("heavy", f"h{index}")
        queue.push("light", f"l{index}")

    order = served(queue, {"heavy": 5, "light": 1}, 6)

    assert order.count("heavy") == 5
    assert order.count("light") == 1


def test_a_newcomer_joins_at_virtual_time_instead_of_bursting():
    queue: StrideQueue[str] = StrideQueue()
    queue.push("first", "a")
    queue.push("first", "b")
    served(queue, {}, 2)

    queue.push("first", "c")
    queue.push("late", "d")

    # The late arrival does not get a run of catch-up service it never waited for.
    assert served(queue, {}, 2) == ["first", "late"]


def test_a_held_key_neither_banks_credit_nor_is_punished():
    queue: StrideQueue[str] = StrideQueue()
    for index in range(4):
        queue.push("held", f"x{index}")
        queue.push("open", f"y{index}")

    served(queue, {}, 3, admits=lambda key: key == "open")
    order = served(queue, {}, 4)

    # Re-admitted, it alternates rather than taking every slot back at once.
    assert order[:2] == ["held", "open"]


@pytest.mark.parametrize("weight", [0, -3, True, "five", None, 99])
def test_an_impossible_weight_is_a_stride_of_one_rather_than_a_crash(weight):
    queue: StrideQueue[str] = StrideQueue()
    queue.push("a", "job")
    queue.push("a", "job-2")

    assert queue.pop(lambda _key: weight) == ("a", "job")
    assert 1.0 / MAX_STRIDE_WEIGHT <= queue.passes["a"] <= 1.0


def test_a_drained_key_is_pruned_from_every_table():
    queue: StrideQueue[str] = StrideQueue()
    queue.push("a", "job")
    assert queue.pop(lambda _key: 1) == ("a", "job")

    assert queue.items == {}
    assert queue.order == []
    assert queue.passes == {}
    assert queue.pop(lambda _key: 1) is None


def test_both_schedulers_are_built_on_it():
    """OMNI-0458: the rule was implemented twice and had begun to drift."""
    assert issubclass(_BackendQueue, StrideQueue)
    pool = PluginAdmissionQueue(weight_of=lambda _profile: 1)
    assert isinstance(pool._queue, StrideQueue)  # noqa: SLF001
