"""Refusing to measure a card nobody asked for.

Nothing here touches a GPU or assumes one exists: the devices are fabricated,
because what is worth holding is the choosing and the refusing, not this desk's
hardware.
"""

from __future__ import annotations

import pytest

from omnitensor.benchmark.vulkan_devices import DeviceError, VulkanDevice, confirm, select

FIRST = VulkanDevice(0, "Vendor Model A (DRIVER GEN1)", True)
SECOND = VulkanDevice(1, "Vendor Model B (DRIVER GEN2)", False)
BOTH = (FIRST, SECOND)


def test_a_name_fragment_finds_the_one_card_that_matches():
    assert select("Model B", BOTH) is SECOND
    assert select("gen1", BOTH) is FIRST


def test_a_fragment_matching_two_cards_is_refused_rather_than_guessed():
    with pytest.raises(DeviceError, match="more than one device"):
        select("Vendor", BOTH)


def test_a_name_no_card_has_lists_what_this_machine_offers():
    with pytest.raises(DeviceError) as refusal:
        select("Model C", BOTH)

    assert "Model A" in str(refusal.value)
    assert "Model B" in str(refusal.value)


def test_an_index_is_checked_against_the_list_rather_than_trusted():
    assert select("1", BOTH) is SECOND

    with pytest.raises(DeviceError, match="no Vulkan device at index 9"):
        select("9", BOTH)


def test_a_load_on_another_card_is_refused_before_it_becomes_a_result():
    # A run on 2026-08-17 asked for the integrated card, got the discrete one,
    # and wrote twenty minutes of numbers under the wrong heading. The index
    # selecting a card and the index naming it were never compared.
    confirm(SECOND, "  vendor model b (driver gen2) ")

    with pytest.raises(DeviceError) as refusal:
        confirm(FIRST, "Vendor Model B (DRIVER GEN2)")

    assert "would describe the wrong card" in str(refusal.value)
