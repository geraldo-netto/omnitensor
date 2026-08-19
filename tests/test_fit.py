"""What fits on a card, and how the question has been answered wrongly before.

Every model decision on this desk has turned on three numbers nobody could
check: the weights, the key/value cache, and whatever else the runtime
allocates. This is where they become checkable — including against a real card,
whose own driver publishes what it has.

The two mistakes these pin are both mistakes that were actually made. Reading
only VRAM reports the integrated GPU as too small for a model it can hold
comfortably. Reading GTT for *any* card reports the discrete one as having 45
GiB, which is true and useless: those bytes are across a PCIe bus.
"""

from __future__ import annotations

import struct

import pytest

from omnitensor import gguf, probe_cli
from omnitensor.fit import (
    DEFAULT_MARGIN_BYTES,
    DeviceMemory,
    UnknownCacheError,
    device_memory,
    estimate,
    gibibytes,
    verdict,
)
from omnitensor.gguf import GgufError, ModelShape

GIB = 1024**3


def shape(**overrides) -> ModelShape:
    """A model the size of the one this desk actually runs."""
    base = {
        "name": "Qwen3 8B",
        "architecture": "qwen3",
        "file_bytes": int(4.68 * GIB),
        "blocks": 36,
        "kv_heads": 8,
        "key_length": 128,
        "value_length": 128,
    }
    base.update(overrides)
    return ModelShape(**base)


class TestTheCacheTerm:
    """The term people forget, and the one that decides the answer."""

    def test_a_token_costs_every_layer_and_both_halves_of_the_cache(self):
        # 36 blocks x 8 key/value heads x (128 + 128) elements.
        assert shape().kv_bytes_per_token == 73_728

    def test_at_thirty_two_thousand_tokens_the_cache_rivals_the_model(self):
        """Which is the whole reason a 4.7 GiB model does not fit an 8 GiB
        card with room to spare."""
        estimated = estimate(shape(), 32_768, cache="q8_0")

        assert 2.3 * GIB < estimated.cache_bytes < 2.5 * GIB
        assert estimated.total_bytes > 7.5 * GIB

    def test_a_shorter_context_is_a_smaller_cache_in_proportion(self):
        long = estimate(shape(), 32_768).cache_bytes
        short = estimate(shape(), 8_192).cache_bytes

        assert short == pytest.approx(long / 4, rel=0.01)

    def test_a_coarser_cache_is_cheaper_and_says_which_one_it_priced(self):
        """Recorded on the estimate rather than folded into a number, because
        it is a tradeoff somebody would be taking, not a fact about the model."""
        fine = estimate(shape(), 32_768, cache="q8_0")
        coarse = estimate(shape(), 32_768, cache="q4_0")

        assert coarse.cache_bytes < fine.cache_bytes
        assert (fine.cache, coarse.cache) == ("q8_0", "q4_0")

    def test_a_precision_llama_cpp_does_not_have_is_refused(self):
        with pytest.raises(UnknownCacheError):
            estimate(shape(), 1_024, cache="q3_k_m")

    def test_the_estimated_term_is_kept_separate_from_the_derived_ones(self):
        """An estimate that hides its guess is worse than no estimate."""
        estimated = estimate(shape(), 32_768, overhead_bytes=700 * 1024 * 1024)

        assert estimated.overhead_bytes == 700 * 1024 * 1024
        assert estimated.total_bytes == (
            estimated.weight_bytes + estimated.cache_bytes + estimated.overhead_bytes
        )


class TestWhichMemoryCounts:
    def discrete(self, free=8 * GIB):
        return DeviceMemory(
            device_id="gpu-renderD128",
            total_bytes=free,
            used_bytes=0,
            mapped_total_bytes=45 * GIB,
            mapped_used_bytes=0,
            memory_vendor="samsung",
        )

    def integrated(self):
        return DeviceMemory(
            device_id="gpu-renderD129",
            total_bytes=4 * GIB,
            used_bytes=0,
            mapped_total_bytes=45 * GIB,
            mapped_used_bytes=0,
            memory_vendor="",
        )

    def test_a_discrete_card_is_bounded_by_its_own_memory(self):
        """It maps system memory too, and reports 45 GiB of it. Those bytes are
        across a PCIe bus: counting them would call a model that must stream
        its weights every token a fit."""
        card = self.discrete()

        assert card.integrated is False
        assert card.usable_free_bytes == 8 * GIB

    def test_an_integrated_gpu_is_bounded_by_the_memory_it_may_map(self):
        """Its VRAM is a 4 GiB aperture carved out of system RAM. Reading only
        that reports it as too small for a 4B model it holds comfortably."""
        card = self.integrated()

        assert card.integrated is True
        assert card.usable_free_bytes == 45 * GIB

    def test_the_named_vendor_is_what_tells_them_apart(self):
        """Not the ratio: both report the same GTT. A card with memory
        soldered to it names who made it."""
        assert self.discrete().integrated is False
        assert self.integrated().integrated is True

    def test_what_another_process_holds_is_not_free(self):
        card = DeviceMemory("gpu-renderD128", 8 * GIB, 3 * GIB, memory_vendor="samsung")

        assert card.free_bytes == 5 * GIB


class TestTheVerdict:
    def card(self, free_bytes):
        return DeviceMemory("gpu-renderD128", free_bytes, 0, memory_vendor="samsung")

    def test_the_model_this_desk_runs_fits_the_card_it_runs_on(self):
        answer = verdict(estimate(shape(), 32_768), self.card(8 * GIB))

        assert answer.fits is True
        assert 0 < answer.headroom_bytes < GIB

    def test_a_fourteen_billion_parameter_model_does_not(self):
        """The measured answer to "why not something bigger": at Q4_K_M it is
        over the card before a single cached token."""
        fourteen = shape(file_bytes=int(8.5 * GIB), blocks=40, kv_heads=8)

        answer = verdict(estimate(fourteen, 32_768), self.card(8 * GIB))

        assert answer.fits is False
        assert answer.short_by_bytes > 3 * GIB

    def test_a_card_filled_to_the_last_byte_is_not_called_a_fit(self):
        """It would fail on the first allocation nobody counted."""
        estimated = estimate(shape(), 32_768)
        exactly = self.card(estimated.total_bytes)

        assert verdict(estimated, exactly).fits is False
        assert verdict(estimated, self.card(estimated.total_bytes + DEFAULT_MARGIN_BYTES)).fits

    def test_a_card_that_never_said_its_size_is_answered_unknown(self):
        """Not "no", which would refuse a card that may well hold the model."""
        answer = verdict(
            estimate(shape(), 32_768), DeviceMemory("gpu-renderD128", 0, 0, capacity_known=False)
        )

        assert answer.fits is None
        assert answer.headroom_bytes is None
        assert answer.known is False
        assert answer.short_by_bytes == 0

    def test_a_refusal_never_comes_with_headroom_to_spare(self):
        """fits and headroom are one number: the margin is counted in both."""
        estimated = estimate(shape(), 32_768)
        card = self.card(estimated.total_bytes + DEFAULT_MARGIN_BYTES // 2)

        answer = verdict(estimated, card)

        assert answer.fits is False
        assert answer.headroom_bytes < 0

    def test_a_model_that_does_not_fit_reports_how_short_it_is(self):
        """Because "no" answers nothing a person can act on."""
        answer = verdict(estimate(shape(), 32_768), self.card(4 * GIB))

        assert answer.short_by_bytes > 3 * GIB


class TestReadingRealCards:
    def test_a_render_node_is_read_as_its_driver_publishes_it(self, tmp_path):
        node = tmp_path / "renderD128/device"
        node.mkdir(parents=True)
        (node / "mem_info_vram_total").write_text("8573157376\n")
        (node / "mem_info_vram_used").write_text("117452800\n")
        (node / "mem_info_gtt_total").write_text("48348819456\n")
        (node / "mem_info_gtt_used").write_text("33230848\n")
        (node / "mem_info_vram_vendor").write_text("samsung\n")

        cards = device_memory(tmp_path)

        assert len(cards) == 1
        assert cards[0].device_id == "gpu-renderD128"
        assert cards[0].total_bytes == 8_573_157_376
        assert cards[0].integrated is False

    def test_a_node_that_publishes_no_memory_is_reported_as_unknown(self, tmp_path):
        """Only amdgpu exports mem_info_*. An NVIDIA or Intel node is still a
        card, so it is reported with its capacity marked unknown rather than
        dropped, which would read as a machine with no GPU at all."""
        (tmp_path / "renderD200/device").mkdir(parents=True)

        cards = device_memory(tmp_path)

        assert len(cards) == 1
        assert cards[0].device_id == "gpu-renderD200"
        assert cards[0].capacity_known is False

    def test_a_card_that_reports_only_its_mapped_pool_is_still_read(self, tmp_path):
        node = tmp_path / "renderD128/device"
        node.mkdir(parents=True)
        (node / "mem_info_gtt_total").write_text("48348819456\n")

        cards = device_memory(tmp_path)

        assert cards[0].capacity_known is True
        assert cards[0].mapped_total_bytes == 48_348_819_456

    def test_nothing_raises_when_there_are_no_cards_at_all(self, tmp_path):
        assert device_memory(tmp_path) == ()


class TestReadingAModelsOwnHeader:
    """Loading a model to ask how big it is costs the memory in question, and
    fails for exactly the candidates worth asking about."""

    def build(self, entries: dict) -> bytes:
        out = bytearray(gguf.MAGIC + struct.pack("<IQQ", 3, 0, len(entries)))
        for key, (kind, value) in entries.items():
            out += struct.pack("<Q", len(key)) + key.encode()
            out += struct.pack("<I", kind)
            if kind == 8:  # string
                out += struct.pack("<Q", len(value)) + value.encode()
            else:
                out += struct.pack("<I", value)
        return bytes(out)

    def qwen(self, **overrides) -> dict:
        entries = {
            "general.architecture": (8, "qwen3"),
            "general.name": (8, "Qwen3 8B"),
            "qwen3.block_count": (4, 36),
            "qwen3.attention.head_count": (4, 32),
            "qwen3.attention.head_count_kv": (4, 8),
            "qwen3.embedding_length": (4, 4096),
        }
        entries.update(overrides)
        return entries

    def written(self, tmp_path, entries) -> ModelShape:
        path = tmp_path / "model.gguf"
        path.write_bytes(self.build(entries))
        return gguf.read_shape(path)

    def test_the_shape_comes_from_the_header(self, tmp_path):
        read = self.written(tmp_path, self.qwen())

        assert (read.blocks, read.kv_heads) == (36, 8)
        assert read.architecture == "qwen3"

    def test_a_key_is_as_wide_as_a_head_unless_the_file_says_otherwise(self, tmp_path):
        """4096 embedding over 32 heads. The published default, and the one
        every Qwen export relies on."""
        read = self.written(tmp_path, self.qwen())

        assert (read.key_length, read.value_length) == (128, 128)

    def test_a_file_that_states_its_own_widths_is_believed(self, tmp_path):
        read = self.written(
            tmp_path,
            self.qwen(
                **{
                    "qwen3.attention.key_length": (4, 192),
                    "qwen3.attention.value_length": (4, 64),
                }
            ),
        )

        assert (read.key_length, read.value_length) == (192, 64)

    def test_an_export_without_grouped_heads_uses_its_head_count(self, tmp_path):
        """Older files omit the key, and what they omit is "the same as the
        attention head count"."""
        entries = self.qwen()
        del entries["qwen3.attention.head_count_kv"]

        assert self.written(tmp_path, entries).kv_heads == 32

    def test_weights_are_the_whole_file(self, tmp_path):
        """Nothing here runs a model partly on the CPU, so every byte lands on
        the card."""
        path = tmp_path / "model.gguf"
        path.write_bytes(self.build(self.qwen()) + b"\x00" * 4096)

        assert gguf.read_shape(path).file_bytes == path.stat().st_size

    def test_something_that_is_not_a_model_is_refused_rather_than_guessed(self, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_bytes(b"this is not a model")

        with pytest.raises(GgufError):
            gguf.read_shape(path)

    def test_a_header_that_ends_mid_value_is_refused(self, tmp_path):
        """These bytes came from somewhere else, and a length field is the
        first thing to lie."""
        path = tmp_path / "truncated.gguf"
        path.write_bytes(self.build(self.qwen())[:-6])

        with pytest.raises(GgufError):
            gguf.read_shape(path)

    def test_a_length_that_claims_more_than_the_file_holds_is_refused(self, tmp_path):
        path = tmp_path / "hostile.gguf"
        path.write_bytes(gguf.MAGIC + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", 2**40))

        with pytest.raises(GgufError):
            gguf.read_shape(path)


def test_bytes_are_printed_the_way_a_person_compares_them():
    assert gibibytes(8 * GIB) == "8.00 GiB"


class TestTheTable:
    """What the prober prints, since that is what a person actually reads."""

    def models(self):
        return (("qwen3-8b-q4-k-m", shape()),)

    def cards(self):
        return (
            DeviceMemory("gpu-renderD128", 8 * GIB, 0, 45 * GIB, 0, "samsung"),
            DeviceMemory("gpu-renderD129", 4 * GIB, 0, 45 * GIB, 0, ""),
        )

    def test_one_row_per_model_and_card(self):
        built = probe_cli.rows(self.models(), self.cards(), context_tokens=32_768, cache="q8_0")

        assert len(built) == 2
        assert [row[1] for row in built] == ["gpu-renderD128", "gpu-renderD129"]

    def test_the_row_says_where_the_memory_went(self):
        """Weights and cache separately: "it does not fit" teaches nothing,
        "the cache is 2.39 GiB of it" teaches what to change."""
        built = probe_cli.rows(self.models(), self.cards()[:1], context_tokens=32_768, cache="q8_0")

        assert built[0][2] == "4.68 GiB"
        assert built[0][3] == "2.39 GiB"
        assert built[0][6] == "yes"

    def test_a_mapped_pool_is_labelled_as_mapped(self):
        """So nobody reads 45 GiB on an integrated GPU as 45 GiB of GDDR6."""
        built = probe_cli.rows(self.models(), self.cards()[1:], context_tokens=32_768, cache="q8_0")

        assert "(mapped)" in built[0][5]

    def test_a_shortfall_is_reported_as_the_amount(self):
        built = probe_cli.rows(
            (("qwen3-14b", shape(file_bytes=int(8.5 * GIB))),),
            self.cards()[:1],
            context_tokens=32_768,
            cache="q8_0",
        )

        assert built[0][6] == "no"
        assert built[0][7].startswith("short ")

    def test_the_table_lines_up_under_its_headings(self):
        printed = probe_cli.table(
            probe_cli.rows(self.models(), self.cards(), context_tokens=32_768, cache="q8_0")
        )
        lines = printed.splitlines()

        assert lines[0].split() == list(probe_cli.HEADINGS)
        assert len({len(line.rstrip()) > 0 for line in lines}) == 1

    def test_a_file_that_is_not_a_language_model_is_passed_over(self, tmp_path):
        """A vision projector sits beside the models in the artifact tree, and
        was never a candidate — not a failure to report."""
        (tmp_path / "vision").mkdir()
        (tmp_path / "vision/mmproj.gguf").write_bytes(b"not really")

        assert probe_cli.artifacts(tmp_path) == ()

    def test_the_artifact_id_is_the_directory_the_service_installed_it_as(self, tmp_path):
        """Because that is the id a receipt, a manifest and a policy all use."""
        model = tmp_path / "qwen3-8b-q4-k-m/model.gguf"
        model.parent.mkdir(parents=True)
        model.write_bytes(TestReadingAModelsOwnHeader().build(TestReadingAModelsOwnHeader().qwen()))

        assert [name for name, _shape in probe_cli.artifacts(tmp_path)] == ["qwen3-8b-q4-k-m"]
