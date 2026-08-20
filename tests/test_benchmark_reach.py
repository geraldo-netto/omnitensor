"""The measuring command, on a machine that is not a checkout.

`bench_cli` is the one command that can say whether a workload still answers
its labelled cases correctly. Two things stopped it being usable where it
matters. Its corpus lived only in the source tree, so an installed host was
told "no case file" and named a path under `site-packages` that was never
going to exist. And with no `--device` it measured on Vulkan index 0 — the
integrated card on this desk, and on most laptops — producing numbers that
cannot be compared with the ones they replace, without saying so.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

from omnitensor.benchmark import cases as case_files
from omnitensor.benchmark.vulkan_devices import DeviceError, VulkanDevice, preferred, select

ROOT = Path(__file__).parents[1]
SOURCE_CASES = ROOT / "benchmarks" / "cases"
WORKLOADS = ("ask-selected-files", "event-extraction", "file-organizer", "selected-text-tools")


def test_the_wheel_carries_the_cases_it_reads():
    """One copy in the repository, force-included at build time.

    Copying them into `src/` would be a second corpus to keep in step, which
    is the arrangement the schemas and the manifests already avoid.
    """
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    included = project["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]

    assert included["benchmarks/cases"] == "omnitensor/benchmark/cases"
    assert case_files.PACKAGED_CASE_ROOT.name == "cases"
    assert case_files.PACKAGED_CASE_ROOT.parent.name == "benchmark"


def test_the_source_tree_wins_while_it_is_there():
    """It is the copy somebody edits."""
    assert case_files.case_root({}) == SOURCE_CASES


def test_a_configured_corpus_outranks_both(tmp_path):
    assert case_files.case_root({case_files.BENCHMARK_ROOT_VARIABLE: str(tmp_path)}) == (
        tmp_path / "cases"
    )


def test_an_installed_host_falls_back_to_the_packaged_copy(monkeypatch, tmp_path):
    """No checkout, no environment variable, and the cases are still found."""
    monkeypatch.setattr(case_files, "SOURCE_BENCHMARK_ROOT", tmp_path / "absent")

    assert case_files.case_root({}) == case_files.PACKAGED_CASE_ROOT


def test_a_missing_case_file_names_the_way_out(tmp_path):
    with pytest.raises(case_files.CaseError) as refusal:
        case_files.load("ask-selected-files", tmp_path)

    assert case_files.BENCHMARK_ROOT_VARIABLE in str(refusal.value)


@pytest.mark.parametrize("workload_id", WORKLOADS)
def test_every_workload_with_cases_still_loads_them(workload_id):
    loaded = case_files.load(workload_id)

    assert loaded, workload_id
    document = json.loads((SOURCE_CASES / f"{workload_id}.json").read_text(encoding="utf-8"))
    assert len(loaded) == len(document["cases"])


class TestWhichCardIsMeasuredOn:
    LAPTOP = (
        VulkanDevice(0, "AMD Radeon 610M", True),
        VulkanDevice(1, "AMD Radeon RX 6600 XT", False),
        VulkanDevice(2, "llvmpipe", True),
    )

    def test_the_discrete_card_is_what_nobody_naming_one_means(self):
        assert preferred(self.LAPTOP) == self.LAPTOP[1]

    def test_index_zero_is_not_the_answer_just_for_being_first(self):
        """It is the integrated card on this desk, and on most laptops."""
        assert preferred(self.LAPTOP).index != 0

    def test_with_nothing_discrete_the_first_card_is_as_good_as_there_is(self):
        integrated = (VulkanDevice(0, "610M", True), VulkanDevice(1, "llvmpipe", True))

        assert preferred(integrated) == integrated[0]

    def test_naming_a_card_still_picks_exactly_that_one(self):
        """A bare number is an index — `6600` is not the RX 6600 XT."""
        assert select("0", self.LAPTOP) == self.LAPTOP[0]
        assert select("RX 6600", self.LAPTOP) == self.LAPTOP[1]

    def test_no_device_at_all_is_a_refusal_rather_than_a_choice(self):
        with pytest.raises(DeviceError, match="no Vulkan device answered"):
            preferred(())
