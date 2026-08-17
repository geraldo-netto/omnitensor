"""How full the host is, and what the runtime does about it."""

from __future__ import annotations

from omnitensor.host_pressure import (
    CPU_PERCENT_VARIABLE,
    DEFAULT_CPU_PERCENT,
    DEFAULT_MEMORY_PERCENT,
    MEMORY_PERCENT_VARIABLE,
    bounded_percent,
    configured_limits,
    cpu_percent,
    memory_percent,
    read_pressure,
)


def meminfo(tmp_path, total_kb, available_kb, swap_total_kb=0, swap_free_kb=0):
    path = tmp_path / "meminfo"
    path.write_text(
        f"MemTotal:       {total_kb} kB\n"
        f"MemFree:        {available_kb} kB\n"
        f"MemAvailable:   {available_kb} kB\n"
        f"SwapTotal:      {swap_total_kb} kB\n"
        f"SwapFree:       {swap_free_kb} kB\n",
        encoding="ascii",
    )
    return path


def loadavg(tmp_path, value):
    path = tmp_path / "loadavg"
    path.write_text(f"{value} 2.00 1.50 3/1234 5678\n", encoding="ascii")
    return path


class TestMemory:
    def test_an_idle_machine_reads_as_almost_empty(self, tmp_path):
        assert memory_percent(meminfo(tmp_path, 1_000_000, 900_000)) == 10.0

    def test_reclaimable_cache_counts_as_available_not_as_pressure(self, tmp_path):
        """MemAvailable, not MemFree: any machine that has been up a while has
        little free memory and plenty it can have back."""
        path = tmp_path / "meminfo"
        path.write_text(
            "MemTotal:       1000000 kB\n"
            "MemFree:          50000 kB\n"
            "MemAvailable:    800000 kB\n"
            "SwapTotal:            0 kB\n"
            "SwapFree:             0 kB\n",
            encoding="ascii",
        )

        assert memory_percent(path) == 20.0

    def test_swap_is_part_of_the_same_budget(self, tmp_path):
        # Half the swap gone on a half-used machine is more pressure than the
        # RAM figure alone would say.
        path = meminfo(tmp_path, 1_000_000, 500_000, 1_000_000, 500_000)

        assert memory_percent(path) == 50.0

    def test_an_unreadable_proc_reads_as_no_pressure(self, tmp_path):
        # Better than treating it as a full machine, which would stop every job
        # on a host whose /proc is not what this expects.
        assert memory_percent(tmp_path / "missing") == 0.0

    def test_a_machine_with_no_memory_at_all_does_not_divide_by_zero(self, tmp_path):
        path = tmp_path / "meminfo"
        path.write_text("MemTotal: 0 kB\nMemAvailable: 0 kB\n", encoding="ascii")

        assert memory_percent(path) == 0.0


class TestCpu:
    def test_load_is_read_against_the_cores_available(self, tmp_path):
        assert cpu_percent(loadavg(tmp_path, "8.00"), cores=16) == 50.0
        assert cpu_percent(loadavg(tmp_path, "32.00"), cores=16) == 100.0

    def test_an_unreadable_loadavg_reads_as_idle(self, tmp_path):
        assert cpu_percent(tmp_path / "missing", cores=8) == 0.0


class TestThresholds:
    def test_both_thresholds_default_to_eighty_percent(self):
        assert configured_limits({}) == (DEFAULT_MEMORY_PERCENT, DEFAULT_CPU_PERCENT)
        assert DEFAULT_MEMORY_PERCENT == 80.0
        assert DEFAULT_CPU_PERCENT == 80.0

    def test_each_threshold_is_configurable_on_its_own(self):
        limits = configured_limits({MEMORY_PERCENT_VARIABLE: "65", CPU_PERCENT_VARIABLE: "95"})

        assert limits == (65.0, 95.0)

    def test_a_threshold_nobody_could_satisfy_is_the_default(self):
        """A typo in a tuning knob must not take the runtime down with it."""
        for value in ("", "soon", "0", "-5", "140", "nan"):
            assert bounded_percent(value, DEFAULT_MEMORY_PERCENT) == DEFAULT_MEMORY_PERCENT


class TestReading:
    def test_a_full_machine_is_saturated_and_says_by_how_much(self, tmp_path):
        reading = read_pressure(
            environ={},
            meminfo=meminfo(tmp_path, 1_000_000, 100_000),
            loadavg=loadavg(tmp_path, "1.00"),
            cores=32,
        )

        assert reading.memory_percent == 90.0
        assert reading.memory_saturated is True
        assert reading.saturated is True
        assert "memory 90% of 80%" in reading.detail()

    def test_a_busy_cpu_alone_does_not_hold_work(self, tmp_path):
        """A pegged CPU is usually a job doing exactly what it was asked to."""
        reading = read_pressure(
            environ={},
            meminfo=meminfo(tmp_path, 1_000_000, 900_000),
            loadavg=loadavg(tmp_path, "64.00"),
            cores=32,
        )

        assert reading.cpu_saturated is True
        assert reading.memory_saturated is False
        assert reading.saturated is False

    def test_the_reading_carries_the_thresholds_it_was_judged_against(self, tmp_path):
        reading = read_pressure(
            environ={MEMORY_PERCENT_VARIABLE: "50"},
            meminfo=meminfo(tmp_path, 1_000_000, 400_000),
            loadavg=loadavg(tmp_path, "1.00"),
            cores=8,
        )

        assert reading.memory_limit == 50.0
        assert reading.saturated is True
