"""Per-plugin call deadlines, derived from what a manifest declares."""

from omnitensor.plugins.budgets import DEFAULT_CALL_TIMEOUT_SECONDS, MAX_CALL_TIMEOUT_SECONDS
from omnitensor.plugins.worker_budgets import (
    FLOW_MARGIN_SECONDS,
    GENERATIVE_CALL_TIMEOUT_SECONDS,
    call_timeout_for,
    flow_deadline_for,
    limits_for,
)


def manifest(**plugin) -> dict:
    return {"plugin": plugin}


class TestDerivedDeadline:
    def test_a_language_model_gets_time_to_load_and_answer(self):
        """The failure this exists for: 30s went entirely on loading an 8B model."""
        document = manifest(artifacts=[{"id": "qwen3-8b-q4-k-m", "format": "gguf"}])
        assert call_timeout_for(document) == GENERATIVE_CALL_TIMEOUT_SECONDS

    def test_a_plugin_that_loads_no_model_keeps_the_short_default(self):
        document = manifest(artifacts=[{"id": "classifier", "format": "ncnn"}])
        assert call_timeout_for(document) == DEFAULT_CALL_TIMEOUT_SECONDS

    def test_a_manifest_declaring_nothing_keeps_the_short_default(self):
        assert call_timeout_for({}) == DEFAULT_CALL_TIMEOUT_SECONDS
        assert call_timeout_for(manifest()) == DEFAULT_CALL_TIMEOUT_SECONDS
        assert call_timeout_for(manifest(artifacts="not a list")) == DEFAULT_CALL_TIMEOUT_SECONDS

    def test_the_format_is_read_whatever_its_case(self):
        assert call_timeout_for(manifest(artifacts=[{"format": "GGUF"}])) == (
            GENERATIVE_CALL_TIMEOUT_SECONDS
        )

    def test_one_generative_artifact_among_others_is_enough(self):
        document = manifest(artifacts=[{"format": "ncnn"}, {"format": "gguf"}])
        assert call_timeout_for(document) == GENERATIVE_CALL_TIMEOUT_SECONDS


class TestDeclaredBudget:
    def test_a_plugin_that_states_a_budget_is_believed(self):
        document = manifest(budgets={"callTimeoutSeconds": 90})
        assert call_timeout_for(document) == 90

    def test_a_declared_budget_outranks_the_derived_one(self):
        document = manifest(artifacts=[{"format": "gguf"}], budgets={"callTimeoutSeconds": 45})
        assert call_timeout_for(document) == 45

    def test_an_unusable_budget_falls_back_rather_than_removing_the_bound(self):
        for value in (0, -1, True, "60", None, MAX_CALL_TIMEOUT_SECONDS + 1):
            document = manifest(budgets={"callTimeoutSeconds": value})
            assert call_timeout_for(document) == DEFAULT_CALL_TIMEOUT_SECONDS


class TestFlowDeadline:
    def test_the_flow_waits_longer_so_the_worker_reports_first(self):
        document = manifest(artifacts=[{"format": "gguf"}])
        assert flow_deadline_for(document) == (
            GENERATIVE_CALL_TIMEOUT_SECONDS + FLOW_MARGIN_SECONDS
        )

    def test_the_margin_survives_even_the_maximum_declaration(self):
        """Clamping the sum made both clocks equal at exactly the maximum
        declaration, recreating the equal-clocks race the margin exists to
        prevent (OMNI-0596). The ceiling governs declarations; this derived
        wait always exceeds the worker budget it wraps."""
        document = manifest(budgets={"callTimeoutSeconds": MAX_CALL_TIMEOUT_SECONDS})

        assert flow_deadline_for(document) == MAX_CALL_TIMEOUT_SECONDS + 5.0


class TestLimits:
    def test_only_the_deadline_moves(self):
        limits = limits_for(manifest(artifacts=[{"format": "gguf"}]))
        assert limits.call_timeout_seconds == GENERATIVE_CALL_TIMEOUT_SECONDS
        assert limits.max_concurrency == 1
