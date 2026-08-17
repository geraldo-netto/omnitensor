from __future__ import annotations

import asyncio

import pytest

import omnitensor.sdk as sdk


def context():
    return sdk.PluginContext("fixture-plugin", 1, {}, frozenset())


def request():
    return sdk.PluginRequest("job-1", "fixture-plugin", "manual", {}, 0, 100)


class ConformingPlugin:
    def __init__(self, *, cancellation=False, health=None, result=None, execute_error=None):
        self.cancellation = cancellation
        self.health_value = health
        self.result_value = result
        self.execute_error = execute_error
        self.events = []

    async def start(self, plugin_context):
        self.events.append(("start", plugin_context.plugin_id))

    async def health(self):
        self.events.append(("health",))
        return self.health_value or sdk.PluginHealth(
            sdk.PluginHealthStatus.READY,
            "ready",
            1,
        )

    async def execute(self, plugin_request, cancellation, progress):
        self.events.append(("execute", plugin_request.job_id))
        if self.execute_error is not None:
            raise self.execute_error
        if self.cancellation:
            assert cancellation.cancelled
            return sdk.cancelled_result(
                plugin_request,
                "cancelled",
                completed_at_ms=2,
            )
        await progress.report(sdk.PluginProgress("job-1", "collect", 0.25, "", 1))
        await progress.report(sdk.PluginProgress("job-1", "deliver", 1.0, "", 2))
        return self.result_value or sdk.succeeded_result(
            plugin_request,
            {"ok": True},
            completed_at_ms=2,
        )

    async def stop(self):
        self.events.append(("stop",))


def test_normal_and_cancellation_contracts_validate_and_stop_plugins():
    async def scenario():
        plugin = ConformingPlugin()
        report = await sdk.run_plugin_contract(plugin, context(), request())

        assert report.health.status is sdk.PluginHealthStatus.READY
        assert report.result.status is sdk.PluginResultStatus.SUCCEEDED
        assert [item.fraction for item in report.progress] == [0.25, 1.0]
        assert plugin.events == [
            ("start", "fixture-plugin"),
            ("health",),
            ("execute", "job-1"),
            ("stop",),
        ]

        cancelled_plugin = ConformingPlugin(cancellation=True)
        cancelled = await sdk.run_cancellation_contract(
            cancelled_plugin,
            context(),
            request(),
        )
        assert cancelled.result.status is sdk.PluginResultStatus.CANCELLED
        assert cancelled.progress == ()
        assert cancelled_plugin.events[-1] == ("stop",)

    asyncio.run(scenario())


def test_contract_always_stops_after_execute_failure():
    async def scenario():
        plugin = ConformingPlugin(execute_error=RuntimeError("fixture"))
        with pytest.raises(RuntimeError, match="fixture"):
            await sdk.run_plugin_contract(plugin, context(), request())
        assert plugin.events[-1] == ("stop",)

    asyncio.run(scenario())


def test_contract_rejects_missing_surface_and_identity_mismatch():
    async def scenario():
        with pytest.raises(sdk.ConformanceError, match="does not implement"):
            await sdk.run_plugin_contract(object(), context(), request())
        wrong = sdk.PluginRequest("job", "other", "manual", {}, 0, None)
        with pytest.raises(sdk.ConformanceError, match="identities differ"):
            await sdk.run_plugin_contract(ConformingPlugin(), context(), wrong)
        with pytest.raises(sdk.ConformanceError, match="does not implement"):
            await sdk.run_cancellation_contract(object(), context(), request())

    asyncio.run(scenario())


@pytest.mark.parametrize("timeout", [0, -1, True, "1"])
def test_contract_timeout_must_be_positive(timeout):
    async def scenario():
        with pytest.raises(ValueError, match="timeout must be positive"):
            await sdk.run_plugin_contract(ConformingPlugin(), context(), request(), timeout=timeout)
        with pytest.raises(ValueError, match="timeout must be positive"):
            await sdk.run_cancellation_contract(
                ConformingPlugin(cancellation=True),
                context(),
                request(),
                timeout=timeout,
            )

    asyncio.run(scenario())


def test_cancellation_contract_requires_cancelled_terminal_status():
    async def scenario():
        with pytest.raises(sdk.ConformanceError, match="did not return cancelled"):
            await sdk.run_cancellation_contract(
                ConformingPlugin(),
                context(),
                request(),
            )

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("health", "message"),
    [
        ("ready", "did not return PluginHealth"),
        (sdk.PluginHealth(sdk.PluginHealthStatus.READY, None, 0), "detail is not"),
        (sdk.PluginHealth(sdk.PluginHealthStatus.READY, "ready", -1), "timestamp is not"),
        (sdk.PluginHealth(sdk.PluginHealthStatus.READY, "ready", True), "timestamp is not"),
    ],
)
def test_contract_rejects_invalid_health(health, message):
    async def scenario():
        plugin = ConformingPlugin(health=health)
        with pytest.raises(sdk.ConformanceError, match=message):
            await sdk.run_plugin_contract(plugin, context(), request())
        assert plugin.events[-1] == ("stop",)

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("result", "message"),
    [
        ("done", "did not return PluginResult"),
        (
            sdk.PluginResult("other", sdk.PluginResultStatus.SUCCEEDED, {}, "", 1),
            "job identity differs",
        ),
        (sdk.PluginResult("job-1", "success", {}, "", 1), "status is invalid"),
        (
            sdk.PluginResult("job-1", sdk.PluginResultStatus.SUCCEEDED, [], "", 1),
            "output is not",
        ),
        (
            sdk.PluginResult("job-1", sdk.PluginResultStatus.SUCCEEDED, {}, None, 1),
            "detail is not",
        ),
        (
            sdk.PluginResult("job-1", sdk.PluginResultStatus.SUCCEEDED, {}, "", -1),
            "timestamp is not",
        ),
    ],
)
def test_contract_rejects_invalid_terminal_results(result, message):
    async def scenario():
        plugin = ConformingPlugin(result=result)
        with pytest.raises(sdk.ConformanceError, match=message):
            await sdk.run_plugin_contract(plugin, context(), request())

    asyncio.run(scenario())


def test_progress_probe_enforces_identity_order_bounds_and_type():
    async def scenario():
        probe = sdk.ProgressProbe("job", limit=1)
        with pytest.raises(sdk.ConformanceError, match="non-PluginProgress"):
            await probe.report(object())
        with pytest.raises(sdk.ConformanceError, match="job identity"):
            await probe.report(sdk.PluginProgress("other", "stage", 0.0, "", 0))
        await probe.report(sdk.PluginProgress("job", "stage", 0.5, "", 0))
        with pytest.raises(sdk.ConformanceError, match="moved backward"):
            await probe.report(sdk.PluginProgress("job", "stage", 0.4, "", 0))
        with pytest.raises(sdk.ConformanceError, match="more than 1"):
            await probe.report(sdk.PluginProgress("job", "stage", 0.6, "", 0))

    asyncio.run(scenario())
    with pytest.raises(ValueError, match="job_id must be"):
        sdk.ProgressProbe("")
    with pytest.raises(ValueError, match="limit must be"):
        sdk.ProgressProbe("job", limit=0)


@pytest.mark.parametrize("fraction", [-1, 1.1, float("nan"), float("inf"), True, "1"])
def test_progress_probe_rejects_invalid_fractions(fraction):
    async def scenario():
        probe = sdk.ProgressProbe("job")
        with pytest.raises(sdk.ConformanceError, match="fraction is invalid"):
            await probe.report(sdk.PluginProgress("job", "stage", fraction, "", 0))

    asyncio.run(scenario())


def test_failure_injector_raises_configured_number_of_times():
    injector = sdk.FailureInjector({sdk.FailurePoint.INFER: 2})

    for remaining in (1, 0):
        with pytest.raises(sdk.InjectedFailureError) as excinfo:
            injector.hit(sdk.FailurePoint.INFER)
        assert excinfo.value.point is sdk.FailurePoint.INFER
        assert str(excinfo.value) == "injected failure at infer"
        assert injector.remaining(sdk.FailurePoint.INFER) == remaining
    injector.hit(sdk.FailurePoint.INFER)
    injector.hit(sdk.FailurePoint.DELIVER)
    assert injector.hits == [
        sdk.FailurePoint.INFER,
        sdk.FailurePoint.INFER,
        sdk.FailurePoint.INFER,
        sdk.FailurePoint.DELIVER,
    ]


def test_failure_injector_validates_points_and_counts():
    with pytest.raises(TypeError, match="failure point"):
        sdk.FailureInjector({"infer": 1})
    with pytest.raises(ValueError, match="failure count"):
        sdk.FailureInjector({sdk.FailurePoint.INFER: 0})
    injector = sdk.FailureInjector({})
    with pytest.raises(TypeError, match="failure point"):
        injector.hit("infer")
    with pytest.raises(TypeError, match="failure point"):
        injector.remaining("infer")


def test_bounded_fuzz_corpus_is_reproducible_and_respects_limits():
    first = sdk.bounded_json_payloads(
        seed=42,
        cases=20,
        max_depth=3,
        max_width=4,
        max_string_chars=12,
    )
    second = sdk.bounded_json_payloads(
        seed=42,
        cases=20,
        max_depth=3,
        max_width=4,
        max_string_chars=12,
    )

    assert first == second
    assert len(first) == 20
    assert [payload["case"] for payload in first] == list(range(20))
    for payload in first:
        sdk.assert_json_bounds(
            payload,
            max_depth=3,
            max_width=4,
            max_string_chars=12,
        )
    assert sdk.bounded_json_payloads(seed=43, cases=20) != sdk.bounded_json_payloads(
        seed=42, cases=20
    )


@pytest.mark.parametrize("seed", [True, 1.5, "1"])
def test_fuzz_seed_must_be_an_integer(seed):
    with pytest.raises(ValueError, match="seed must be an integer"):
        sdk.bounded_json_payloads(seed=seed)


@pytest.mark.parametrize("name", ["cases", "max_depth", "max_width", "max_string_chars"])
def test_fuzz_limits_must_be_positive(name):
    arguments = {name: 0}
    with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
        sdk.bounded_json_payloads(**arguments)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([[[]]], "maximum depth"),
        ([1, 2], "array exceeds"),
        ({"a": 1, "b": 2}, "object exceeds"),
        ("long", "string exceeds"),
        ({1: "value"}, "key is not"),
        ({"long": 1}, "key exceeds"),
        ({"v": object()}, "not JSON compatible"),
        ({"v": float("nan")}, "not finite"),
    ],
)
def test_json_bound_assertions_reject_unsafe_values(value, message):
    with pytest.raises(sdk.ConformanceError, match=message):
        sdk.assert_json_bounds(
            value,
            max_depth=1,
            max_width=1,
            max_string_chars=3,
        )


@pytest.mark.parametrize("name", ["max_depth", "max_width", "max_string_chars"])
def test_json_bound_limits_must_be_positive(name):
    arguments = {"max_depth": 1, "max_width": 1, "max_string_chars": 1}
    arguments[name] = 0
    with pytest.raises(ValueError, match=f"{name} must be a positive integer"):
        sdk.assert_json_bounds({}, **arguments)


def test_changed_function_selector_chooses_narrowest_functions_and_decorators():
    source = """\
def top():
    value = 1
    def nested():
        return value
    return nested()

class Worker:
    @decorator
    async def run(self):
        return 2
"""

    targets = sdk.changed_function_targets(source, {1, 2, 4, 8, 10})

    assert targets == (
        sdk.ChangedFunction("top", 1, 5),
        sdk.ChangedFunction("top.nested", 3, 4),
        sdk.ChangedFunction("Worker.run", 8, 10),
    )
    assert sdk.mutmut_function_patterns("package.module", targets) == (
        "package.module.top*",
        "package.module.top.nested*",
        "package.module.Worker.run*",
    )
    assert sdk.changed_function_targets(source, {7}) == ()


def test_changed_function_selector_validates_inputs_and_syntax():
    with pytest.raises(TypeError, match="source must be a string"):
        sdk.changed_function_targets(None, {1})
    with pytest.raises(ValueError, match="positive integers"):
        sdk.changed_function_targets("x = 1", {0})
    with pytest.raises(SyntaxError):
        sdk.changed_function_targets("def broken(", {1})
    with pytest.raises(ValueError, match="module must be a non-empty string"):
        sdk.mutmut_function_patterns("", ())
