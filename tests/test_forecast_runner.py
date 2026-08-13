from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import write_workload
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.contract import build_contract_document
from omnitensor.plugins.recorder import FeatureRow, TelemetryRecorder
from omnitensor.registry import Workload
from omnitensor.training.cli import forecast_main
from omnitensor.training.runner import (
    MAX_WIRE_BYTES,
    REQUIRED_METHODS,
    DbusForecastClient,
    ForecastRunError,
    TrustedForecastRunner,
    _load_catalog,
    _runtime_job_schema_versions,
    _validate_binding_models,
    latest_forecast_payload,
    load_forecast_binding,
    wire_document,
)

PROFILE = "resource-scheduler"
DIGEST = "a" * 64
_UNSET = object()


def _base_manifest() -> dict:
    return json.loads(
        (Path(__file__).parents[1] / "workloads/resource-scheduler/manifest.json").read_text()
    )


def _feature_contract(**changes) -> dict:
    contract = {
        "version": 1,
        "recipe": "forecast-v1",
        "featureNames": ["load", "queue"],
        "targetFeature": "load",
        "window": 2,
        "horizon": 1,
        "observationOrder": "oldest-first",
        "flattenOrder": "observations-then-features",
    }
    contract.update(changes)
    return contract


def _model(model_format="ncnn", **changes) -> dict:
    model = {
        "id": f"local-forecast-{model_format}",
        "version": "1.0.0",
        "format": model_format,
        "fullyQuantized": False,
        "minimumCompilerVersion": "0.0.0",
        "minimumRuntimeVersion": "0.0.0",
        "sha256": DIGEST,
        "tensorContract": {"inputs": [{"shape": [1, 4], "dtype": "float32", "layout": "NC"}]},
        "featureContract": _feature_contract(),
        "outputContract": {"kind": "raw"},
    }
    if model_format in {"ncnn", "openvino"}:
        model["companions"] = {
            "model.bin": "b" * 64,
        }
    model.update(changes)
    return model


def _binding_manifest(*models: dict) -> dict:
    manifest = _base_manifest()
    requirements = manifest["requirements"]
    requirements.pop("model")
    requirements["accelerator"] = "gpu"
    requirements["acceleratorPreference"] = ["gpu"]
    requirements["models"] = list(models or (_model(),))
    return manifest


def _binding_roots(tmp_path, manifest=None):
    bundled = tmp_path / "bundled"
    bindings = tmp_path / "bindings"
    write_workload(bundled, _base_manifest())
    if manifest is not False:
        write_workload(bindings, manifest or _binding_manifest())
    return bundled, bindings


def _workload(window=2, features=("load", "queue")) -> Workload:
    contract = _feature_contract(
        featureNames=list(features), targetFeature=features[0], window=window
    )
    model = _model(
        featureContract=contract,
        tensorContract={
            "inputs": [{"shape": [1, len(features) * window], "dtype": "float32", "layout": "NC"}]
        },
    )
    return Workload(PROFILE, _binding_manifest(model))


def _rows(count=3):
    return tuple(
        FeatureRow(PROFILE, index + 1, {"load": index + 0.5, "queue": index + 10})
        for index in range(count)
    )


def _contract(**changes):
    document = build_contract_document(REQUIRED_METHODS)
    document.update(changes)
    return json.dumps(document)


def _ack(request_id, *, status="accepted", job_id="job-1", code="job-accepted"):
    return json.dumps(
        {
            "version": 1,
            "requestId": request_id,
            "jobId": job_id,
            "status": status,
            "code": code,
            "message": "reply detail",
            "timestamp": 1,
        }
    )


def _result(request_id, *, state="succeeded", job_id="job-1", output=_UNSET, code=None):
    default_output = {
        "outputs": [[[0.75]]],
        "reading": {
            "kind": "forecast",
            "targetFeature": "load",
            "horizon": 1,
            "value": 0.75,
        },
    }
    return json.dumps(
        {
            "version": 1,
            "requestId": request_id,
            "jobId": job_id,
            "state": state,
            "code": code or f"job-{state}",
            "message": "result detail",
            "timestamp": 2,
            "progress": None,
            "output": default_output if output is _UNSET else output,
        }
    )


class FakeClient:
    def __init__(self, *, contract=None, acknowledgement=None, results=None):
        self.contract = contract or _contract()
        self.acknowledgement = acknowledgement
        self.results = list(results or [])
        self.submissions = []
        self.polls = []
        self.closed = False

    async def describe_contract(self):
        return self.contract

    async def submit_job(self, request):
        document = json.loads(request)
        self.submissions.append(document)
        return self.acknowledgement or _ack(document["requestId"])

    async def get_job_result(self, request):
        document = json.loads(request)
        self.polls.append(document)
        if self.results:
            reply = self.results.pop(0)
            return reply(document) if callable(reply) else reply
        return _result(document["requestId"], job_id=document["jobId"])

    def close(self):
        self.closed = True


def test_binding_loader_accepts_only_pinned_restricted_forecast_overlay(tmp_path):
    bundled, bindings = _binding_roots(tmp_path)

    workload = load_forecast_binding(PROFILE, bindings, bundled_root=bundled)

    assert workload.id == PROFILE
    assert workload.model["sha256"] == DIGEST
    assert workload.model["featureContract"] == _feature_contract()


@pytest.mark.parametrize(
    ("profile", "manifest", "code"),
    [
        ("unknown-profile", _binding_manifest(), "profile-unknown"),
        (PROFILE, False, "binding-unavailable"),
        (PROFILE, _binding_manifest(_model(sha256=None)), "binding-invalid"),
        (PROFILE, _binding_manifest(_model(featureContract=None)), "binding-invalid"),
    ],
)
def test_binding_loader_refuses_missing_or_invalid_state(tmp_path, profile, manifest, code):
    bundled, bindings = _binding_roots(tmp_path, manifest)

    with pytest.raises(ForecastRunError) as captured:
        load_forecast_binding(profile, bindings, bundled_root=bundled)

    assert captured.value.code == code


def test_binding_loader_refuses_policy_changes(tmp_path):
    manifest = _binding_manifest()
    manifest["ui"]["title"] = "Impostor"
    bundled, bindings = _binding_roots(tmp_path, manifest)

    with pytest.raises(ForecastRunError) as captured:
        load_forecast_binding(PROFILE, bindings, bundled_root=bundled)

    assert captured.value.code == "binding-untrusted"
    assert captured.value.detail == "binding changes bundled profile policy"


def test_binding_loader_defends_against_unvalidated_contract_drift(monkeypatch, tmp_path):
    first = _model()
    second = _model("openvino", featureContract=_feature_contract(horizon=2))
    base = Workload(PROFILE, _base_manifest())
    binding = Workload(PROFILE, _binding_manifest(first, second))
    monkeypatch.setattr(
        "omnitensor.training.runner.load_workloads",
        lambda root: {PROFILE: base if root == tmp_path / "bundled" else binding},
    )

    with pytest.raises(ForecastRunError) as captured:
        load_forecast_binding(PROFILE, tmp_path / "bindings", bundled_root=tmp_path / "bundled")

    assert captured.value.code == "feature-contract-mismatch"


def test_binding_loader_requires_digest_after_schema_validation(monkeypatch, tmp_path):
    model = _model()
    model.pop("sha256")
    base = Workload(PROFILE, _base_manifest())
    binding = Workload(PROFILE, _binding_manifest(model))
    monkeypatch.setattr(
        "omnitensor.training.runner.load_workloads",
        lambda root: {PROFILE: base if root == tmp_path / "bundled" else binding},
    )

    with pytest.raises(ForecastRunError) as captured:
        load_forecast_binding(PROFILE, tmp_path / "bindings", bundled_root=tmp_path / "bundled")

    assert captured.value.code == "model-unpinned"
    assert captured.value.detail == "ncnn model has no digest"


def test_binding_catalog_failure_preserves_diagnostic(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "omnitensor.training.runner.load_workloads",
        lambda _root: (_ for _ in ()).throw(ValueError("broken catalog")),
    )

    with pytest.raises(ForecastRunError) as captured:
        _load_catalog(tmp_path)

    assert captured.value.code == "binding-invalid"
    assert captured.value.detail == "broken catalog"


@pytest.mark.parametrize(
    ("models", "code"),
    [
        ((), "binding-invalid"),
        ((_model(featureContract=None),), "feature-contract-missing"),
    ],
)
def test_binding_model_defense_rejects_unvalidated_missing_contracts(models, code):
    manifest = _binding_manifest()
    manifest["requirements"]["models"] = list(models)
    with pytest.raises(ForecastRunError) as captured:
        _validate_binding_models(Workload(PROFILE, manifest))

    assert captured.value.code == code
    assert (
        captured.value.detail
        == {
            "binding-invalid": "binding declares no model",
            "feature-contract-missing": "every model must declare featureContract",
        }[code]
    )


def test_binding_model_defense_preserves_contract_mismatch_detail():
    models = (_model(), _model("openvino", featureContract=_feature_contract(horizon=2)))

    with pytest.raises(ForecastRunError) as captured:
        _validate_binding_models(Workload(PROFILE, _binding_manifest(*models)))

    assert captured.value.code == "feature-contract-mismatch"
    assert captured.value.detail == "model feature contracts disagree"


def test_latest_payload_uses_only_exact_newest_window_and_declared_order():
    payload = latest_forecast_payload(_workload(), _rows())

    assert payload == {"inputs": [[[1.5, 11.0, 2.5, 12.0]]]}


@pytest.mark.parametrize(
    ("rows", "code"),
    [
        (_rows(1), "insufficient-history"),
        (
            (_rows(1)[0], FeatureRow("other", 2, {"load": 1.0, "queue": 2.0})),
            "history-profile-mismatch",
        ),
        (
            (_rows(1)[0], FeatureRow(PROFILE, 1, {"load": 1.0, "queue": 2.0})),
            "observations-unordered",
        ),
        ((_rows(1)[0], FeatureRow(PROFILE, 2, {"load": 1.0})), "features-missing"),
        ((_rows(1)[0], FeatureRow(PROFILE, 2, {"load": True, "queue": 2.0})), "features-missing"),
        (
            (_rows(1)[0], FeatureRow(PROFILE, 2, {"load": float("inf"), "queue": 2.0})),
            "features-missing",
        ),
    ],
)
def test_latest_payload_refuses_untrusted_history(rows, code):
    with pytest.raises(ForecastRunError) as captured:
        latest_forecast_payload(_workload(), rows)

    assert captured.value.code == code


@pytest.mark.parametrize(
    ("rows", "detail"),
    [
        (
            (FeatureRow("other", 0, {"load": 1.0, "queue": 2.0}),),
            "history contains another profile",
        ),
        (
            (FeatureRow(PROFILE, -1, {"load": 1.0, "queue": 2.0}),),
            "history timestamps must increase strictly",
        ),
    ],
)
def test_latest_payload_history_failures_have_exact_details(rows, detail):
    with pytest.raises(ForecastRunError) as captured:
        latest_forecast_payload(_workload(window=1), rows)

    assert captured.value.detail == detail


def test_latest_payload_requires_semantic_contract():
    manifest = _binding_manifest(_model(featureContract=None))

    with pytest.raises(ForecastRunError) as captured:
        latest_forecast_payload(Workload(PROFILE, manifest), _rows())

    assert captured.value.code == "feature-contract-missing"


@given(
    st.lists(
        st.text(alphabet="abcde", min_size=1, max_size=4),
        min_size=1,
        max_size=5,
        unique=True,
    ),
    st.integers(min_value=1, max_value=5),
)
def test_latest_payload_property_preserves_observation_then_feature_order(names, window):
    workload = _workload(window, tuple(names))
    rows = tuple(
        FeatureRow(
            PROFILE, index + 1, {name: index * 10 + offset for offset, name in enumerate(names)}
        )
        for index in range(window + 2)
    )

    payload = latest_forecast_payload(workload, rows)

    assert payload["inputs"][0][0] == [
        float(index * 10 + offset) for index in range(2, window + 2) for offset in range(len(names))
    ]


@pytest.mark.parametrize(
    ("text", "detail"),
    [
        (None, "runtime reply is not text"),
        ("\ud800", "runtime reply is not UTF-8"),
        ("not json", "runtime reply is not JSON"),
        ("[]", "/: [] is not of type 'object'"),
        (
            json.dumps({"version": 1}),
            "/: 'methods' is a required property; /: 'schemas' is a required property",
        ),
        ('{"padding":"' + "x" * MAX_WIRE_BYTES + '"}', "runtime reply exceeds 1 MiB"),
    ],
)
def test_wire_document_refuses_malformed_or_oversized_replies(text, detail):
    with pytest.raises(ForecastRunError) as captured:
        wire_document(text, "runtime-contract.schema.json")

    assert captured.value.code == "runtime-response-invalid"
    assert captured.value.detail == detail


def test_wire_document_accepts_canonical_contract():
    assert wire_document(_contract(), "runtime-contract.schema.json")["version"] == 1


def test_wire_document_accepts_exact_byte_ceiling():
    text = _contract()
    text += " " * (MAX_WIRE_BYTES - len(text.encode("utf-8")))

    assert len(text.encode("utf-8")) == MAX_WIRE_BYTES
    assert wire_document(text, "runtime-contract.schema.json")["version"] == 1


def test_runtime_job_schema_snapshot_fails_closed_when_packaging_is_incomplete(monkeypatch):
    monkeypatch.setattr("omnitensor.training.runner.schema_versions", lambda: {})

    with pytest.raises(ForecastRunError) as captured:
        _runtime_job_schema_versions()

    assert captured.value.code == "runtime-contract-mismatch"
    assert captured.value.detail == "local runtime job schema versions are incomplete"


@pytest.mark.asyncio
async def test_runner_handshakes_submits_exact_tensor_and_polls_to_success(tmp_path):
    recorder = TelemetryRecorder(tmp_path)
    for row in _rows():
        recorder.record(row.profile_id, row.features, row.observed_at_ms)
    ids = iter(("submit-1", "poll-1", "poll-2"))
    client = FakeClient(
        results=[
            lambda request: _result(request["requestId"], state="running", output=None),
            lambda request: _result(request["requestId"]),
        ]
    )
    sleeps = []
    runner = TrustedForecastRunner(
        _workload(),
        recorder,
        client,
        attempts=2,
        poll_interval=0.25,
        sleep=lambda delay: _record_sleep(sleeps, delay),
        request_id=lambda: next(ids),
    )

    output = await runner.run()

    assert output == {
        "outputs": [[[0.75]]],
        "reading": {
            "kind": "forecast",
            "targetFeature": "load",
            "horizon": 1,
            "value": 0.75,
        },
    }
    assert client.submissions == [
        {
            "version": 1,
            "requestId": "submit-1",
            "workloadId": PROFILE,
            "payload": {"inputs": [[[1.5, 11.0, 2.5, 12.0]]]},
        }
    ]
    assert client.polls == [
        {"version": 1, "requestId": "poll-1", "jobId": "job-1"},
        {"version": 1, "requestId": "poll-2", "jobId": "job-1"},
    ]
    assert sleeps == [0.25]


@pytest.mark.asyncio
async def test_runner_uses_one_derived_schema_snapshot_and_ignores_unrelated_versions(
    tmp_path, monkeypatch
):
    recorder = TelemetryRecorder(tmp_path)
    for row in _rows(2):
        recorder.record(row.profile_id, row.features, row.observed_at_ms)
    local_versions = {
        "runtime-job-submit": 7,
        "runtime-job-acknowledgement": 1,
        "runtime-job-result-request": 9,
        "runtime-job-result": 1,
        "runtime-command": 2,
    }
    announced_versions = dict(local_versions)
    announced_versions["runtime-command"] = 99
    calls = []

    def snapshot():
        calls.append(True)
        return dict(local_versions)

    monkeypatch.setattr("omnitensor.training.runner.schema_versions", snapshot)
    client = FakeClient(contract=_contract(schemas=announced_versions))
    ids = iter(("submit-derived", "poll-derived"))

    output = await TrustedForecastRunner(
        _workload(), recorder, client, attempts=1, request_id=lambda: next(ids)
    ).run()

    assert output["reading"]["value"] == 0.75
    assert calls == [True]
    assert client.submissions[0]["version"] == 7
    assert client.polls == [
        {"version": 9, "requestId": "poll-derived", "jobId": "job-1"}
    ]


async def _record_sleep(seen, delay):
    seen.append(delay)


@pytest.mark.parametrize(
    ("attempts", "interval"),
    [(0, 0.1), (121, 0.1), (True, 0.1), (1, -0.1), (1, 5.1), (1, True)],
)
def test_runner_refuses_unsafe_poll_bounds(tmp_path, attempts, interval):
    with pytest.raises(ForecastRunError) as captured:
        TrustedForecastRunner(
            _workload(),
            TelemetryRecorder(tmp_path),
            FakeClient(),
            attempts=attempts,
            poll_interval=interval,
        )

    assert captured.value.code == "bounds-invalid"


def test_runner_accepts_exact_bounds_and_preserves_defaults(tmp_path):
    upper = TrustedForecastRunner(
        _workload(),
        TelemetryRecorder(tmp_path),
        FakeClient(),
        attempts=120,
        poll_interval=5,
    )
    defaults = TrustedForecastRunner(_workload(), TelemetryRecorder(tmp_path), FakeClient())

    assert upper._attempts == 120
    assert upper._poll_interval == 5.0
    assert defaults._attempts == 40
    assert defaults._poll_interval == 0.1


@pytest.mark.parametrize(
    ("attempts", "interval", "detail"),
    [
        (0, 0.1, "attempts must be from 1 to 120"),
        (1, -0.1, "poll interval must be from 0 to 5 seconds"),
    ],
)
def test_runner_poll_bound_failures_have_exact_details(tmp_path, attempts, interval, detail):
    with pytest.raises(ForecastRunError) as captured:
        TrustedForecastRunner(
            _workload(),
            TelemetryRecorder(tmp_path),
            FakeClient(),
            attempts=attempts,
            poll_interval=interval,
        )

    assert captured.value.detail == detail


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "contract", [_contract(methods=["DescribeContract"]), _contract(schemas={})]
)
async def test_runner_refuses_runtime_contract_drift(tmp_path, contract):
    with pytest.raises(ForecastRunError) as captured:
        await TrustedForecastRunner(
            _workload(), TelemetryRecorder(tmp_path), FakeClient(contract=contract)
        ).run()

    assert captured.value.code == "runtime-contract-mismatch"


@pytest.mark.asyncio
async def test_runner_preserves_submission_refusal(tmp_path):
    client = FakeClient(
        acknowledgement=_ack("submit", status="rejected", job_id=None, code="artifact-unavailable")
    )
    recorder = TelemetryRecorder(tmp_path)
    for row in _rows(2):
        recorder.record(row.profile_id, row.features, row.observed_at_ms)

    with pytest.raises(ForecastRunError) as captured:
        await TrustedForecastRunner(
            _workload(), recorder, client, request_id=lambda: "submit"
        ).run()

    assert captured.value.code == "artifact-unavailable"
    assert captured.value.detail == "reply detail"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "code"),
    [
        (lambda request: _result(request["requestId"], state="failed", output=None), "job-failed"),
        (
            lambda request: _result(request["requestId"], state="cancelled", output=None),
            "job-cancelled",
        ),
        (lambda request: _result(request["requestId"], state="succeeded", output={}), None),
        (lambda request: _result("wrong", output={}), "runtime-response-invalid"),
        (
            lambda request: _result(request["requestId"], job_id="wrong", output={}),
            "runtime-response-invalid",
        ),
    ],
)
async def test_runner_handles_terminal_and_identity_results(tmp_path, result, code):
    recorder = TelemetryRecorder(tmp_path)
    for row in _rows(2):
        recorder.record(row.profile_id, row.features, row.observed_at_ms)
    client = FakeClient(results=[result])
    runner = TrustedForecastRunner(_workload(), recorder, client)

    if code is None:
        assert await runner.run() == {}
    else:
        with pytest.raises(ForecastRunError) as captured:
            await runner.run()
        assert captured.value.code == code


@pytest.mark.asyncio
async def test_runner_refuses_success_without_output_and_times_out(tmp_path):
    recorder = TelemetryRecorder(tmp_path)
    for row in _rows(2):
        recorder.record(row.profile_id, row.features, row.observed_at_ms)
    no_output = FakeClient(
        results=[lambda request: _result(request["requestId"], state="succeeded", output=None)]
    )
    with pytest.raises(ForecastRunError) as captured:
        await TrustedForecastRunner(_workload(), recorder, no_output).run()
    assert captured.value.code == "runtime-response-invalid"

    running = FakeClient(
        results=[lambda request: _result(request["requestId"], state="running", output=None)]
    )
    with pytest.raises(ForecastRunError) as captured:
        await TrustedForecastRunner(_workload(), recorder, running, attempts=1).run()
    assert captured.value.code == "forecast-timeout"


@pytest.mark.asyncio
async def test_runner_refuses_changed_submission_identity(tmp_path):
    recorder = TelemetryRecorder(tmp_path)
    for row in _rows(2):
        recorder.record(row.profile_id, row.features, row.observed_at_ms)
    client = FakeClient(acknowledgement=_ack("different"))

    with pytest.raises(ForecastRunError) as captured:
        await TrustedForecastRunner(_workload(), recorder, client).run()

    assert captured.value.code == "runtime-response-invalid"


class FakeInterface:
    async def call_describe_contract(self):
        return "contract"

    async def call_submit_job(self, request):
        return f"submit:{request}"

    async def call_get_job_result(self, request):
        return f"result:{request}"


class FakeBus:
    def __init__(self):
        self.disconnected = False

    def disconnect(self):
        self.disconnected = True


@pytest.mark.asyncio
async def test_dbus_adapter_delegates_and_disconnects():
    bus = FakeBus()
    client = DbusForecastClient(bus, FakeInterface())

    assert await client.describe_contract() == "contract"
    assert await client.submit_job("one") == "submit:one"
    assert await client.get_job_result("two") == "result:two"
    client.close()

    assert bus.disconnected is True


@pytest.mark.asyncio
async def test_dbus_adapter_connects_to_exact_local_interface(monkeypatch):
    from dbus_fast import BusType

    interface = FakeInterface()
    calls = []

    class Proxy:
        def get_interface(self, name):
            calls.append(("interface", name))
            return interface

    class MessageBus:
        def __init__(self, *, bus_type):
            calls.append(("type", bus_type))

        async def connect(self):
            calls.append(("connect",))
            return self

        async def introspect(self, name, path):
            calls.append(("introspect", name, path))
            return "introspection"

        def get_proxy_object(self, name, path, introspection):
            calls.append(("proxy", name, path, introspection))
            return Proxy()

        def disconnect(self):
            calls.append(("disconnect",))

    monkeypatch.setattr("dbus_fast.aio.MessageBus", MessageBus)

    client = await DbusForecastClient.connect()

    assert client._interface is interface
    assert calls == [
        ("type", BusType.SESSION),
        ("connect",),
        ("introspect", "org.cinnamon.OmniTensor1", "/org/cinnamon/OmniTensor1"),
        (
            "proxy",
            "org.cinnamon.OmniTensor1",
            "/org/cinnamon/OmniTensor1",
            "introspection",
        ),
        ("interface", "org.cinnamon.OmniTensor1"),
    ]


@pytest.mark.asyncio
async def test_dbus_adapter_disconnects_failed_connection(monkeypatch):
    calls = []

    class MessageBus:
        def __init__(self, *, bus_type):
            self.bus_type = bus_type

        async def connect(self):
            return self

        async def introspect(self, _name, _path):
            raise RuntimeError("introspection lost")

        def disconnect(self):
            calls.append("disconnect")

    monkeypatch.setattr("dbus_fast.aio.MessageBus", MessageBus)

    with pytest.raises(ForecastRunError) as captured:
        await DbusForecastClient.connect()

    assert captured.value.code == "runtime-unavailable"
    assert captured.value.detail == "introspection lost"
    assert calls == ["disconnect"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method", ["call_describe_contract", "call_submit_job", "call_get_job_result"]
)
async def test_dbus_adapter_translates_transport_failures(method):
    class BrokenInterface(FakeInterface):
        def __getattribute__(self, name):
            if name == method:

                async def broken(*_args):
                    raise RuntimeError("bus lost")

                return broken
            return super().__getattribute__(name)

    client = DbusForecastClient(FakeBus(), BrokenInterface())
    call = {
        "call_describe_contract": lambda: client.describe_contract(),
        "call_submit_job": lambda: client.submit_job("x"),
        "call_get_job_result": lambda: client.get_job_result("x"),
    }[method]

    with pytest.raises(ForecastRunError) as captured:
        await call()

    assert captured.value.code == "runtime-unavailable"
    assert captured.value.detail == "bus lost"


def test_forecast_cli_runs_without_accepting_inline_features(tmp_path, monkeypatch, capsys):
    recorder = TelemetryRecorder(tmp_path / "records")
    for row in _rows(2):
        recorder.record(row.profile_id, row.features, row.observed_at_ms)
    client = FakeClient()

    class ClientFactory:
        @staticmethod
        async def connect():
            return client

    monkeypatch.setattr(
        "omnitensor.training.cli.load_forecast_binding", lambda *a, **k: _workload()
    )
    monkeypatch.setattr("omnitensor.training.cli.DbusForecastClient", ClientFactory)

    code = forecast_main(
        [
            "--profile",
            PROFILE,
            "--records-root",
            str(tmp_path / "records"),
            "--bindings-root",
            str(tmp_path / "bindings"),
            "--attempts",
            "1",
            "--poll-interval",
            "0",
        ]
    )

    assert code == 0
    assert json.loads(capsys.readouterr().out) == {
        "kind": "forecast",
        "targetFeature": "load",
        "horizon": 1,
        "value": 0.75,
    }
    assert client.closed is True
    with pytest.raises(SystemExit):
        forecast_main(["--profile", PROFILE, "--feature", "load=1"])


def test_forecast_cli_reports_stable_failure(monkeypatch, capsys):
    monkeypatch.setattr(
        "omnitensor.training.cli.load_forecast_binding",
        lambda *a, **k: (_ for _ in ()).throw(ForecastRunError("binding-unavailable", "missing")),
    )

    assert forecast_main(["--profile", PROFILE]) == 1
    assert capsys.readouterr().err == "forecast failed: binding-unavailable: missing\n"


def test_forecast_cli_refuses_runtime_output_without_canonical_reading(monkeypatch, capsys):
    client = FakeClient()

    class ClientFactory:
        @staticmethod
        async def connect():
            return client

    class Runner:
        def __init__(self, *_args, **_options):
            pass

        async def run(self):
            return {"outputs": [[[0.75]]]}

    monkeypatch.setattr(
        "omnitensor.training.cli.load_forecast_binding", lambda *a, **k: _workload()
    )
    monkeypatch.setattr("omnitensor.training.cli.DbusForecastClient", ClientFactory)
    monkeypatch.setattr("omnitensor.training.cli.TrustedForecastRunner", Runner)

    assert forecast_main(["--profile", PROFILE]) == 1
    assert capsys.readouterr().err == (
        "forecast failed: runtime-response-invalid: runtime returned no valid forecast reading\n"
    )
    assert client.closed is True


def test_forecast_cli_default_contract_and_exact_help(monkeypatch, capsys):
    captured = {}
    client = FakeClient()

    def load(profile, root):
        captured["binding"] = (profile, root)
        return _workload()

    class ClientFactory:
        @staticmethod
        async def connect():
            return client

    class Runner:
        def __init__(self, workload, recorder, selected_client, **options):
            captured["runner"] = (workload, recorder.root, selected_client, options)

        async def run(self):
            return {
                "reading": {
                    "kind": "forecast",
                    "targetFeature": "load",
                    "horizon": 1,
                    "value": 1,
                }
            }

    monkeypatch.setattr("omnitensor.training.cli.load_forecast_binding", load)
    monkeypatch.setattr("omnitensor.training.cli.DbusForecastClient", ClientFactory)
    monkeypatch.setattr("omnitensor.training.cli.TrustedForecastRunner", Runner)

    assert forecast_main(["--profile", PROFILE]) == 0
    output = capsys.readouterr().out
    assert output == (
        '{\n  "kind": "forecast",\n  "targetFeature": "load",\n  "horizon": 1,\n  "value": 1.0\n}\n'
    )
    assert captured["binding"] == (
        PROFILE,
        Path("~/.local/share/omnitensor/model-bindings").expanduser(),
    )
    workload, records_root, selected_client, options = captured["runner"]
    assert workload.id == PROFILE
    assert records_root == Path("~/.local/state/omnitensor/telemetry").expanduser()
    assert selected_client is client
    assert options == {"attempts": 40, "poll_interval": 0.1}
    assert client.closed is True

    with pytest.raises(SystemExit) as help_exit:
        forecast_main(["--help"])
    help_text = capsys.readouterr().out
    assert help_exit.value.code == 0
    assert help_text.startswith("usage: omnitensor-run-forecast")
    assert "Submit one forecast assembled only from trusted local history" in help_text

    with pytest.raises(SystemExit) as missing_exit:
        forecast_main([])
    assert missing_exit.value.code == 2
    assert "the following arguments are required: --profile" in capsys.readouterr().err
