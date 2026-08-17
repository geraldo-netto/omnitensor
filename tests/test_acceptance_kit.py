from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import pickle
import re
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from omnitensor.atomicio import JsonTooLargeError
from omnitensor.plugins.acceptance_kit import (
    JsonDigestMismatchError,
    NativeLoadReport,
    discover_resource,
    native_load_report_document,
    parse_native_load_report,
    read_bounded_json,
    require_boolean,
    require_integer,
    require_mapping,
    require_match,
    require_sequence,
    require_text,
    run_acceptance_cli,
    validate_acceptance_report,
    validate_gpu_load,
    validate_npu_load,
)
from omnitensor.plugins.generation import ProviderGenerationError


class KitError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def test_bounded_json_accepts_exact_limit_and_refuses_one_byte_less(tmp_path):
    payload = b'{"value":1}'
    path = tmp_path / "document.json"
    path.write_bytes(payload)

    snapshot = read_bounded_json(path, len(payload))

    assert snapshot.document == {"value": 1}
    assert snapshot.raw == payload
    assert snapshot.sha256 == hashlib.sha256(payload).hexdigest()
    with pytest.raises(JsonTooLargeError):
        read_bounded_json(path, len(payload) - 1)


@pytest.mark.parametrize("maximum", [True, False, 0, -1, 1.0, "1"])
def test_bounded_json_requires_a_positive_integer_limit(tmp_path, maximum):
    path = tmp_path / "document.json"
    path.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="max_bytes must be a positive integer"):
        read_bounded_json(path, maximum)


def test_bounded_json_digest_and_document_share_one_descriptor_snapshot(tmp_path, monkeypatch):
    original = b'{"generation":1}'
    replacement = b'{"generation":2}'
    target = tmp_path / "evidence.json"
    staged = tmp_path / "replacement.json"
    target.write_bytes(original)
    staged.write_bytes(replacement)
    real_open = Path.open

    class ReplacingStream(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            os.replace(staged, target)
            return super().read(size)

    def open_once(path: Path, *args, **kwargs):
        if path == target:
            return ReplacingStream(original)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_once)
    snapshot = read_bounded_json(target, len(original))

    assert snapshot.document == {"generation": 1}
    assert snapshot.sha256 == hashlib.sha256(original).hexdigest()
    with real_open(target, "rb") as stream:
        assert stream.read() == replacement


def test_expected_digest_is_checked_before_json_parsing(tmp_path):
    path = tmp_path / "malformed.json"
    path.write_bytes(b"{")

    with pytest.raises(JsonDigestMismatchError):
        read_bounded_json(path, 1, expected_sha256="0" * 64)
    with pytest.raises(ValueError):
        read_bounded_json(path, 1, expected_sha256=hashlib.sha256(b"{").hexdigest())


def test_typed_contract_helpers_inject_errors_and_preserve_policies():
    mapping = {"value": 1}
    sequence = [1]
    assert (
        require_mapping(mapping, error_type=KitError, code="bad", detail="object required")
        is mapping
    )
    assert (
        require_sequence(
            sequence,
            error_type=KitError,
            code="bad",
            detail="array required",
            allow_empty=False,
        )
        is sequence
    )
    assert (
        require_sequence(
            [],
            error_type=KitError,
            code="bad",
            detail="array required",
            allow_empty=True,
        )
        == []
    )
    assert (
        require_text(
            " ",
            error_type=KitError,
            code="bad",
            detail="text required",
            maximum=1,
            allow_whitespace=True,
        )
        == " "
    )
    assert (
        require_match(
            "valid-id",
            re.compile(r"^[a-z]+(?:-[a-z]+)*$"),
            error_type=KitError,
            code="bad",
            detail="identifier required",
        )
        == "valid-id"
    )
    assert (
        require_integer(
            0,
            error_type=KitError,
            code="bad",
            detail="integer required",
            minimum=0,
            maximum=1,
        )
        == 0
    )
    assert (
        require_boolean(False, error_type=KitError, code="bad", detail="boolean required") is False
    )

    refusals = (
        lambda: require_mapping([], error_type=KitError, code="mapping", detail="refused"),
        lambda: require_sequence(
            "value",
            error_type=KitError,
            code="sequence",
            detail="refused",
            allow_empty=True,
        ),
        lambda: require_sequence(
            [],
            error_type=KitError,
            code="sequence",
            detail="refused",
            allow_empty=False,
        ),
        lambda: require_text(
            " ",
            error_type=KitError,
            code="text",
            detail="refused",
            maximum=1,
            allow_whitespace=False,
        ),
        lambda: require_text(
            "too long",
            error_type=KitError,
            code="text",
            detail="refused",
            maximum=2,
            allow_whitespace=True,
        ),
        lambda: require_match(
            "BAD",
            re.compile("^[a-z]+$"),
            error_type=KitError,
            code="match",
            detail="refused",
        ),
        lambda: require_integer(
            True,
            error_type=KitError,
            code="integer",
            detail="refused",
            minimum=0,
        ),
        lambda: require_integer(
            2,
            error_type=KitError,
            code="integer",
            detail="refused",
            minimum=0,
            maximum=1,
        ),
        lambda: require_boolean(0, error_type=KitError, code="boolean", detail="refused"),
    )
    for refusal in refusals:
        with pytest.raises(KitError, match="refused"):
            refusal()


@given(st.integers(min_value=-100, max_value=100))
def test_integer_helper_accepts_exactly_the_closed_interval(value):
    if -5 <= value <= 7:
        assert (
            require_integer(
                value,
                error_type=KitError,
                code="integer",
                detail="out of bounds",
                minimum=-5,
                maximum=7,
            )
            == value
        )
    else:
        with pytest.raises(KitError, match="out of bounds"):
            require_integer(
                value,
                error_type=KitError,
                code="integer",
                detail="out of bounds",
                minimum=-5,
                maximum=7,
            )


def test_resource_discovery_is_ordered_and_retains_the_final_fallback(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    fallback = tmp_path / "fallback"
    second.write_text("second", encoding="utf-8")
    fallback.write_text("fallback", encoding="utf-8")

    assert discover_resource(first, second, fallback) == second
    second.unlink()
    assert discover_resource(first, second, fallback) == fallback
    fallback.unlink()
    assert discover_resource(first, second, fallback) == fallback
    with pytest.raises(ValueError, match="at least one"):
        discover_resource()


def test_report_validation_returns_identity_and_uses_first_violation():
    report = {"qualified": True}
    assert (
        validate_acceptance_report(
            "acceptance.schema.json",
            report,
            validator=lambda _schema, _value: [],
            error_type=KitError,
        )
        is report
    )
    with pytest.raises(KitError) as caught:
        validate_acceptance_report(
            "acceptance.schema.json",
            report,
            validator=lambda _schema, _value: ["first", "second"],
            error_type=KitError,
        )
    assert (caught.value.code, caught.value.detail) == ("report-invalid", "first")


@pytest.mark.parametrize("ensure_ascii", [True, False])
def test_shared_cli_preserves_execution_order_and_stdout_encoding(tmp_path, capsys, ensure_ascii):
    trace = []
    report = {"text": "שלום"}

    def invoke():
        trace.append("invoke")
        return report

    def writer(path, value, *, prefix):
        trace.append(("write", path, value, prefix))

    output = tmp_path / "report.json"
    assert (
        run_acceptance_cli(
            invoke,
            output,
            writer=writer,
            output_prefix=".acceptance-",
            failure_prefix="acceptance",
            error_types=(KitError,),
            ensure_ascii=ensure_ascii,
        )
        == 0
    )
    assert trace == ["invoke", ("write", output, report, ".acceptance-")]
    assert capsys.readouterr().out == json.dumps(report, indent=2, ensure_ascii=ensure_ascii) + "\n"


def test_shared_cli_refuses_without_writing_or_printing_success(tmp_path, capsys):
    def refuse():
        raise KitError("evidence-invalid", "missing")

    assert (
        run_acceptance_cli(
            refuse,
            tmp_path / "report.json",
            writer=lambda *_args, **_kwargs: pytest.fail("writer called"),
            output_prefix=".acceptance-",
            failure_prefix="acceptance",
            error_types=(KitError,),
            ensure_ascii=True,
        )
        == 1
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "acceptance failed: evidence-invalid: missing\n"


@pytest.mark.parametrize(
    "report",
    [
        object(),
        NativeLoadReport("wrong", "Vulkan", 1, 1, False),
        NativeLoadReport("llama.cpp-vulkan", "CPU", 1, 1, False),
        NativeLoadReport("llama.cpp-vulkan", "Vulkan", 1, 1, True),
        NativeLoadReport("llama.cpp-vulkan", "Vulkan", 0, 0, False),
        NativeLoadReport("llama.cpp-vulkan", "Vulkan", 2, 1, False),
    ],
)
def test_public_gpu_validator_rejects_backend_device_fallback_and_layer_drift(report):
    with pytest.raises(ProviderGenerationError) as caught:
        validate_gpu_load(report)
    assert caught.value.code == "model-load-failed"
    assert caught.value.detail == "llama.cpp did not prove complete Vulkan model-layer offload"
    assert caught.value.generation_started is False


def test_public_gpu_validator_accepts_the_minimum_complete_offload():
    validate_gpu_load(NativeLoadReport("llama.cpp-vulkan", "Vulkan", 1, 1, False))


@pytest.mark.parametrize(
    "report",
    [
        object(),
        NativeLoadReport("wrong", "NPU", 0, 0, False),
        NativeLoadReport("openvino-genai", "GPU", 0, 0, False),
        NativeLoadReport("openvino-genai", "NPU", 0, 0, True),
    ],
)
def test_public_npu_validator_rejects_backend_device_and_fallback(report):
    with pytest.raises(ProviderGenerationError) as caught:
        validate_npu_load(report)
    assert caught.value.code == "model-load-failed"
    assert caught.value.detail == "OpenVINO GenAI did not prove NPU-only execution"
    assert caught.value.generation_started is False


def test_public_npu_validator_preserves_the_existing_layer_agnostic_contract():
    validate_npu_load(NativeLoadReport("openvino-genai", "NPU", 0, -1, False))


def test_the_native_report_pickles_through_the_module_that_defines_it():
    # It used to claim `omnitensor.plugins.qwen` as its `__module__` so that
    # pickle resolved it through a facade named after one model. It says where
    # it lives, which is what pickle needs and what a traceback should show.
    report = NativeLoadReport("llama.cpp-vulkan", "Vulkan", 1, 1, False)

    assert NativeLoadReport.__module__ == "omnitensor.plugins.acceptance_kit"
    assert pickle.loads(pickle.dumps(report)) == report


def test_native_load_report_codec_round_trips_and_injects_family_errors():
    report = NativeLoadReport("llama.cpp-vulkan", "Vulkan", 37, 37, False)
    document = native_load_report_document(report)

    assert document == {
        "backend": "llama.cpp-vulkan",
        "device": "Vulkan",
        "totalModelLayers": 37,
        "acceleratorLayers": 37,
        "cpuFallback": False,
    }
    assert (
        parse_native_load_report(
            document,
            error_type=KitError,
            code="load-invalid",
            object_detail="load must be object",
            fields_detail="load fields differ",
        )
        == report
    )
    with pytest.raises(TypeError, match="report must be NativeLoadReport"):
        native_load_report_document(object())
    with pytest.raises(KitError, match="load must be object"):
        parse_native_load_report(
            [],
            error_type=KitError,
            code="load-invalid",
            object_detail="load must be object",
            fields_detail="load fields differ",
        )
    with pytest.raises(KitError, match="load fields differ"):
        parse_native_load_report(
            {**document, "extra": True},
            error_type=KitError,
            code="load-invalid",
            object_detail="load must be object",
            fields_detail="load fields differ",
        )


def test_external_provider_factory_imports_only_the_public_load_validator():
    root = Path(__file__).parents[1]
    source = (
        root / "providers/qwen-vulkan-runtime/src/omnitensor_vulkan_runtime/factories.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        (node.module, alias.name)
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }

    assert ("omnitensor.plugins.acceptance_kit", "validate_gpu_load") in imports
    assert ("omnitensor.plugins.generation_workers", "_validate_gpu_load") not in imports
