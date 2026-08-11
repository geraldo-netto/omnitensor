from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import omnitensor.training.document_model as document_model
from omnitensor.training.document_model import (
    MAX_TEXT_BYTES,
    NATIVE_TENSOR_CONTRACT,
    RECIPE_ID,
    BgeTokenizer,
    DocumentModelError,
    DocumentModelEvidence,
    PortableBgeRunner,
    TokenizedText,
    VulkanBgeRunner,
    _append_ncnn_l2_normalization,
    _binding_document,
    _bundled_corpus_path,
    _expected_retrieval_hits,
    _maximum_embedding_error,
    _report_document,
    _select_vulkan_device,
    _source_path,
    export_bge_ncnn,
    install_document_model,
    load_bge_holdout,
    main,
)
from omnitensor.training.embedding_production import EmbeddingGateEvidence
from omnitensor.training.recipes import FetchedModelSource, load_model_recipe


def _tokens() -> TokenizedText:
    return TokenizedText(
        tuple(range(128)),
        tuple(1.0 if index < 8 else 0.0 for index in range(128)),
        (0,) * 128,
    )


def _source(tmp_path: Path) -> FetchedModelSource:
    recipe = load_model_recipe(Path("model-recipes") / f"{RECIPE_ID}.json")
    root = tmp_path / "source"
    root.mkdir()
    for item in recipe.sources:
        (root / item.filename).write_bytes(b"source")
    receipt = root / "source-receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    return FetchedModelSource(recipe, root, receipt)


def _assert_document_error(error, code: str, detail: str) -> None:
    assert error.code == code
    assert error.detail == detail
    assert str(error) == f"{code}: {detail}"


class _Encoding:
    ids = list(range(128))
    attention_mask = [1] * 8 + [0] * 120
    type_ids = [0] * 128


class _Tokenizer:
    observed = []

    @classmethod
    def from_file(cls, path):
        cls.observed.append(("load", path))
        return cls()

    def enable_truncation(self, **options):
        self.observed.append(("truncate", options))

    def enable_padding(self, **options):
        self.observed.append(("pad", options))

    def encode(self, text):
        self.observed.append(("encode", text))
        if text == "explode":
            raise RuntimeError("broken")
        return _Encoding()


def test_bge_tokenizer_is_fixed_bounded_and_query_explicit(monkeypatch, tmp_path):
    _Tokenizer.observed = []
    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=_Tokenizer))
    tokenizer = BgeTokenizer(tmp_path / "tokenizer.json")

    document = tokenizer.encode("a document")
    query = tokenizer.encode("a question", query=True)

    assert document == _tokens()
    assert query == _tokens()
    assert _Tokenizer.observed == [
        ("load", str(tmp_path / "tokenizer.json")),
        ("truncate", {"max_length": 128}),
        ("pad", {"length": 128}),
        ("encode", "a document"),
        (
            "encode",
            "Represent this sentence for searching relevant passages: a question",
        ),
    ]
    assert [len(item[0]) for item in query.inputs()] == [128, 128, 128]


def test_bge_tokenizer_contains_invalid_tokenizer_file(monkeypatch, tmp_path):
    class BrokenTokenizer:
        @staticmethod
        def from_file(_path):
            raise ValueError("malformed tokenizer")

    monkeypatch.setitem(
        sys.modules, "tokenizers", SimpleNamespace(Tokenizer=BrokenTokenizer)
    )
    with pytest.raises(DocumentModelError) as caught:
        BgeTokenizer(tmp_path / "tokenizer.json")

    _assert_document_error(
        caught.value, "tokenizer-invalid", "cannot load tokenizer: malformed tokenizer"
    )


@pytest.mark.parametrize(
    ("value", "code"),
    [
        ("", "text-invalid"),
        (" ", "text-invalid"),
        (7, "text-invalid"),
        ("x" * (MAX_TEXT_BYTES + 1), "text-too-large"),
        ("explode", "tokenizer-failed"),
    ],
)
def test_bge_tokenizer_refuses_invalid_or_failed_text(monkeypatch, tmp_path, value, code):
    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=_Tokenizer))
    tokenizer = BgeTokenizer(tmp_path / "tokenizer.json")
    with pytest.raises(DocumentModelError) as caught:
        tokenizer.encode(value)
    expected = {
        "text-invalid": "text must be a non-empty string",
        "text-too-large": "text exceeds the 2 MiB producer bound",
        "tokenizer-failed": "cannot tokenize text: broken",
    }
    _assert_document_error(caught.value, code, expected[code])


def test_bge_tokenizer_accepts_exact_byte_bound_and_rejects_wrong_tensor_length(
    monkeypatch, tmp_path
):
    monkeypatch.setitem(sys.modules, "tokenizers", SimpleNamespace(Tokenizer=_Tokenizer))
    tokenizer = BgeTokenizer(tmp_path / "tokenizer.json")
    assert tokenizer.encode("x" * MAX_TEXT_BYTES) == _tokens()

    class ShortTokenizer(_Tokenizer):
        def encode(self, _text):
            return SimpleNamespace(ids=[1] * 127, attention_mask=[1] * 127, type_ids=[0] * 127)

    monkeypatch.setitem(
        sys.modules, "tokenizers", SimpleNamespace(Tokenizer=ShortTokenizer)
    )
    tokenizer = BgeTokenizer(tmp_path / "short.json")
    with pytest.raises(DocumentModelError) as caught:
        tokenizer.encode("text")
    _assert_document_error(
        caught.value, "tokenizer-invalid", "tokenizer did not emit fixed inputs"
    )


def test_bge_holdout_is_bundled_versioned_and_has_exact_retrieval_targets(tmp_path):
    holdout, expected = load_bge_holdout(_bundled_corpus_path())
    assert holdout.license_id == "CC0-1.0"
    assert holdout.corpus_sha256 == (
        "0f9efc25e4563bf0c29aea80772052fc0c32c1c1adacf4136c78657aae3fc14e"
    )
    assert len(holdout.queries) == 2
    assert len(holdout.documents) == 12
    assert expected == (0, 1)
    assert all(query.startswith("Represent this sentence") for query in holdout.queries)

    malformed = tmp_path / "corpus.json"
    malformed.write_text('{"corpusVersion":2}', encoding="utf-8")
    with pytest.raises(DocumentModelError) as caught:
        load_bge_holdout(malformed)
    _assert_document_error(
        caught.value,
        "corpus-invalid",
        "BGE corpus fields or version are invalid",
    )

    document = json.loads(_bundled_corpus_path().read_text(encoding="utf-8"))
    document["extra"] = True
    malformed.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(DocumentModelError) as extra:
        load_bge_holdout(malformed)
    _assert_document_error(
        extra.value,
        "corpus-invalid",
        "BGE corpus fields or version are invalid",
    )


def test_bge_holdout_contains_read_failure_and_non_mapping_shape(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "omnitensor.training.document_model.read_json_bounded",
        lambda _path, _limit: (_ for _ in ()).throw(ValueError("broken document")),
    )
    with pytest.raises(DocumentModelError) as unreadable:
        load_bge_holdout(tmp_path / "corpus.json")
    _assert_document_error(
        unreadable.value,
        "corpus-invalid",
        "cannot read BGE corpus: broken document",
    )

    fields = [
        "corpusVersion",
        "id",
        "license",
        "provenance",
        "queries",
        "documents",
        "expectedTopDocument",
    ]
    monkeypatch.setattr(
        "omnitensor.training.document_model.read_json_bounded",
        lambda _path, _limit: fields,
    )
    with pytest.raises(DocumentModelError) as non_mapping:
        load_bge_holdout(tmp_path / "corpus.json")
    _assert_document_error(
        non_mapping.value,
        "corpus-invalid",
        "BGE corpus fields or version are invalid",
    )


@pytest.mark.parametrize(
    ("change", "detail"),
    [
        ({"queries": "not-a-list"}, "expectations"),
        ({"queries": "ab"}, "expectations"),
        ({"documents": "abcdefghijkl"}, "expectations"),
        ({"expectedTopDocument": [0]}, "expectations"),
        ({"expectedTopDocument": [True, 1]}, "expectations"),
        ({"expectedTopDocument": [12, 1]}, "expectations"),
        ({"documents": [7] * 12}, "texts"),
        ({"license": ""}, "license"),
    ],
)
def test_bge_holdout_rejects_malformed_semantics(tmp_path, change, detail):
    document = json.loads(_bundled_corpus_path().read_text(encoding="utf-8"))
    document.update(change)
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(DocumentModelError) as caught:
        load_bge_holdout(path)

    expected = {
        "expectations": "BGE retrieval expectations are invalid",
        "texts": "BGE corpus texts must be strings",
        "license": "embedding holdout license id is required",
    }
    _assert_document_error(caught.value, "corpus-invalid", expected[detail])


def test_bundled_corpus_falls_back_to_packaged_location(monkeypatch, tmp_path):
    module_path = tmp_path / "package" / "training" / "document_model.py"
    packaged = module_path.resolve().parents[1] / "evaluation-corpora" / f"{RECIPE_ID}.json"
    packaged.parent.mkdir(parents=True)
    packaged.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(document_model, "__file__", str(module_path))

    assert _bundled_corpus_path() == packaged


class _GpuInfo:
    def __init__(self, kind, name):
        self._kind = kind
        self._name = name

    def type(self):
        return self._kind

    def device_name(self):
        return self._name


class _RuntimeInventory:
    def __init__(self, devices):
        self.devices = devices

    def get_gpu_count(self):
        return len(self.devices)

    def get_gpu_info(self, index):
        return self.devices[index]


def test_vulkan_selection_prefers_discrete_and_refuses_software_or_bad_request():
    runtime = _RuntimeInventory(
        [_GpuInfo(1, "integrated"), _GpuInfo(0, "discrete"), _GpuInfo(3, "software")]
    )
    assert _select_vulkan_device(runtime, None) == (1, "discrete")
    assert _select_vulkan_device(runtime, 0) == (0, "integrated")
    with pytest.raises(DocumentModelError) as software:
        _select_vulkan_device(runtime, 2)
    _assert_document_error(
        software.value,
        "device-unavailable",
        "Vulkan device 2 is absent or software-only",
    )
    with pytest.raises(DocumentModelError) as hardware:
        _select_vulkan_device(_RuntimeInventory([_GpuInfo(3, "software")]), None)
    _assert_document_error(
        hardware.value,
        "device-unavailable",
        "no hardware Vulkan device is available",
    )

    class BrokenInventory:
        @staticmethod
        def get_gpu_count():
            raise OSError("Vulkan loader failed")

    with pytest.raises(DocumentModelError) as enumeration:
        _select_vulkan_device(BrokenInventory(), None)
    _assert_document_error(
        enumeration.value,
        "device-unavailable",
        "cannot enumerate Vulkan: Vulkan loader failed",
    )


class _FakeMat:
    def __init__(self, value):
        self.value = np.asarray(value)

    def clone(self):
        return _FakeMat(self.value.copy())


class _Extractor:
    def __init__(self):
        self.inputs = {}

    def input(self, name, value):
        self.inputs[name] = value
        return 0

    def extract(self, name):
        assert name == "out0"
        return 0, np.asarray([1.0] + [0.0] * 383, dtype=np.float32)


class _Net:
    def __init__(self, runtime):
        self.runtime = runtime
        self.opt = SimpleNamespace()

    def set_vulkan_device(self, index):
        self.runtime.selected = index

    def load_param(self, path):
        self.runtime.loaded.append(path)
        return 0

    def load_model(self, path):
        self.runtime.loaded.append(path)
        return 0

    def create_extractor(self):
        self.runtime.extractor = _Extractor()
        return self.runtime.extractor


class _Ncnn(_RuntimeInventory):
    Mat = _FakeMat

    def __init__(self):
        super().__init__([_GpuInfo(0, "named-gpu")])
        self.selected = None
        self.loaded = []
        self.extractor = None

    def Net(self):  # noqa: N802 - mirrors the ncnn Python API
        self.net = _Net(self)
        return self.net


class _StaticTokenizer:
    @staticmethod
    def encode(_text):
        return _tokens()


def test_native_runner_uses_mixed_dtypes_precise_vulkan_and_normalizes(tmp_path):
    runtime = _Ncnn()
    runner = VulkanBgeRunner(
        tmp_path / "model.ncnn.param", _StaticTokenizer(), runtime=runtime
    )
    vector = runner.embed("text")

    assert runner.device_name == "named-gpu"
    assert runtime.selected == 0
    assert runtime.loaded == [
        str(tmp_path / "model.ncnn.param"),
        str(tmp_path / "model.ncnn.bin"),
    ]
    assert runtime.net.opt.use_vulkan_compute is True
    assert runtime.net.opt.use_fp16_packed is False
    assert runtime.net.opt.use_fp16_storage is False
    assert runtime.net.opt.use_fp16_arithmetic is False
    assert [runtime.extractor.inputs[f"in{i}"].value.dtype for i in range(3)] == [
        np.dtype("int32"),
        np.dtype("float32"),
        np.dtype("int32"),
    ]
    assert len(vector) == 384
    assert vector == (1.0,) + (0.0,) * 383
    assert sum(value * value for value in vector) == pytest.approx(1.0)


def test_native_runner_honors_requested_device_and_contains_load_failures(tmp_path):
    runtime = _Ncnn()
    runtime.devices = [_GpuInfo(0, "discrete"), _GpuInfo(1, "integrated")]
    runner = VulkanBgeRunner(
        tmp_path / "model.ncnn.param",
        _StaticTokenizer(),
        device_index=1,
        runtime=runtime,
    )
    assert (runner.device_index, runner.device_name, runtime.selected) == (
        1,
        "integrated",
        1,
    )

    class BrokenNet(_Net):
        def load_param(self, path):
            self.runtime.loaded.append(path)
            return self.runtime.param_code

        def load_model(self, path):
            self.runtime.loaded.append(path)
            return self.runtime.model_code

    class BrokenRuntime(_Ncnn):
        def __init__(self, param_code, model_code):
            super().__init__()
            self.param_code = param_code
            self.model_code = model_code

        def Net(self):  # noqa: N802 - mirrors the ncnn Python API
            self.net = BrokenNet(self)
            return self.net

    for param_code, model_code in ((-1, 0), (0, -1)):
        with pytest.raises(DocumentModelError) as caught:
            VulkanBgeRunner(
                tmp_path / "model.ncnn.param",
                _StaticTokenizer(),
                runtime=BrokenRuntime(param_code, model_code),
            )
        _assert_document_error(
            caught.value, "native-invalid", "cannot load the compiled ncnn model"
        )


def test_native_runner_contains_dependency_and_extractor_failures(monkeypatch, tmp_path):
    dependency_detail = "install OmniTensor with document-producers"
    monkeypatch.setitem(sys.modules, "ncnn", None)
    with pytest.raises(DocumentModelError) as missing:
        VulkanBgeRunner(tmp_path / "model.ncnn.param", _StaticTokenizer())
    _assert_document_error(
        missing.value, "producer-dependency-missing", dependency_detail
    )

    runtime = _Ncnn()
    runner = VulkanBgeRunner(
        tmp_path / "model.ncnn.param", _StaticTokenizer(), runtime=runtime
    )

    class RefusingExtractor(_Extractor):
        def input(self, name, value):
            super().input(name, value)
            return -1 if name == "in1" else 0

    runner._net = SimpleNamespace(  # noqa: SLF001 - injected native failure boundary
        create_extractor=RefusingExtractor
    )
    with pytest.raises(DocumentModelError) as refused:
        runner.embed("text")
    _assert_document_error(
        refused.value, "native-failed", "native input 1 was refused"
    )

    class ExtractionFailure(_Extractor):
        def extract(self, _name):
            return -1, None

    runner._net = SimpleNamespace(  # noqa: SLF001 - injected native failure boundary
        create_extractor=ExtractionFailure
    )
    with pytest.raises(DocumentModelError) as extraction:
        runner.embed("text")
    _assert_document_error(
        extraction.value, "native-failed", "native embedding extraction failed"
    )

    monkeypatch.setitem(sys.modules, "numpy", None)
    with pytest.raises(DocumentModelError) as numpy_missing:
        runner.embed("text")
    _assert_document_error(
        numpy_missing.value, "producer-dependency-missing", dependency_detail
    )


def test_portable_runner_uses_only_cpu_reference_provider(monkeypatch, tmp_path):
    observed = {}

    class Session:
        def __init__(self, path, providers):
            observed["init"] = (path, providers)

        def run(self, outputs, inputs):
            observed["run"] = (outputs, inputs)
            return [np.ones((1, 384), dtype=np.float32)]

    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=Session))
    runner = PortableBgeRunner(tmp_path / "reference.onnx", _StaticTokenizer())

    assert runner.embed("text") == (1.0,) * 384
    assert observed["init"] == (
        str(tmp_path / "reference.onnx"),
        ["CPUExecutionProvider"],
    )
    assert [value.dtype for value in observed["run"][1].values()] == [
        np.dtype("int64"),
        np.dtype("int64"),
        np.dtype("int64"),
    ]
    assert tuple(observed["run"][1]) == (
        "input_ids",
        "attention_mask",
        "token_type_ids",
    )


def test_portable_runner_contains_session_and_inference_failures(monkeypatch, tmp_path):
    class InvalidSession:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("bad graph")

    monkeypatch.setitem(
        sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=InvalidSession)
    )
    with pytest.raises(DocumentModelError) as invalid:
        PortableBgeRunner(tmp_path / "reference.onnx", _StaticTokenizer())
    _assert_document_error(
        invalid.value, "portable-invalid", "cannot load portable model: bad graph"
    )

    class FailingSession:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, *_args):
            raise RuntimeError("provider stopped")

    monkeypatch.setitem(
        sys.modules, "onnxruntime", SimpleNamespace(InferenceSession=FailingSession)
    )
    runner = PortableBgeRunner(tmp_path / "reference.onnx", _StaticTokenizer())
    with pytest.raises(DocumentModelError) as failed:
        runner.embed("text")
    _assert_document_error(
        failed.value, "portable-failed", "portable inference failed: provider stopped"
    )


def test_producer_only_dependency_failures_are_exact(monkeypatch, tmp_path):
    dependency_detail = "install OmniTensor with document-producers"

    monkeypatch.setitem(sys.modules, "tokenizers", None)
    with pytest.raises(DocumentModelError) as tokenizer:
        BgeTokenizer(tmp_path / "tokenizer.json")
    _assert_document_error(
        tokenizer.value, "producer-dependency-missing", dependency_detail
    )

    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(DocumentModelError) as portable:
        PortableBgeRunner(tmp_path / "reference.onnx", _StaticTokenizer())
    _assert_document_error(
        portable.value, "producer-dependency-missing", dependency_detail
    )

    monkeypatch.setitem(sys.modules, "pnnx", None)
    with pytest.raises(DocumentModelError) as stack:
        document_model._document_dependencies()
    _assert_document_error(stack.value, "producer-dependency-missing", dependency_detail)

    monkeypatch.setitem(sys.modules, "numpy", None)
    with pytest.raises(DocumentModelError) as array:
        VulkanBgeRunner._array([1.0])
    _assert_document_error(array.value, "producer-dependency-missing", dependency_detail)


def test_retrieval_and_error_gates_measure_every_vector():
    holdout, expected = load_bge_holdout(_bundled_corpus_path())

    class Runner:
        def embed(self, text):
            vector = [0.0] * 384
            if "software" in text or "Package managers" in text:
                vector[0] = 1.0
            elif "solar" in text.lower() or "Solar panels" in text:
                vector[1] = 1.0
            else:
                vector[2] = 1.0
            return vector

    assert _maximum_embedding_error(holdout, Runner(), Runner()) == 0.0
    assert _expected_retrieval_hits(holdout, expected, Runner()) == 2


def test_export_reconstructs_pinned_weights_and_requires_complete_pair(  # noqa: C901
    monkeypatch, tmp_path
):
    source = _source(tmp_path)
    observed = {}

    class Tensor:
        def __getitem__(self, _item):
            return self

        def __rsub__(self, _value):
            return self

        def __mul__(self, _value):
            return self

    class Module:
        def __init__(self):
            pass

        def eval(self):
            return self

    torch = SimpleNamespace(
        nn=SimpleNamespace(Module=Module),
        tensor=lambda value, dtype: (value, dtype),
        long="long",
        float32="float32",
    )

    class Config:
        @classmethod
        def from_json_file(cls, path):
            observed["config"] = path
            return cls()

    class Embeddings:
        def __call__(self, **values):
            observed["embeddings"] = values
            return "hidden"

    class Encoded:
        def __getitem__(self, item):
            if item == 0:
                return self
            observed["pooled"] = item
            return "pooled"

    class EncoderStack:
        def __call__(self, hidden, **options):
            observed["encoder"] = (hidden, options)
            return Encoded()

    class Encoder:
        embeddings = Embeddings()
        encoder = EncoderStack()

        def __init__(self, config):
            observed["implementation"] = config._attn_implementation

        def eval(self):
            return self

        def load_state_dict(self, state, strict):
            observed["state"] = (state, strict)
            return [], ["embeddings.position_ids"]

    class Pnnx:
        @staticmethod
        def export(model, path, inputs, fp16):
            observed["export"] = (model, path, inputs, fp16)
            root = Path(path).parent
            (root / "model.ncnn.param").write_text(
                "7767517\n1 2\n"
                "Squeeze squeeze_0 1 1 in0 out0 -23303=1,0\n",
                encoding="utf-8",
            )
            (root / "model.ncnn.bin").write_bytes(b"weights")

    def load_weights(path):
        observed["weights"] = path
        return {"path": path}

    monkeypatch.setattr(
        "omnitensor.training.document_model._document_dependencies",
        lambda: (Pnnx, torch, load_weights, Config, Encoder),
    )
    destination = tmp_path / "nested" / "build"
    output = export_bge_ncnn(source, destination, _tokens())

    assert output == destination / "model.ncnn.param"
    assert observed["config"] == str(source.root / "config.json")
    assert observed["weights"] == str(source.root / "model.safetensors")
    assert observed["implementation"] == "eager"
    assert observed["state"] == (
        {"path": str(source.root / "model.safetensors")},
        False,
    )
    assert observed["export"][1:] == (
        str(destination / "model.pt"),
        (
            ([_tokens().input_ids], "long"),
            ([_tokens().attention_mask], "float32"),
            ([_tokens().token_type_ids], "long"),
        ),
        False,
    )
    mask = np.asarray([[1.0, 0.0, 1.0]], dtype=np.float32)
    assert observed["export"][0].forward("ids", mask, "segments") == "pooled"
    assert observed["embeddings"] == {
        "input_ids": "ids",
        "token_type_ids": "segments",
    }
    assert observed["encoder"][0] == "hidden"
    assert observed["encoder"][1]["return_dict"] is False
    np.testing.assert_array_equal(
        observed["encoder"][1]["attention_mask"],
        np.asarray([[[[0.0, -10000.0, 0.0]]]], dtype=np.float32),
    )
    assert observed["pooled"] == (slice(None), 0, slice(None))
    graph = output.read_text(encoding="utf-8")
    assert "4 6" in graph
    assert "Reduction omnitensor_embedding_norm" in graph
    assert graph.rstrip().endswith("omnitensor_embedding_norm_value out0 0=3")

    with pytest.raises(DocumentModelError) as complete_conflict:
        export_bge_ncnn(source, destination, _tokens())
    _assert_document_error(
        complete_conflict.value,
        "producer-conflict",
        "native model output already exists",
    )

    for filename in ("model.ncnn.param", "model.ncnn.bin"):
        conflict = tmp_path / f"only-{filename.rsplit('.', 1)[-1]}"
        conflict.mkdir()
        (conflict / filename).write_bytes(b"present")
        with pytest.raises(DocumentModelError) as partial_conflict:
            export_bge_ncnn(source, conflict, _tokens())
        _assert_document_error(
            partial_conflict.value,
            "producer-conflict",
            "native model output already exists",
        )


def test_export_rejects_wrong_recipe_weights_compiler_and_incomplete_outputs(  # noqa: C901
    monkeypatch, tmp_path
):
    source = _source(tmp_path)
    wrong_recipe = load_model_recipe(Path("model-recipes/all-minilm-l6-v2.json"))
    wrong = FetchedModelSource(wrong_recipe, source.root, source.receipt_path)
    with pytest.raises(DocumentModelError) as recipe_error:
        export_bge_ncnn(wrong, tmp_path / "wrong-recipe", _tokens())
    _assert_document_error(
        recipe_error.value,
        "recipe-incompatible",
        f"expected {RECIPE_ID}",
    )

    class Module:
        def __init__(self):
            pass

        def eval(self):
            return self

    torch = SimpleNamespace(
        nn=SimpleNamespace(Module=Module),
        tensor=lambda value, dtype: (value, dtype),
        long="long",
        float32="float32",
    )

    class Config:
        @classmethod
        def from_json_file(cls, _path):
            return cls()

    class Encoder:
        embeddings = object()
        encoder = object()
        load_result = ([], ["embeddings.position_ids"])

        def __init__(self, _config):
            pass

        def eval(self):
            return self

        def load_state_dict(self, _state, strict):
            assert strict is False
            return self.load_result

    class Pnnx:
        mode = "complete"

        @classmethod
        def export(cls, _model, path, _inputs, fp16):
            assert fp16 is False
            if cls.mode == "failure":
                raise RuntimeError("compiler broke")
            root = Path(path).parent
            if cls.mode not in {"missing-param", "empty-param"}:
                (root / "model.ncnn.param").write_text(
                    "7767517\n1 2\nSqueeze final 1 1 input out0\n",
                    encoding="utf-8",
                )
            elif cls.mode == "empty-param":
                (root / "model.ncnn.param").touch()
            if cls.mode not in {"missing-bin", "empty-bin"}:
                (root / "model.ncnn.bin").write_bytes(b"weights")
            elif cls.mode == "empty-bin":
                (root / "model.ncnn.bin").touch()

    monkeypatch.setattr(
        "omnitensor.training.document_model._document_dependencies",
        lambda: (Pnnx, torch, lambda _path: {}, Config, Encoder),
    )

    for load_result in ((["missing"], ["embeddings.position_ids"]), ([], [])):
        Encoder.load_result = load_result
        with pytest.raises(DocumentModelError) as weights_error:
            export_bge_ncnn(
                source,
                tmp_path / f"weights-{len(load_result[0])}-{len(load_result[1])}",
                _tokens(),
            )
        _assert_document_error(
            weights_error.value,
            "weights-invalid",
            "safetensors do not match the BGE encoder",
        )

    Encoder.load_result = ([], ["embeddings.position_ids"])
    Pnnx.mode = "failure"
    with pytest.raises(DocumentModelError) as compiler_error:
        export_bge_ncnn(source, tmp_path / "compiler", _tokens())
    _assert_document_error(
        compiler_error.value,
        "compilation-failed",
        "pnnx export failed: compiler broke",
    )

    for mode in ("missing-param", "missing-bin", "empty-param", "empty-bin"):
        Pnnx.mode = mode
        with pytest.raises(DocumentModelError) as incomplete:
            export_bge_ncnn(source, tmp_path / mode, _tokens())
        _assert_document_error(
            incomplete.value,
            "compilation-incomplete",
            "pnnx produced no complete ncnn pair",
        )


@given(
    layer_count=st.integers(min_value=1, max_value=10_000),
    blob_count=st.integers(min_value=2, max_value=20_000),
)
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_native_l2_graph_patch_preserves_counts_and_one_normalized_output(
    tmp_path, layer_count, blob_count
):
    graph = tmp_path / "model.ncnn.param"
    graph.write_text(
        f"7767517\n{layer_count} {blob_count}\n"
        "Squeeze final 1 1 input out0 -23303=1,0\n",
        encoding="utf-8",
    )

    _append_ncnn_l2_normalization(graph)

    lines = graph.read_text(encoding="utf-8").splitlines()
    assert lines[1] == f"{layer_count + 3} {blob_count + 4}"
    assert sum(" out0 " in f" {line} " for line in lines) == 1
    assert lines[-1].split()[0:2] == ["BinaryOp", "omnitensor_embedding_divide"]
    assert lines[-4:] == [
        "Squeeze final 1 1 input omnitensor_embedding_raw -23303=1,0",
        (
            "Split omnitensor_embedding_split 1 2 omnitensor_embedding_raw "
            "omnitensor_embedding_value omnitensor_embedding_norm_input"
        ),
        (
            "Reduction omnitensor_embedding_norm 1 1 omnitensor_embedding_norm_input "
            "omnitensor_embedding_norm_value 0=8 1=0 -23303=1,0 4=1 5=1"
        ),
        (
            "BinaryOp omnitensor_embedding_divide 2 1 omnitensor_embedding_value "
            "omnitensor_embedding_norm_value out0 0=3"
        ),
    ]


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "7767517\nbad header\nSqueeze final 1 1 input out0\n",
        "wrong\n1 2\nSqueeze final 1 1 input out0\n",
        "7767517\n1 2\nConvolution final 1 1 input out0\n",
        "7767517\n1 2\nSqueeze final 1 1 input not_out\n",
        "7767517\n1 2\nSqueeze omnitensor_embedding_existing 1 1 input out0\n",
    ],
)
def test_native_l2_graph_patch_rejects_unknown_or_repeated_boundaries(tmp_path, payload):
    graph = tmp_path / "model.ncnn.param"
    graph.write_text(payload, encoding="utf-8")

    with pytest.raises(DocumentModelError) as caught:
        _append_ncnn_l2_normalization(graph)

    assert caught.value.code == "compilation-incompatible"


def test_native_l2_graph_patch_bounds_and_contains_read_failure(tmp_path):
    oversized = tmp_path / "oversized.param"
    oversized.write_text("x" * (1024 * 1024 + 1), encoding="utf-8")
    with pytest.raises(DocumentModelError) as too_large:
        _append_ncnn_l2_normalization(oversized)
    _assert_document_error(
        too_large.value, "compilation-incompatible", "ncnn graph is oversized"
    )

    with pytest.raises(DocumentModelError) as unreadable:
        _append_ncnn_l2_normalization(tmp_path)
    assert unreadable.value.code == "compilation-incomplete"
    assert unreadable.value.detail.startswith("cannot read ncnn graph:")


def test_document_dependency_loader_returns_the_exact_producer_stack(monkeypatch):
    pnnx = ModuleType("pnnx")
    torch = ModuleType("torch")
    safetensors = ModuleType("safetensors")
    safetensors.__path__ = []
    safetensors_torch = ModuleType("safetensors.torch")
    load_file = object()
    safetensors_torch.load_file = load_file
    transformers = ModuleType("transformers")
    bert_config = object()
    bert_model = object()
    transformers.BertConfig = bert_config
    transformers.BertModel = bert_model
    for name, module in {
        "pnnx": pnnx,
        "torch": torch,
        "safetensors": safetensors,
        "safetensors.torch": safetensors_torch,
        "transformers": transformers,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    assert document_model._document_dependencies() == (
        pnnx,
        torch,
        load_file,
        bert_config,
        bert_model,
    )


def test_install_document_model_publishes_restricted_gpu_binding(  # noqa: C901
    monkeypatch, tmp_path
):
    source = _source(tmp_path)
    observed = {}
    tokenizer = _StaticTokenizer()
    holdout, expected = load_bge_holdout(_bundled_corpus_path())

    def tokenizer_factory(path):
        observed["tokenizer"] = path
        return tokenizer

    monkeypatch.setattr(
        "omnitensor.training.document_model.BgeTokenizer", tokenizer_factory
    )

    def portable_export(_self, received_source, destination):
        observed["portable_export"] = (received_source, destination)
        destination.write_bytes(b"portable")

    monkeypatch.setattr(
        "omnitensor.training.document_model.SentenceEmbeddingOnnxExporter.export",
        portable_export,
    )

    def native_export(received_source, destination, sample):
        observed["native_export"] = (received_source, destination, sample)
        param = destination / "model.ncnn.param"
        param.write_text("graph", encoding="utf-8")
        param.with_suffix(".bin").write_bytes(b"weights")
        return param

    monkeypatch.setattr("omnitensor.training.document_model.export_bge_ncnn", native_export)

    portable_runner = object()

    def portable_factory(path, received_tokenizer):
        observed["portable_runner"] = (path, received_tokenizer)
        return portable_runner

    monkeypatch.setattr(
        "omnitensor.training.document_model.PortableBgeRunner", portable_factory
    )
    native = SimpleNamespace(device_index=1, device_name="RX 6600 XT")

    def native_factory(path, received_tokenizer, *, device_index):
        observed["native_runner"] = (path, received_tokenizer, device_index)
        return native

    monkeypatch.setattr(
        "omnitensor.training.document_model.VulkanBgeRunner", native_factory
    )

    gate = EmbeddingGateEvidence(0.9999, 1.0, 0.0, 28, 2, 12, True)

    def evaluate(received_holdout, received_portable, received_native):
        observed["gate"] = (received_holdout, received_portable, received_native)
        return gate

    monkeypatch.setattr(
        "omnitensor.training.document_model.evaluate_embedding_gate",
        evaluate,
    )

    def maximum_error(received_holdout, received_portable, received_native):
        observed["maximum_error"] = (
            received_holdout,
            received_portable,
            received_native,
        )
        return 0.0002

    monkeypatch.setattr(
        "omnitensor.training.document_model._maximum_embedding_error", maximum_error
    )

    def retrieval(received_holdout, received_expected, received_native):
        observed["retrieval"] = (
            received_holdout,
            received_expected,
            received_native,
        )
        return 2

    monkeypatch.setattr(
        "omnitensor.training.document_model._expected_retrieval_hits", retrieval
    )

    prepare = document_model.prepare_artifact

    def prepare_spy(path, **options):
        observed["prepare"] = (path, options)
        prepared = prepare(path, **options)
        observed["prepared"] = prepared
        return prepared

    monkeypatch.setattr("omnitensor.training.document_model.prepare_artifact", prepare_spy)

    installed_artifact = tmp_path / "installed" / "model.param"

    def install_spy(prepared, root):
        observed["install"] = (prepared, root)
        return installed_artifact

    monkeypatch.setattr(
        "omnitensor.training.document_model.install_prepared",
        install_spy,
    )

    report_document = document_model._report_document

    def report_spy(*args):
        observed["report"] = args
        return report_document(*args)

    monkeypatch.setattr("omnitensor.training.document_model._report_document", report_spy)
    binding_document = document_model._binding_document

    def binding_spy(*args):
        observed["binding"] = args
        return binding_document(*args)

    monkeypatch.setattr("omnitensor.training.document_model._binding_document", binding_spy)
    atomic_write = document_model.write_json_atomic
    observed["writes"] = []

    def write_spy(path, payload, *, prefix):
        observed["writes"].append((path, payload, prefix))
        return atomic_write(path, payload, prefix=prefix)

    monkeypatch.setattr("omnitensor.training.document_model.write_json_atomic", write_spy)

    build_root = tmp_path / "nested" / "build"
    artifact_root = tmp_path / "artifacts"
    bindings_root = tmp_path / "bindings"
    bundled_root = Path("workloads")

    installed = install_document_model(
        source,
        corpus_path=_bundled_corpus_path(),
        build_root=build_root,
        artifact_root=artifact_root,
        bindings_root=bindings_root,
        device_index=1,
        bundled_root=bundled_root,
    )

    assert installed.artifact_path == installed_artifact
    assert installed.binding_path == bindings_root / "document-intelligence/manifest.json"
    assert installed.report_path == build_root / "document-model-report.json"
    assert installed.evidence == DocumentModelEvidence(1, "RX 6600 XT", 28, 0.9999, 1.0, 0.0002, 2)
    assert observed["tokenizer"] == source.root / "tokenizer.json"
    assert observed["portable_export"] == (
        source,
        build_root / "model.reference.onnx",
    )
    assert observed["native_export"] == (source, build_root, _tokens())
    assert observed["portable_runner"] == (
        build_root / "model.reference.onnx",
        tokenizer,
    )
    assert observed["native_runner"] == (
        build_root / "model.ncnn.param",
        tokenizer,
        1,
    )
    assert observed["gate"] == (holdout, portable_runner, native)
    assert observed["maximum_error"] == (holdout, portable_runner, native)
    assert observed["retrieval"] == (holdout, expected, native)
    assert observed["prepare"] == (
        build_root / "model.ncnn.param",
        {
            "artifact_id": "bge-small-en-v1-5-gpu",
            "version": source.recipe.version,
            "model_format": "ncnn",
        },
    )
    assert observed["install"] == (observed["prepared"], artifact_root)
    assert observed["report"][0] is source
    assert observed["report"][2] == observed["prepared"].reference.sha256
    assert observed["report"][3:] == (holdout, installed.evidence)
    assert observed["binding"][0] is source
    assert observed["binding"][1] == observed["prepared"].manifest_fragment()
    assert observed["binding"][2] == observed["report"][1]
    assert observed["binding"][4:] == (installed.evidence, bundled_root)
    assert [(path, prefix) for path, _payload, prefix in observed["writes"]] == [
        (build_root / "document-model-report.json", ".document-model-report-"),
        (
            bindings_root / "document-intelligence/manifest.json",
            ".document-model-binding-",
        ),
    ]
    manifest = json.loads(installed.binding_path.read_text())
    model = manifest["requirements"]["model"]
    assert manifest["requirements"]["acceleratorPreference"] == ["gpu"]
    assert model["tensorContract"] == NATIVE_TENSOR_CONTRACT
    assert model["outputContract"] == {"kind": "embedding"}
    assert model["nativeEvidence"]["namedDeviceAccepted"] is False
    assert installed.evidence.device_name == "RX 6600 XT"
    assert installed.document()["device"] == {"index": 1, "name": "RX 6600 XT"}
    report = json.loads(installed.report_path.read_text())
    assert report["nativeGate"]["accepted"] is True
    assert report["limitations"]["otherProfiles"] == "disabled"


def test_install_document_model_rejects_each_recipe_identity_and_portable_conflict(
    monkeypatch, tmp_path
):
    source = _source(tmp_path)
    for recipe in (
        replace(source.recipe, id="wrong"),
        replace(source.recipe, profile_ids=("hardware-health",)),
    ):
        with pytest.raises(DocumentModelError) as incompatible:
            install_document_model(
                FetchedModelSource(recipe, source.root, source.receipt_path),
                corpus_path=_bundled_corpus_path(),
                build_root=tmp_path / recipe.id,
                artifact_root=tmp_path / "artifacts",
                bindings_root=tmp_path / "bindings",
            )
        _assert_document_error(
            incompatible.value,
            "recipe-incompatible",
            "recipe is not the BGE document model",
        )

    tokenizer = _StaticTokenizer()
    monkeypatch.setattr(
        "omnitensor.training.document_model.BgeTokenizer", lambda _path: tokenizer
    )
    build = tmp_path / "conflict"
    build.mkdir()
    (build / "model.reference.onnx").write_bytes(b"present")
    with pytest.raises(DocumentModelError) as conflict:
        install_document_model(
            source,
            corpus_path=_bundled_corpus_path(),
            build_root=build,
            artifact_root=tmp_path / "artifacts",
            bindings_root=tmp_path / "bindings",
        )
    _assert_document_error(
        conflict.value,
        "producer-conflict",
        "portable reference output already exists",
    )


def test_binding_and_report_are_schema_valid_and_do_not_claim_profile_acceptance(tmp_path):
    source = _source(tmp_path)
    evidence = DocumentModelEvidence(1, "GPU", 28, 0.9999, 1.0, 0.0002, 2)
    report = _report_document(
        source,
        "a" * 64,
        "b" * 64,
        load_bge_holdout(_bundled_corpus_path())[0],
        evidence,
    )
    holdout = load_bge_holdout(_bundled_corpus_path())[0]
    assert report == {
        "reportVersion": 1,
        "kind": "bge-small-document-model",
        "recipeId": RECIPE_ID,
        "recipeVersion": "1.0.0",
        "recipeSha256": source.recipe.document_sha256,
        "sourceReceiptSha256": (
            "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
        ),
        "portableReferenceSha256": "a" * 64,
        "nativeArtifactSha256": "b" * 64,
        "tensorContract": NATIVE_TENSOR_CONTRACT,
        "outputContract": {"kind": "embedding"},
        "holdout": {
            "licenseId": "CC0-1.0",
            "corpusSha256": holdout.corpus_sha256,
            "queryCount": 2,
            "documentCount": 12,
        },
        "device": {"index": 1, "name": "GPU"},
        "nativeGate": {
            "samples": 28,
            "minimumCosineSimilarity": 0.9999,
            "minimumTop10Overlap": 1.0,
            "maximumAbsoluteError": 0.0002,
            "expectedTopHits": 2,
            "accepted": True,
        },
        "limitations": {
            "productionDomainQuality": "not-claimed",
            "namedDeviceProfileAcceptance": False,
            "cpuFallback": "forbidden",
            "otherProfiles": "disabled",
        },
    }
    binding = _binding_document(
        source,
        {
            "id": "bge-gpu",
            "version": "1.0.0",
            "format": "ncnn",
            "sha256": "b" * 64,
            "companions": {"model.bin": "c" * 64},
        },
        "a" * 64,
        "d" * 64,
        evidence,
        None,
    )
    assert validate_manifest(binding) == []
    assert binding["defaults"]["enabled"] is False


def validate_manifest(document):
    from omnitensor.registry import validate_workload_document

    return validate_workload_document(document)


def test_source_lookup_and_cli_success_and_failure(monkeypatch, tmp_path, capsys):
    source = _source(tmp_path)
    assert _source_path(source, "weights").name == "model.safetensors"
    with pytest.raises(DocumentModelError) as missing:
        _source_path(source, "labels")
    _assert_document_error(
        missing.value, "recipe-incompatible", "recipe has no labels source"
    )

    observed = {}
    installed = SimpleNamespace(document=lambda: {"installed": True})
    monkeypatch.setattr(
        "omnitensor.training.document_model.resolve_model_recipe_path",
        lambda value: observed.setdefault("recipe", value),
    )

    def fetch(*args, **kwargs):
        observed["fetch"] = (args, kwargs)
        return source

    def install(*args, **kwargs):
        observed["install"] = (args, kwargs)
        return installed

    monkeypatch.setattr("omnitensor.training.document_model.fetch_model_sources", fetch)
    monkeypatch.setattr("omnitensor.training.document_model.install_document_model", install)
    arguments = [
        "--accept-license",
        "MIT",
        "--source-root",
        str(tmp_path / "sources"),
        "--build-root",
        str(tmp_path / "build"),
        "--artifact-root",
        str(tmp_path / "artifacts"),
        "--bindings-root",
        str(tmp_path / "bindings"),
        "--gpu-device",
        "2",
    ]
    assert main(arguments) == 0
    assert capsys.readouterr().out == '{\n  "installed": true\n}\n'
    assert observed["recipe"] == RECIPE_ID
    assert observed["fetch"] == (
        (RECIPE_ID, tmp_path / "sources"),
        {"accepted_license": "MIT"},
    )
    assert observed["install"] == (
        (source,),
        {
            "corpus_path": _bundled_corpus_path(),
            "build_root": tmp_path / "build",
            "artifact_root": tmp_path / "artifacts",
            "bindings_root": tmp_path / "bindings",
            "device_index": 2,
        },
    )

    observed.clear()
    assert main(["--accept-license", "MIT"]) == 0
    capsys.readouterr()
    assert observed["fetch"] == (
        (RECIPE_ID, Path(document_model.DEFAULT_SOURCE_ROOT).expanduser()),
        {"accepted_license": "MIT"},
    )
    assert observed["install"][1] == {
        "corpus_path": _bundled_corpus_path(),
        "build_root": Path(document_model.DEFAULT_BUILD_ROOT).expanduser(),
        "artifact_root": Path(document_model.DEFAULT_ARTIFACT_ROOT).expanduser(),
        "bindings_root": Path(document_model.DEFAULT_BINDINGS_ROOT).expanduser(),
        "device_index": None,
    }

    monkeypatch.setattr(
        "omnitensor.training.document_model.fetch_model_sources",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            DocumentModelError("offline", "network unavailable")
        ),
    )
    assert main(["--accept-license", "MIT"]) == 1
    assert capsys.readouterr().err == (
        "document model installation failed: offline: network unavailable\n"
    )


def test_document_model_cli_help_is_an_exact_operator_contract(capsys):
    with pytest.raises(SystemExit) as stopped:
        main(["--help"])

    assert stopped.value.code == 0
    output = capsys.readouterr().out
    assert output.startswith(
        "usage: omnitensor-install-document-model [-h] --accept-license ACCEPT_LICENSE"
    )
    assert (
        "Fetch, build, GPU-gate, and install pinned BGE-small for Document Intelligence"
        in output
    )
    for option in (
        "--source-root SOURCE_ROOT",
        "--build-root BUILD_ROOT",
        "--artifact-root ARTIFACT_ROOT",
        "--bindings-root BINDINGS_ROOT",
        "--gpu-device GPU_DEVICE",
    ):
        assert option in output

    with pytest.raises(SystemExit) as missing:
        main([])
    assert missing.value.code == 2
