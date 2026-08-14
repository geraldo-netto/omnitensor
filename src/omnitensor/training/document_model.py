"""Produce and install the pinned BGE-small Document Intelligence model.

The upstream ONNX graph remains the immutable portable reference.  The GPU
artifact is reconstructed independently from the pinned safetensors and
exported directly from PyTorch to ncnn; accepting a converter exit code is not
enough, so both paths must agree on a frozen retrieval corpus while the ncnn
runner is attached to a named non-CPU Vulkan device.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from jsonschema.exceptions import SchemaError

from ..atomicio import JsonTooLargeError, read_json_bounded, write_json_atomic
from ..preparation import file_digest, install_prepared, prepare_artifact
from ..registry import (
    bundled_workloads_path,
    load_workloads,
    validate_document,
    validate_workload_document,
)
from .embedding_production import (
    EmbeddingHoldout,
    SentenceEmbeddingOnnxExporter,
    evaluate_embedding_gate,
    l2_normalize,
)
from .recipes import (
    FetchedModelSource,
    ModelRecipeError,
    fetch_model_sources,
    resolve_model_recipe_path,
)

PROFILE_ID = "document-intelligence"
RECIPE_ID = "bge-small-en-v1-5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
SEQUENCE_LENGTH = 128
MAX_TEXT_BYTES = 2 * 1024 * 1024
MAX_CORPUS_BYTES = 256 * 1024
MIN_NATIVE_COSINE = 0.999
MIN_RETRIEVAL_OVERLAP = 0.9
MAX_NATIVE_ABSOLUTE_ERROR = 0.001
DEFAULT_SOURCE_ROOT = "~/.local/share/omnitensor/model-sources"
DEFAULT_BUILD_ROOT = "~/.local/share/omnitensor/document-model-build"
DEFAULT_ARTIFACT_ROOT = "~/.local/share/omnitensor/artifacts"
DEFAULT_BINDINGS_ROOT = "~/.local/share/omnitensor/model-bindings"
DOCUMENT_MODEL_REPORT_SCHEMA = "document-model-report.schema.json"
NATIVE_TENSOR_CONTRACT = {
    "inputs": [
        {"shape": [1, SEQUENCE_LENGTH], "dtype": "int32", "layout": "NC"},
        {"shape": [1, SEQUENCE_LENGTH], "dtype": "float32", "layout": "NC"},
        {"shape": [1, SEQUENCE_LENGTH], "dtype": "int32", "layout": "NC"},
    ]
}


class DocumentModelError(ValueError):
    """Stable producer failure safe to show to a local operator."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class TokenizedText:
    """Three fixed tensors in the exact order the native graph declares."""

    input_ids: tuple[int, ...]
    attention_mask: tuple[float, ...]
    token_type_ids: tuple[int, ...]

    def inputs(self) -> list[list[list[int | float]]]:
        return [
            [list(self.input_ids)],
            [list(self.attention_mask)],
            [list(self.token_type_ids)],
        ]


@dataclass(frozen=True, slots=True)
class DocumentModelEvidence:
    """Portable/native and retrieval evidence from one named Vulkan device."""

    device_index: int
    device_name: str
    samples: int
    minimum_cosine_similarity: float
    minimum_top10_overlap: float
    maximum_absolute_error: float
    expected_top_hits: int


@dataclass(frozen=True, slots=True)
class InstalledDocumentModel:
    """Paths and evidence published by the one-shot producer."""

    artifact_path: Path
    binding_path: Path
    report_path: Path
    evidence: DocumentModelEvidence

    def document(self) -> dict:
        return {
            "profileId": PROFILE_ID,
            "artifactPath": str(self.artifact_path),
            "bindingPath": str(self.binding_path),
            "reportPath": str(self.report_path),
            "device": {
                "index": self.evidence.device_index,
                "name": self.evidence.device_name,
            },
            "minimumCosineSimilarity": self.evidence.minimum_cosine_similarity,
            "minimumTop10Overlap": self.evidence.minimum_top10_overlap,
            "maximumAbsoluteError": self.evidence.maximum_absolute_error,
            "restartRequired": True,
        }


class BgeTokenizer:
    """Pinned one-shot tokenizer; it never reads or watches the clipboard."""

    def __init__(self, path: Path | str):
        try:
            from tokenizers import Tokenizer  # noqa: PLC0415
        except ImportError as error:  # pragma: no cover - environment-dependent
            raise DocumentModelError(
                "producer-dependency-missing", "install OmniTensor with document-producers"
            ) from error
        try:
            self._tokenizer = Tokenizer.from_file(str(path))
        except Exception as error:  # noqa: BLE001 - tokenizer parser failures vary
            raise DocumentModelError(
                "tokenizer-invalid", f"cannot load tokenizer: {error}"
            ) from error
        self._tokenizer.enable_truncation(max_length=SEQUENCE_LENGTH)
        self._tokenizer.enable_padding(length=SEQUENCE_LENGTH)

    def encode(self, text: str, *, query: bool = False) -> TokenizedText:
        if not isinstance(text, str) or not text.strip():
            raise DocumentModelError("text-invalid", "text must be a non-empty string")
        if len(text.encode("utf-8")) > MAX_TEXT_BYTES:
            raise DocumentModelError("text-too-large", "text exceeds the 2 MiB producer bound")
        value = f"{QUERY_PREFIX}{text}" if query else text
        try:
            encoded = self._tokenizer.encode(value)
        except Exception as error:  # noqa: BLE001 - tokenizer runtime failures vary
            raise DocumentModelError(
                "tokenizer-failed", f"cannot tokenize text: {error}"
            ) from error
        tensors = TokenizedText(
            tuple(encoded.ids),
            tuple(float(value) for value in encoded.attention_mask),
            tuple(encoded.type_ids),
        )
        if not (
            len(tensors.input_ids)
            == len(tensors.attention_mask)
            == len(tensors.token_type_ids)
            == SEQUENCE_LENGTH
        ):
            raise DocumentModelError("tokenizer-invalid", "tokenizer did not emit fixed inputs")
        return tensors


class PortableBgeRunner:
    """Producer-only ONNX reference runner; never selected by the service."""

    def __init__(self, model: Path, tokenizer: BgeTokenizer):
        try:
            import numpy  # noqa: PLC0415
            import onnxruntime  # noqa: PLC0415
        except ImportError as error:  # pragma: no cover - environment-dependent
            raise DocumentModelError(
                "producer-dependency-missing", "install OmniTensor with document-producers"
            ) from error
        self._numpy = numpy
        try:
            self._session = onnxruntime.InferenceSession(
                str(model), providers=["CPUExecutionProvider"]
            )
        except Exception as error:  # noqa: BLE001 - ORT errors vary
            raise DocumentModelError(
                "portable-invalid", f"cannot load portable model: {error}"
            ) from error
        self._tokenizer = tokenizer

    def embed(self, text: str):
        tokens = self._tokenizer.encode(text)
        numpy = self._numpy
        inputs = {
            "input_ids": numpy.asarray([tokens.input_ids], dtype=numpy.int64),
            "attention_mask": numpy.asarray([tokens.attention_mask], dtype=numpy.int64),
            "token_type_ids": numpy.asarray([tokens.token_type_ids], dtype=numpy.int64),
        }
        try:
            output = self._session.run(None, inputs)[0]
        except Exception as error:  # noqa: BLE001 - ORT errors vary
            raise DocumentModelError(
                "portable-failed", f"portable inference failed: {error}"
            ) from error
        return tuple(float(value) for value in output[0])


class VulkanBgeRunner:
    """Precise ncnn runner bound to one named hardware Vulkan device."""

    def __init__(
        self,
        param: Path,
        tokenizer: BgeTokenizer,
        *,
        device_index: int | None = None,
        runtime=None,
    ):
        if runtime is None:
            try:
                import ncnn as runtime  # type: ignore[no-redef]  # noqa: PLC0415
            except ImportError as error:  # pragma: no cover - environment-dependent
                raise DocumentModelError(
                    "producer-dependency-missing", "install OmniTensor with document-producers"
                ) from error
        self._runtime = runtime
        self.device_index, self.device_name = _select_vulkan_device(runtime, device_index)
        self._tokenizer = tokenizer
        self._net = runtime.Net()
        self._net.opt.use_vulkan_compute = True
        # The BERT attention graph loses semantic parity under ncnn's fp16
        # arithmetic/storage defaults.  Its compiled weights remain fp32 and
        # the installed service executor uses these same precise options.
        self._net.opt.use_fp16_packed = False
        self._net.opt.use_fp16_storage = False
        self._net.opt.use_fp16_arithmetic = False
        self._net.set_vulkan_device(self.device_index)
        binary = param.with_suffix(".bin")
        if self._net.load_param(str(param)) != 0 or self._net.load_model(str(binary)) != 0:
            raise DocumentModelError("native-invalid", "cannot load the compiled ncnn model")

    def embed(self, text: str):
        try:
            import numpy  # noqa: PLC0415
        except ImportError as error:  # pragma: no cover - environment-dependent
            raise DocumentModelError(
                "producer-dependency-missing", "install OmniTensor with document-producers"
            ) from error
        tokens = self._tokenizer.encode(text)
        values = (
            numpy.asarray(tokens.input_ids, dtype=numpy.int32),
            numpy.asarray(tokens.attention_mask, dtype=numpy.float32),
            numpy.asarray(tokens.token_type_ids, dtype=numpy.int32),
        )
        extractor = self._net.create_extractor()
        try:
            for index, value in enumerate(values):
                if extractor.input(f"in{index}", self._runtime.Mat(value).clone()) != 0:
                    raise DocumentModelError("native-failed", f"native input {index} was refused")
            code, output = extractor.extract("out0")
            if code != 0:
                raise DocumentModelError("native-failed", "native embedding extraction failed")
            return tuple(float(value) for value in self._array(output))
        finally:
            del extractor

    @staticmethod
    def _array(output):
        try:
            import numpy  # noqa: PLC0415
        except ImportError as error:  # pragma: no cover - environment-dependent
            raise DocumentModelError(
                "producer-dependency-missing", "install OmniTensor with document-producers"
            ) from error
        return numpy.asarray(output).reshape(-1)


def _select_vulkan_device(runtime, requested: int | None) -> tuple[int, str]:
    try:
        count = runtime.get_gpu_count()
    except Exception as error:  # noqa: BLE001 - native loader failures vary
        raise DocumentModelError(
            "device-unavailable", f"cannot enumerate Vulkan: {error}"
        ) from error
    devices = []
    preference = {0: 0, 1: 1, 2: 2}
    for index in range(count):
        info = runtime.get_gpu_info(index)
        kind = info.type()
        if kind in preference:
            name = info.device_name()
            devices.append((preference[kind], index, name))
    if requested is not None:
        matching = [item for item in devices if item[1] == requested]
        if not matching:
            raise DocumentModelError(
                "device-unavailable", f"Vulkan device {requested} is absent or software-only"
            )
        return matching[0][1], matching[0][2]
    if not devices:
        raise DocumentModelError("device-unavailable", "no hardware Vulkan device is available")
    selected = min(devices)
    return selected[1], selected[2]


def export_bge_ncnn(
    source: FetchedModelSource,
    destination: Path,
    sample: TokenizedText,
) -> Path:
    """Reconstruct pinned BERT weights and export a fixed precise ncnn graph."""
    if source.recipe.id != RECIPE_ID:
        raise DocumentModelError("recipe-incompatible", f"expected {RECIPE_ID}")
    pnnx, torch, load_file, bert_config, bert_model = _document_dependencies()
    destination.mkdir(parents=True, exist_ok=True)
    param = destination / "model.ncnn.param"
    binary = destination / "model.ncnn.bin"
    if param.exists() or binary.exists():
        raise DocumentModelError("producer-conflict", "native model output already exists")
    config_path = _source_path(source, "config")
    weights_path = _source_path(source, "weights")
    config = bert_config.from_json_file(str(config_path))
    config._attn_implementation = "eager"
    encoder = bert_model(config).eval()
    state = load_file(str(weights_path))
    missing, unexpected = encoder.load_state_dict(state, strict=False)
    if missing or unexpected != ["embeddings.position_ids"]:
        raise DocumentModelError("weights-invalid", "safetensors do not match the BGE encoder")

    inputs = (
        torch.tensor([sample.input_ids], dtype=torch.long),
        torch.tensor([sample.attention_mask], dtype=torch.float32),
        torch.tensor([sample.token_type_ids], dtype=torch.long),
    )
    try:
        model = _fixed_bge_model(torch, encoder)
        pnnx.export(model, str(destination / "model.pt"), inputs, fp16=False)
    except Exception as error:  # noqa: BLE001 - pnnx failures vary by graph/toolchain
        raise DocumentModelError("compilation-failed", f"pnnx export failed: {error}") from error
    if (
        not param.is_file()
        or not binary.is_file()
        or not param.stat().st_size
        or not binary.stat().st_size
    ):
        raise DocumentModelError("compilation-incomplete", "pnnx produced no complete ncnn pair")
    _append_ncnn_l2_normalization(param)
    return param


def _append_ncnn_l2_normalization(param: Path) -> None:
    """Put BGE's declared L2 output meaning inside the native graph.

    pnnx's current ``torch.nn.functional.normalize`` lowering leaves an
    unsupported ``aten::clamp_min`` custom layer. The encoder's CLS output is
    nonzero for every fixed tokenizer input, so append ncnn's native L2
    reduction and division to the otherwise portable raw graph. Matching the
    exact generated boundary makes a future pnnx graph change fail closed.
    """
    try:
        payload = param.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise DocumentModelError(
            "compilation-incomplete", f"cannot read ncnn graph: {error}"
        ) from error
    if len(payload.encode("utf-8")) > 1024 * 1024:
        raise DocumentModelError("compilation-incompatible", "ncnn graph is oversized")
    lines = payload.splitlines()
    try:
        layer_count, blob_count = (int(value) for value in lines[1].split())
        final = lines[-1].split()
    except (IndexError, TypeError, ValueError) as error:
        raise DocumentModelError(
            "compilation-incompatible", "ncnn graph header is invalid"
        ) from error
    if (
        lines[0] != "7767517"
        or len(final) < 6
        or final[0] != "Squeeze"
        or final[2:4] != ["1", "1"]
        or final[5] != "out0"
        or "omnitensor_embedding_" in payload
    ):
        raise DocumentModelError(
            "compilation-incompatible", "ncnn graph has an unexpected BGE output boundary"
        )
    final[5] = "omnitensor_embedding_raw"
    lines[-1] = " ".join(final)
    lines[1] = f"{layer_count + 3} {blob_count + 4}"
    lines.extend(
        (
            "Split omnitensor_embedding_split 1 2 omnitensor_embedding_raw "
            "omnitensor_embedding_value omnitensor_embedding_norm_input",
            "Reduction omnitensor_embedding_norm 1 1 omnitensor_embedding_norm_input "
            "omnitensor_embedding_norm_value 0=8 1=0 -23303=1,0 4=1 5=1",
            "BinaryOp omnitensor_embedding_divide 2 1 omnitensor_embedding_value "
            "omnitensor_embedding_norm_value out0 0=3",
        )
    )
    try:
        param.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as error:
        raise DocumentModelError(
            "compilation-incomplete", f"cannot write normalized ncnn graph: {error}"
        ) from error


def _document_dependencies():
    try:
        import pnnx  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from safetensors.torch import load_file  # noqa: PLC0415
        from transformers import BertConfig, BertModel  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - environment-dependent
        raise DocumentModelError(
            "producer-dependency-missing", "install OmniTensor with document-producers"
        ) from error
    return pnnx, torch, load_file, BertConfig, BertModel


def _fixed_bge_model(torch, encoder):
    class FixedBge(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.embeddings = model.embeddings
            self.encoder = model.encoder

        def forward(self, input_ids, attention_mask, token_type_ids):
            hidden = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
            additive_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0
            return self.encoder(
                hidden, attention_mask=additive_mask, return_dict=False
            )[0][:, 0, :]

    return FixedBge(encoder).eval()


def _source_path(source: FetchedModelSource, role: str) -> Path:
    item = next((item for item in source.recipe.sources if item.role == role), None)
    if item is None:
        raise DocumentModelError("recipe-incompatible", f"recipe has no {role} source")
    return source.root / item.filename


def load_bge_holdout(path: Path | str) -> tuple[EmbeddingHoldout, tuple[int, ...]]:
    try:
        document = read_json_bounded(Path(path), MAX_CORPUS_BYTES)
    except (OSError, ValueError, JsonTooLargeError) as error:
        raise DocumentModelError("corpus-invalid", f"cannot read BGE corpus: {error}") from error
    fields = {
        "corpusVersion",
        "id",
        "license",
        "provenance",
        "queries",
        "documents",
        "expectedTopDocument",
    }
    if not isinstance(document, dict) or set(document) != fields or document["corpusVersion"] != 1:
        raise DocumentModelError("corpus-invalid", "BGE corpus fields or version are invalid")
    queries = document["queries"]
    documents = document["documents"]
    expected = document["expectedTopDocument"]
    if (
        not isinstance(queries, list)
        or not isinstance(documents, list)
        or not isinstance(expected, list)
        or len(expected) != len(queries)
        or any(type(index) is not int or not 0 <= index < len(documents) for index in expected)
    ):
        raise DocumentModelError("corpus-invalid", "BGE retrieval expectations are invalid")
    if any(not isinstance(text, str) for text in [*queries, *documents]):
        raise DocumentModelError("corpus-invalid", "BGE corpus texts must be strings")
    corpus_hash = hashlib.sha256(
        json.dumps(
            {"queries": queries, "documents": documents},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    try:
        holdout = EmbeddingHoldout(
            tuple(f"{QUERY_PREFIX}{query}" for query in queries),
            tuple(documents),
            document["license"],
            corpus_hash,
        )
    except (TypeError, ValueError) as error:
        raise DocumentModelError("corpus-invalid", str(error)) from error
    return holdout, tuple(expected)


def install_document_model(
    source: FetchedModelSource,
    *,
    corpus_path: Path | str,
    build_root: Path | str,
    artifact_root: Path | str,
    bindings_root: Path | str,
    device_index: int | None = None,
    bundled_root: Path | None = None,
) -> InstalledDocumentModel:
    """Build, gate, digest-install, and bind one GPU BGE-small variant."""
    if source.recipe.id != RECIPE_ID or source.recipe.profile_ids != (PROFILE_ID,):
        raise DocumentModelError("recipe-incompatible", "recipe is not the BGE document model")
    output = Path(build_root)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = BgeTokenizer(_source_path(source, "tokenizer"))
    holdout, expected = load_bge_holdout(corpus_path)
    sample = tokenizer.encode(holdout.queries[0])
    portable = output / "model.reference.onnx"
    if portable.exists():
        raise DocumentModelError("producer-conflict", "portable reference output already exists")
    SentenceEmbeddingOnnxExporter().export(source, portable)
    native = export_bge_ncnn(source, output, sample)
    portable_runner = PortableBgeRunner(portable, tokenizer)
    native_runner = VulkanBgeRunner(
        native, tokenizer, device_index=device_index
    )
    gate = evaluate_embedding_gate(holdout, portable_runner, native_runner)
    maximum_error = _maximum_embedding_error(holdout, portable_runner, native_runner)
    expected_hits = _expected_retrieval_hits(holdout, expected, native_runner)
    if (
        not gate.accepted
        or gate.minimum_cosine_similarity < MIN_NATIVE_COSINE
        or gate.minimum_top10_overlap < MIN_RETRIEVAL_OVERLAP
        or maximum_error > MAX_NATIVE_ABSOLUTE_ERROR
        or expected_hits != len(expected)
    ):
        raise DocumentModelError("native-parity-failed", "BGE native retrieval gate did not pass")
    prepared = prepare_artifact(
        native,
        artifact_id=f"{RECIPE_ID}-gpu",
        version=source.recipe.version,
        model_format="ncnn",
    )
    evidence = DocumentModelEvidence(
        native_runner.device_index,
        native_runner.device_name,
        gate.vector_count,
        gate.minimum_cosine_similarity,
        gate.minimum_top10_overlap,
        maximum_error,
        expected_hits,
    )
    report_path = output / "document-model-report.json"
    portable_sha = file_digest(portable)
    report = _report_document(
        source, portable_sha, prepared.reference.sha256, holdout, evidence
    )
    _write_document_model_report(report_path, report)
    report_sha = file_digest(report_path)
    installed = install_prepared(prepared, artifact_root)
    binding = _binding_document(
        source,
        prepared.manifest_fragment(),
        portable_sha,
        report_sha,
        evidence,
        bundled_root,
    )
    binding_path = Path(bindings_root) / PROFILE_ID / "manifest.json"
    write_json_atomic(binding_path, binding, prefix=".document-model-binding-")
    return InstalledDocumentModel(installed, binding_path, report_path, evidence)


def _maximum_embedding_error(holdout, portable, native) -> float:
    maximum = 0.0
    for text in holdout.queries + holdout.documents:
        left = portable.embed(text)
        right = native.embed(text)
        maximum = max(
            maximum,
            max(abs(a - b) for a, b in zip(left, right, strict=True)),
        )
    return maximum


def _expected_retrieval_hits(holdout, expected, runner) -> int:
    documents = tuple(l2_normalize(runner.embed(text)) for text in holdout.documents)
    hits = 0
    for query, expected_index in zip(holdout.queries, expected, strict=True):
        vector = l2_normalize(runner.embed(query))
        scores = [sum(a * b for a, b in zip(vector, item, strict=True)) for item in documents]
        if max(range(len(scores)), key=lambda index: (scores[index], -index)) == expected_index:
            hits += 1
    return hits


def _report_document(source, portable_sha, native_sha, holdout, evidence) -> dict:
    return {
        "reportVersion": 1,
        "kind": "bge-small-document-model",
        "recipeId": source.recipe.id,
        "recipeVersion": source.recipe.version,
        "recipeSha256": source.recipe.document_sha256,
        "sourceReceiptSha256": file_digest(source.receipt_path),
        "portableReferenceSha256": portable_sha,
        "nativeArtifactSha256": native_sha,
        "tensorContract": copy.deepcopy(NATIVE_TENSOR_CONTRACT),
        "outputContract": {"kind": "embedding"},
        "holdout": {
            "licenseId": holdout.license_id,
            "corpusSha256": holdout.corpus_sha256,
            "queryCount": evidence.samples // 2 - len(holdout.documents),
            "documentCount": len(holdout.documents),
        },
        "device": {"index": evidence.device_index, "name": evidence.device_name},
        "nativeGate": {
            "samples": evidence.samples,
            "minimumCosineSimilarity": evidence.minimum_cosine_similarity,
            "minimumTop10Overlap": evidence.minimum_top10_overlap,
            "maximumAbsoluteError": evidence.maximum_absolute_error,
            "expectedTopHits": evidence.expected_top_hits,
            "accepted": True,
        },
        # This records the measured Vulkan execution route and absence of a
        # fallback route. It deliberately does not claim all-layer residency.
        "cpuFallback": False,
        "limitations": {
            "productionDomainQuality": "not-claimed",
            "namedDeviceProfileAcceptance": False,
            "otherProfiles": "disabled",
        },
    }


def _write_document_model_report(path: Path, document: dict) -> None:
    try:
        violations = validate_document(DOCUMENT_MODEL_REPORT_SCHEMA, document)
    except (OSError, ValueError, SchemaError) as error:
        raise DocumentModelError(
            "report-invalid", f"cannot validate document model report: {error}"
        ) from error
    if violations:
        raise DocumentModelError(
            "report-invalid",
            f"document model report violates schema: {violations[0]}",
        )
    write_json_atomic(path, document, prefix=".document-model-report-")


def _binding_document(
    source, artifact, portable_sha, report_sha, evidence, bundled_root
) -> dict:
    workloads = load_workloads(bundled_root or bundled_workloads_path())
    workload = workloads.get(PROFILE_ID)
    if workload is None:
        raise DocumentModelError("profile-unknown", f"no bundled profile is named {PROFILE_ID}")
    manifest = copy.deepcopy(workload.manifest)
    model = {
        **artifact,
        "fullyQuantized": False,
        "minimumCompilerVersion": "20260526",
        "minimumRuntimeVersion": "1.0.20260526",
        "tensorContract": copy.deepcopy(NATIVE_TENSOR_CONTRACT),
        "outputContract": {"kind": "embedding"},
        "trainingContract": {
            "version": 1,
            "profileId": PROFILE_ID,
            "recipe": RECIPE_ID,
            "reportSha256": report_sha,
            "taskSemanticsSha256": source.recipe.document_sha256,
        },
        "nativeEvidence": {
            "portableSha256": portable_sha,
            "nativeSha256": artifact["sha256"],
            "reportSha256": report_sha,
            "samples": evidence.samples,
            "maximumAbsoluteError": evidence.maximum_absolute_error,
            "tolerance": MAX_NATIVE_ABSOLUTE_ERROR,
            "compilerReportSha256": None,
            "namedDeviceAccepted": False,
        },
    }
    requirements = manifest["requirements"]
    requirements["accelerator"] = "gpu"
    requirements["acceleratorPreference"] = ["gpu"]
    requirements["model"] = model
    requirements.pop("models", None)
    violations = validate_workload_document(manifest)
    if violations:
        raise DocumentModelError("binding-invalid", "; ".join(violations))
    return manifest


def _bundled_corpus_path() -> Path:
    checkout = Path(__file__).resolve().parents[3] / "evaluation-corpora" / f"{RECIPE_ID}.json"
    if checkout.is_file():
        return checkout
    packaged = Path(__file__).resolve().parents[1] / "evaluation-corpora" / f"{RECIPE_ID}.json"
    return packaged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="omnitensor-install-document-model",
        description=(
            "Fetch, build, GPU-gate, and install pinned BGE-small for "
            "Document Intelligence"
        ),
    )
    parser.add_argument("--accept-license", required=True)
    parser.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--build-root", default=DEFAULT_BUILD_ROOT)
    parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--bindings-root", default=DEFAULT_BINDINGS_ROOT)
    parser.add_argument("--gpu-device", type=int, default=None)
    arguments = parser.parse_args(argv)
    try:
        recipe_path = resolve_model_recipe_path(RECIPE_ID)
        source = fetch_model_sources(
            recipe_path,
            Path(arguments.source_root).expanduser(),
            accepted_license=arguments.accept_license,
        )
        installed = install_document_model(
            source,
            corpus_path=_bundled_corpus_path(),
            build_root=Path(arguments.build_root).expanduser(),
            artifact_root=Path(arguments.artifact_root).expanduser(),
            bindings_root=Path(arguments.bindings_root).expanduser(),
            device_index=arguments.gpu_device,
        )
    except (DocumentModelError, ModelRecipeError, OSError, ValueError) as error:
        print(f"document model installation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(installed.document(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
