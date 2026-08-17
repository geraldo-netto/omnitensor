"""Deterministic ncnn export for the pinned BGE document model."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ..atomicio import write_bytes_atomic
from ..document_model_types import RECIPE_ID, DocumentModelError, TokenizedText
from .document_model_contracts import source_path
from .recipe_model import FetchedModelSource


def export_bge_ncnn(
    source: FetchedModelSource,
    destination: Path,
    sample: TokenizedText,
    *,
    dependencies: Callable = None,
    source_resolver: Callable[[FetchedModelSource, str], Path] = source_path,
    model_factory: Callable = None,
    graph_normalizer: Callable[[Path], None] = None,
) -> Path:
    """Reconstruct pinned BERT weights and export a fixed precise ncnn graph."""
    if source.recipe.id != RECIPE_ID:
        raise DocumentModelError("recipe-incompatible", f"expected {RECIPE_ID}")
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = destination / "model.pt"
    param = destination / "model.ncnn.param"
    binary = destination / "model.ncnn.bin"
    if any(path.exists() or path.is_symlink() for path in (checkpoint, param, binary)):
        raise DocumentModelError("producer-conflict", "native model output already exists")
    dependency_loader = dependencies or document_dependencies
    pnnx, torch, load_file, bert_config, bert_model = dependency_loader()
    config_path = source_resolver(source, "config")
    weights_path = source_resolver(source, "weights")
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
        model = (model_factory or fixed_bge_model)(torch, encoder)
        pnnx.export(model, str(checkpoint), inputs, fp16=False)
    except Exception as error:  # noqa: BLE001 - pnnx failures vary by graph/toolchain
        raise DocumentModelError("compilation-failed", f"pnnx export failed: {error}") from error
    if (
        not param.is_file()
        or not binary.is_file()
        or param.is_symlink()
        or binary.is_symlink()
        or not param.stat().st_size
        or not binary.stat().st_size
    ):
        raise DocumentModelError("compilation-incomplete", "pnnx produced no complete ncnn pair")
    (graph_normalizer or append_ncnn_l2_normalization)(param)
    return param


def append_ncnn_l2_normalization(param: Path) -> None:
    """Put BGE's declared L2 output meaning inside the native graph."""
    if param.is_symlink():
        raise DocumentModelError(
            "compilation-incomplete", "ncnn graph output must not be a symlink"
        )
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
        write_bytes_atomic(param, ("\n".join(lines) + "\n").encode("utf-8"), 0o600)
    except OSError as error:
        raise DocumentModelError(
            "compilation-incomplete", f"cannot write normalized ncnn graph: {error}"
        ) from error


def document_dependencies():
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


def fixed_bge_model(torch, encoder):
    class FixedBge(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.embeddings = model.embeddings
            self.encoder = model.encoder

        def forward(self, input_ids, attention_mask, token_type_ids):
            hidden = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)
            additive_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0
            return self.encoder(hidden, attention_mask=additive_mask, return_dict=False)[0][:, 0, :]

    return FixedBge(encoder).eval()
