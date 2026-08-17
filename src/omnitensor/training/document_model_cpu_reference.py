"""Guarded producer-only CPU reference for document-model qualification."""

from __future__ import annotations

from pathlib import Path

from ..document_model_runners import BgeTokenizer
from ..document_model_types import (
    PRODUCER_CPU_REFERENCE_BACKEND,
    DocumentModelError,
)

_LEGACY_MODULE = "omnitensor.training.document_model"
_PRODUCER_CAPABILITY = object()


class PortableBgeRunner:
    """CPU ONNX oracle restricted to the one-shot producer parity gate."""

    backend = PRODUCER_CPU_REFERENCE_BACKEND
    producer_only = True

    def __init__(
        self,
        model: Path,
        tokenizer: BgeTokenizer,
        *,
        _capability=None,
    ):
        if _capability is not _PRODUCER_CAPABILITY:
            raise DocumentModelError(
                "cpu-reference-forbidden",
                "CPU reference execution is restricted to document-model production",
            )
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


def _producer_cpu_reference_runner(model: Path, tokenizer: BgeTokenizer) -> PortableBgeRunner:
    """Construct the CPU oracle for document-model production only."""
    return PortableBgeRunner(
        model,
        tokenizer,
        _capability=_PRODUCER_CAPABILITY,
    )


PortableBgeRunner.__module__ = _LEGACY_MODULE
ProducerCpuBgeReferenceRunner = PortableBgeRunner
