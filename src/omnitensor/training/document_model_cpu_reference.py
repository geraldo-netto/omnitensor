"""Guarded producer-only CPU reference for document-model qualification."""

from __future__ import annotations

from pathlib import Path

from ..document_model_runners import BgeTokenizer
from ..document_model_types import DocumentModelError

# Build-time only, and named here rather than in the service's core types: it
# identifies the numerical-equivalence oracle a producer runs once, and having
# a CPU backend identifier one import away from the runtime made it read like
# an execution path the service could take. There is none.
PRODUCER_CPU_REFERENCE_BACKEND = "producer-cpu-reference"


class PortableBgeRunner:
    """CPU ONNX oracle for the one-shot producer parity gate.

    **Producer-only, by invariant.** This runs while a model is being built,
    to prove the ncnn export agrees numerically with its ONNX source; the
    service never constructs it, and this module ships in the training
    distribution rather than the runtime one. It was guarded by a module-level
    sentinel passed as a private keyword, which restricted nothing: the
    factory that held it is importable by anyone and re-exported from the
    facade, so the guard only made the honest call site longer.
    """

    backend = PRODUCER_CPU_REFERENCE_BACKEND
    producer_only = True

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


def _producer_cpu_reference_runner(model: Path, tokenizer: BgeTokenizer) -> PortableBgeRunner:
    """Construct the CPU oracle for document-model production only."""
    return PortableBgeRunner(model, tokenizer)


ProducerCpuBgeReferenceRunner = PortableBgeRunner
