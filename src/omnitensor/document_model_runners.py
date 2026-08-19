"""Tokenizer and native Vulkan runner for the document model."""

from __future__ import annotations

from pathlib import Path

from omnitensor.executors.vulkan import VulkanSelectionError, select_vulkan_device

from .document_model_types import (
    MAX_TEXT_BYTES,
    QUERY_PREFIX,
    SEQUENCE_LENGTH,
    DocumentModelError,
    TokenizedText,
)


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
                    "producer-dependency-missing",
                    "install OmniTensor with document-producers",
                ) from error
        self._runtime = runtime
        self.device_index, self.device_name = select_document_vulkan_device(runtime, device_index)
        self._tokenizer = tokenizer
        self._net = runtime.Net()
        self._net.opt.use_vulkan_compute = True
        self._net.opt.use_fp16_packed = False
        self._net.opt.use_fp16_storage = False
        self._net.opt.use_fp16_arithmetic = False
        self._net.set_vulkan_device(self.device_index)
        binary = param.with_suffix(".bin")
        if self._net.load_param(str(param)) != 0 or self._net.load_model(str(binary)) != 0:
            raise DocumentModelError("native-invalid", "cannot load the compiled ncnn model")

    def close(self) -> None:
        """Release the network and the Vulkan memory behind it."""
        self._net.clear()

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


def select_document_vulkan_device(runtime, requested: int | None) -> tuple[int, str]:
    try:
        selected = select_vulkan_device(runtime, requested)
    except VulkanSelectionError as error:
        if error.kind == "enumeration":
            detail = f"cannot enumerate Vulkan: {error.reason}"
        elif error.kind == "requested":
            detail = error.reason
        else:
            detail = "no hardware Vulkan device is available"
        raise DocumentModelError("device-unavailable", detail) from error
    return selected.index, selected.name
