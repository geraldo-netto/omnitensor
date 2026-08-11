# Portable model recipes

This directory contains reviewed, immutable source-model declarations. A
recipe is not an executable service artifact and a target marked `planned` or
`convertible` is not runtime acceptance evidence. Use the explicit offline
`omnitensor-fetch-model-source` producer command; the service never downloads
these files.

List the installed catalog with `omnitensor-list-model-recipes`. Fetch by
reviewed id or by an explicit local recipe path, and accept the weight license
separately from the OmniTensor code license:

```console
omnitensor-fetch-model-source bge-small-en-v1-5 --accept-license MIT
```

The bundled embedding catalog currently contains:

| recipe | upstream portable source | intended output | target status |
| --- | --- | --- | --- |
| `bge-small-en-v1-5` | pinned official ONNX graph, tokenizer, and config | 384-value sentence embedding | TPU/NPU/GPU planned |
| `all-minilm-l6-v2` | pinned official ONNX graph, tokenizer, and config | 384-value sentence embedding | TPU/NPU/GPU planned |
| `clip-vit-b-32-image` | pinned official OpenAI TorchScript checkpoint | 512-value image embedding | TPU/NPU/GPU planned |

Each `producer` section states the graph boundary between upstream bytes and a
deployable artifact. BGE uses first-token (`[CLS]`) selection plus L2 normalization; MiniLM uses
attention-mask mean pooling plus L2 normalization. CLIP uses an image-only
encoder export plus L2 normalization. These operations are part of model
meaning, so a native compiler may not omit or reimplement them differently.

All three targets remain `planned`: source portability is not target
acceptance. NPU and GPU still require the producer export, conversion, parity,
and named-device execution evidence. TPU additionally requires representative
fully-int8 calibration, a compiler report with full Edge TPU mapping, semantic
parity, and execution evidence on Coral hardware.
