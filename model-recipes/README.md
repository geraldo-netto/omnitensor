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

MiniLM and BGE have a portable producer API in
`omnitensor.training.embedding_production`. It appends their exact fixed
pooling/L2 graph and gates portable output against the verified source on
operator-reviewed local text. BGE additionally has the explicit
`omnitensor-install-document-model` producer: it reconstructs the same pinned
encoder from safetensors, exports a precise ncnn graph, runs retrieval and
portable/native parity on a named hardware Vulkan device, installs the
digest-locked pair, and writes a restricted local binding. NPU and TPU remain
unqualified, as does MiniLM on every native target.

CLIP has the same fail-closed boundary in
`omnitensor.training.clip_production`: exact cover-resize/normalization helpers,
an image-only ONNX exporter with in-graph L2 normalization, and local
cosine/zero-shot source parity. It does not export the text encoder or qualify
an accelerator.

Chronos-Bolt-Tiny and Granite TTM R2 have a shared producer boundary in
`omnitensor.training.foundation_forecast_production`. A reviewed,
family-specific loader reconstructs the pinned source; the exporter freezes a
512-value context and selects the recipe's exact first-horizon scalar inside
ONNX. Publication requires source/export error at most `0.0001`, at least 35%
skill over repeat-last, and strictly lower MAE than both repeat-last and the
existing local linear forecaster on 32–512 time-ordered, operator-owned samples.
The report records only corpus identity and aggregate error. It never contains
telemetry values and never qualifies a compiler or device.

The bundled catalog currently contains:

| recipe | upstream portable source | intended output | target status |
| --- | --- | --- | --- |
| `bge-small-en-v1-5` | pinned official ONNX graph, tokenizer, config, and safetensors | 384-value sentence embedding | GPU one-shot producer; TPU/NPU planned |
| `all-minilm-l6-v2` | pinned official ONNX graph, tokenizer, and config | 384-value sentence embedding | TPU/NPU/GPU planned |
| `clip-vit-b-32-image` | pinned official OpenAI TorchScript checkpoint | 512-value image embedding | TPU/NPU/GPU planned |
| `ibm-granite-ttm-r2` | pinned IBM Tiny Time Mixer R2 safetensors checkpoint | first-horizon point forecast | TPU/NPU/GPU planned |
| `amazon-chronos-bolt-tiny` | pinned Amazon Chronos-Bolt-Tiny safetensors checkpoint | first-horizon median forecast | TPU/NPU/GPU planned |
| `retinexformer-lol-v1` | official LOL-v1 state dict plus pinned architecture and config | clamped 256x256 enhanced RGB image | TPU/NPU/GPU planned |

Each `producer` section states the graph boundary between upstream bytes and a
deployable artifact. BGE uses first-token (`[CLS]`) selection plus L2 normalization; MiniLM uses
attention-mask mean pooling plus L2 normalization. CLIP uses an image-only
encoder export plus L2 normalization. These operations are part of model
meaning, so a native compiler may not omit or reimplement them differently.

The existing locally fitted linear forecaster remains the working, private
baseline: it trains from opted-in host history, exports directly to ONNX, and
must beat repeat-last on a time-ordered holdout. TTM and Chronos-Bolt are
optional upstream sources, not installed models. Their common producer accepts
one finite 512-observation window and emits one first-horizon scalar. TTM
selects its first point forecast, while Chronos-Bolt selects the first-horizon
0.5 quantile. Keeping that selection inside the portable graph makes target
variants comparable.

Retinexformer uses the authors' official MIT-licensed LOL-v1 checkpoint. Its
immutable Drive file id, digest, byte size, architecture, and training config
are all part of the recipe. `PinnedRetinexformerLoader` executes only that
verified architecture, instantiates the exact one-stage 40-feature layout, and
strictly loads the checkpoint's `params`. The producer fixes RGB preprocessing
and resolution, clamps the enhanced image inside ONNX, and refuses publication
unless separately licensed paired images pass source/export and target SSIM
plus range gates. It is not the profile's existing tonal-curve artifact and
cannot be installed in its place until a target artifact and matching consumer
pass their fidelity and device gates.

`zero-dce` is deliberately refused by the fetch command. The official project
limits its implementation and pretrained model to academic, non-commercial
use under CC-BY-NC-4.0, which conflicts with this catalog's redistributable
commercial-use policy. A third-party conversion cannot relicense those bytes.

Except for BGE's fail-closed local GPU producer, target claims in this catalog
remain `planned`: source portability is not target acceptance. BGE is marked
`convertible`, not prequalified; each invocation must still pass conversion,
retrieval, parity, and named-device execution before it publishes a binding.
NPU targets require their own compiler and device evidence. TPU additionally
requires representative fully-int8 calibration, a compiler report with full
Edge TPU mapping, semantic parity, and execution evidence on Coral hardware.
