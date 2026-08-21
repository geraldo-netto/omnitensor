# Media transcription

`media-transcription` accepts exactly one explicitly selected, bounded file. It
never scans folders, uses network inference, or treats a CPU codec/parser as an
inference fallback. Whisper and Qwen vision inference require the granted GPU
and an accepted qualification receipt.

Supported inputs:

- audio: WAV, FLAC, MP3, OGG/Vorbis, Opus, and M4A/AAC;
- images: PNG, JPEG, WebP, and safely rasterized SVG;
- video: MP4, WebM, MKV, MOV, AVI, and M4V;
- paged documents: PDF and multipage TIFF;
- text documents: Word `.docx` and plain `.txt`;
- presentations: PowerPoint `.pptx` and LibreOffice Impress `.odp`.

The language of the speech is detected from the first window that carries
audio, and a person who knows better can say so: the `speechLanguage` setting
takes any code the speech model understands, and `auto` — the default — keeps
the detection. It is worth setting for a recording that opens with noise, with
music, or with one borrowed English word, because detection reads that opening
and transcribes everything after it as whatever it decided.

Audio returns timestamped speech. Images return exact visible text plus a
scene description. Videos combine timestamped speech with at most twelve
sampled-frame transcriptions. PDF/TIFF results retain page order. A `.docx` or a `.txt` is answered as one
entry with no page number and no description of a picture: a Word file is
paginated by whatever opens it, and this runtime lays out nothing, so claiming
a page would be claiming where the breaks fall. The description says what the
thing is instead — the same answer a presentation slide carrying no images has
always given. PPTX/ODP
results retain slide order and combine structured slide text with descriptions
of embedded raster images. Legacy binary `.ppt` is not parsed; convert it to
PPTX or ODP with LibreOffice before selecting it.

PyAV supplies the pinned FFmpeg decoding boundary. Pillow handles bounded
raster images, PyMuPDF renders bounded PDF pages — and pypdf reads them when
PyMuPDF cannot load, which happens on machines its compiled wheel does not
support. A PDF read that way is text with no rendered page, so the answer
carries no description of one, and that is visible in the answer rather than
being a quietly smaller result. Word documents are read as OOXML through the
same guarded archive reader the presentations use, and CairoSVG rejects XML
entities and external resources before producing a bounded PNG. The runtime
does not invoke `ffmpeg`, ImageMagick, LibreOffice, or a shell subprocess.
Those independent host tools may be used to author acceptance fixtures.

The provider already has a model-oriented preprocessing layer. Raster images,
sampled video frames, rendered document pages, and presentation images are
decoded to RGB, resized with a high-quality filter so neither dimension exceeds
768 pixels, and encoded as PNG before vision inference. SVG, PDF, and TIFF are
first rendered at no more than 1,600 pixels on their longest side. Audio from
both audio and video containers is decoded directly to mono 16 kHz float PCM
before Whisper inference. It is deliberately not transcoded through MP3:
another lossy encode can erase quiet speech and costs work without reducing the
in-memory tensor. Video is not transcoded wholesale; the adapter extracts the
audio stream and samples at most twelve bounded frames, avoiding unnecessary
decode, storage, and GPU pressure.

## Vulkan runtime installation

The Whisper extension must be built for Vulkan and must find its bundled
native libraries after the build directory disappears and inside the plugin's
Bubblewrap sandbox. A wheel whose `RUNPATH` names `/tmp/.../build` can pass a
direct import on the build host and still fail in production. Build it with a
relative `$ORIGIN` runpath and the service interpreter explicitly selected:

```sh
sudo apt install libvulkan-dev glslc
OMNI_PYTHON="$HOME/.local/share/omnitensor/venv/bin/python"
CMAKE_ARGS="-DGGML_VULKAN=ON -DPython_EXECUTABLE=$OMNI_PYTHON -DPython3_EXECUTABLE=$OMNI_PYTHON -DCMAKE_INSTALL_RPATH=\$ORIGIN -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON" \
  "$OMNI_PYTHON" -m pip wheel --no-deps --no-binary pywhispercpp pywhispercpp==1.5.0
"$OMNI_PYTHON" -m pip install --force-reinstall --no-deps ./pywhispercpp-1.5.0-*.whl
readelf -d "$HOME"/.local/share/omnitensor/venv/lib/python*/site-packages/_pywhispercpp*.so \
  | grep 'Library runpath: \[$ORIGIN\]'
```

Install `av`, `Pillow`, `CairoSVG`, `defusedxml`, and `PyMuPDF` from the
provider wheel's declared dependencies. The host `ffmpeg` and ImageMagick
packages are acceptance-fixture authoring tools, not production inference
dependencies. Authoring also needs Pillow and LibreOffice. Run
`python3 scripts/generate-media-acceptance-fixtures.py /new/absolute/path`;
the generator prefers ImageMagick 7's `magick` executable and falls back to
the legacy `convert` executable.

## Installing the pinned artifacts

`omnitensor-install-media-artifacts` is the only supported way to get the
four model families into the artifact store — two vision models, speech, and
the VulkanOCR enrichment pair: it verifies each source file's exact byte
count and SHA-256 against the pinned release *before* importing anything,
then installs every family atomically. Copying the files into the store by
hand skips that check, and a store entry whose digest does not match its
manifest makes the provider unavailable.

Obtain the ten files first, at these exact revisions (the OCR five come
from `Avafly/PaddleOCR-ncnn-CPP` release v0.3.0, MIT — the same clone the
VulkanOCR README's setup fetches):

| File | Source |
| --- | --- |
| `Qwen3.5-9B-Q4_K_M.gguf` | `unsloth/Qwen3.5-9B-GGUF` at `3885219b6810b007914f3a7950a8d1b469d598a5` — the default vision model (OMNI-0588) |
| `mmproj-F16.gguf` | the same repository and revision |
| `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` | `ggml-org/Qwen2.5-VL-7B-Instruct-GGUF` at `508edd0afaa66bb9e9f40587acc2184f02daf1f6` |
| `mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf` | the same repository and revision |
| `ggml-small.bin` | `ggerganov/whisper.cpp` at `5359861c739e955e79d9a303bcbc70fb988958b1` |

Then install them, naming both licences literally — the command refuses any
other value, because accepting a licence you were not shown is not acceptance:

```sh
OMNI_SERVICE=~/.local/share/omnitensor/venv/bin
OMNI_ARTIFACTS=~/.local/share/omnitensor/artifacts

"$OMNI_SERVICE/omnitensor-install-media-artifacts" \
  --artifact-root "$OMNI_ARTIFACTS" \
  --vision-model /absolute/path/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf \
  --vision-projector /absolute/path/mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf \
  --default-vision-model /absolute/path/Qwen3.5-9B-Q4_K_M.gguf \
  --default-vision-projector /absolute/path/mmproj-F16.gguf \
  --speech-model /absolute/path/ggml-small.bin \
  --ocr-det-param /absolute/path/PP_OCRv6_medium_det.param \
  --ocr-det-bin /absolute/path/PP_OCRv6_medium_det.bin \
  --ocr-rec-param /absolute/path/PP_OCRv6_medium_rec.param \
  --ocr-rec-bin /absolute/path/PP_OCRv6_medium_rec.bin \
  --ocr-dictionary /absolute/path/ppocr_keys_v6.txt \
  --accept-model-license Apache-2.0 \
  --accept-whisper-license MIT
```

Set `OMNITENSOR_ARTIFACT_ROOT` for the service to the same absolute store if it
is not using the default. Never replace a file below the store in place; the
projector is a companion of the vision model, and a changed primary or
companion digest makes the provider unavailable.

All public text is Unicode. A KOI8-R fixture is decoded to Cyrillic Unicode at
fixture creation; raw invalid or undecoded legacy bytes are rejected rather
than silently corrupted. Frozen acceptance inputs include mixed Hebrew,
Cyrillic, Latin, bidirectional layout, charts, embedded images, changing video
frames, lossy/lossless audio, malformed archives, and cancellation cases.

RX 6600 XT qualification is frozen in
`acceptance-evidence/media-transcription-rx6600xt-report.json`. Its digest is
bound into the provider receipt; changing evidence, native runtimes, model
artifacts, or provider version makes startup fail closed. Generating a new
fixture set does not rewrite that historical evidence: the new manifest must
be qualified and archived by a fresh hardware run.
