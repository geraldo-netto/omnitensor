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
- presentations: PowerPoint `.pptx` and LibreOffice Impress `.odp`.

Audio returns timestamped speech. Images return exact visible text plus a
scene description. Videos combine timestamped speech with at most twelve
sampled-frame transcriptions. PDF/TIFF results retain page order. PPTX/ODP
results retain slide order and combine structured slide text with descriptions
of embedded raster images. Legacy binary `.ppt` is not parsed; convert it to
PPTX or ODP with LibreOffice before selecting it.

PyAV supplies the pinned FFmpeg decoding boundary. Pillow handles bounded
raster images, PyMuPDF renders bounded PDF pages, and CairoSVG rejects XML
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

`omnitensor-install-media-artifacts` is the only supported way to get the two
model families into the artifact store: it verifies each source file's exact
byte count and SHA-256 against the pinned release *before* importing anything,
then installs both families atomically. Copying the files into the store by
hand skips that check, and a store entry whose digest does not match its
manifest makes the provider unavailable.

Obtain the three files first, at these exact revisions:

| File | Source |
| --- | --- |
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
  --speech-model /absolute/path/ggml-small.bin \
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
