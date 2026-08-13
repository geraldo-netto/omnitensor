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
dependencies.

All public text is Unicode. A KOI8-R fixture is decoded to Cyrillic Unicode at
fixture creation; raw invalid or undecoded legacy bytes are rejected rather
than silently corrupted. Frozen acceptance inputs include mixed Hebrew,
Cyrillic, Latin, bidirectional layout, charts, embedded images, changing video
frames, lossy/lossless audio, malformed archives, and cancellation cases.

RX 6600 XT qualification is frozen in
`acceptance-evidence/media-transcription-rx6600xt-report.json`. Its digest is
bound into the provider receipt; changing evidence, native runtimes, model
artifacts, or provider version makes startup fail closed.
