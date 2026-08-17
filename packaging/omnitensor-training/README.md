# omnitensor-training

The producer half of OmniTensor: local training, model recipes, exporters, and
the forecast producers. It is built from the main repository and installs into
the same `omnitensor` package, so `omnitensor.training` imports unchanged.

It is a separate distribution because the serving runtime never imports it. A
desktop that runs the service gets the service; a machine that produces models
installs this on top, in its own environment, with the producer extras it needs:

```
python3 -m venv ~/.local/share/omnitensor-training/venv
~/.local/share/omnitensor-training/venv/bin/pip install \
    '/path/to/omnitensor' 'path/to/omnitensor-training[train]'
```

The pin on `omnitensor==0.1.0` is exact on purpose: model recipes ship here and
the schemas validating them ship with the service, so the two halves are one
version or they are wrong.

Build it from this directory:

```
python -m build --wheel --outdir dist packaging/omnitensor-training
```

Both halves are one uv workspace, so the repository's own tests — which cover
the producer code paths — get their dependencies from the same lock:

```
uv sync --all-packages --extra dev --extra gpu --extra events \
    --extra train --extra model-producers --extra document-producers \
    --extra foundation-producers --extra retinexformer-producers
```

See `docs/local-training.md` for what the trainers do and how to run them.
