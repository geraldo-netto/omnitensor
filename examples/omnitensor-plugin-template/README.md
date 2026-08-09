# OmniTensor external plugin template

This directory is an independently buildable Python distribution. It declares
the `omnitensor.workloads` entry point, installs one manifest-v2
`omnitensor-plugin.json`, and imports only `omnitensor.sdk`.

The example accepts one manual payload such as `{"values":[-1,0,1]}`. Its
collector bounds and validates the vector, its pipeline demonstrates typed
preprocessing/postprocessing, and its consumer returns a bounded result. The
sample scoring calculation is a packaging fixture, not an ML model or CPU
fallback. Production plugins declare immutable artifacts and submit inference
through the host runtime.

Build from this directory with `python -m build --wheel`. Install the wheel in
the same Python environment as OmniTensor, then restart the service so metadata
discovery can inspect it. Keep the entry-point name, manifest `id`, and manifest
`plugin.entryPoint` identical.
