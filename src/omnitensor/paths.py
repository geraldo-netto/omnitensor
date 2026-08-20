"""Where this service keeps things, written once.

Every one of these was written out in three to five modules — the artifact
root in `composition`, `document_model_types`, `probe_cli` and `training/cli`,
the published snapshot in `composition` and two training CLIs — so a location
that moved left the copies pointing at a directory nobody writes, and the
failure read as "nothing is installed" rather than as a wrong path. They are
strings rather than `Path`s because most of them are argparse defaults and
environment fallbacks, expanded by the code that uses them.

The state file keeps its `xpu-workload-manager` directory: it is the applet's
own path, published for a reader in another repository, and moving it is a
contract change rather than a tidy-up.
"""

from __future__ import annotations

STATE_DIRECTORY = "~/.local/state/omnitensor"
DATA_DIRECTORY = "~/.local/share/omnitensor"

#: The published runtime snapshot the applet reads.
SNAPSHOT_PATH = "~/.local/state/xpu-workload-manager/state.json"
POLICY_PATH = f"{STATE_DIRECTORY}/policy.json"
GRANTS_PATH = f"{STATE_DIRECTORY}/grants.json"
TELEMETRY_RECORDS_ROOT = f"{STATE_DIRECTORY}/telemetry"

WORKLOADS_ROOT = f"{DATA_DIRECTORY}/workloads"
ARTIFACT_ROOT = f"{DATA_DIRECTORY}/artifacts"
MODEL_BINDINGS_ROOT = f"{DATA_DIRECTORY}/model-bindings"
MODEL_SOURCES_ROOT = f"{DATA_DIRECTORY}/model-sources"
TRAINING_OUTPUT_ROOT = f"{DATA_DIRECTORY}/training"
DOCUMENT_MODEL_BUILD_ROOT = f"{DATA_DIRECTORY}/document-model-build"

__all__ = [
    "ARTIFACT_ROOT",
    "DATA_DIRECTORY",
    "DOCUMENT_MODEL_BUILD_ROOT",
    "GRANTS_PATH",
    "MODEL_BINDINGS_ROOT",
    "MODEL_SOURCES_ROOT",
    "POLICY_PATH",
    "SNAPSHOT_PATH",
    "STATE_DIRECTORY",
    "TELEMETRY_RECORDS_ROOT",
    "TRAINING_OUTPUT_ROOT",
    "WORKLOADS_ROOT",
]
