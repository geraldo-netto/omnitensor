"""Turn a pretrained model file into an artifact the service will dispatch.

Three profiles need no training data at all — `visual-library` and
`document-intelligence` are retrieval, and `low-light-enhancement` is
enhancement — so their models are downloaded, not fitted.  What was missing was
not the weights but the path: nothing took a file on disk and produced the
digested, declared, installed artifact that dispatch requires.  This is that
path.

The digest is computed from the file rather than supplied alongside it.  A
caller-provided digest only proves the caller can hash, whereas dispatch
verifies against what the manifest declares — so computing it here is what
makes the manifest fragment and the installed bytes describe the same thing by
construction.

Nothing is fetched from the network here.  Which weights to ship is a licensing
decision and who signs them is a key-custody decision, and a tool that quietly
downloaded a model would be making both.  The file is named by the caller, and
this reports exactly what it installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from .plugins.artifact_installation import ArtifactInstaller
from .plugins.artifacts import ArtifactReference

READ_CHUNK_BYTES = 1024 * 1024
SUPPORTED_FORMATS = ("ncnn", "onnx", "openvino", "tflite-edgetpu")
# Which accelerator each format can actually run on. There is no CPU lane by
# design, so a format with no accelerator here cannot be prepared at all.
FORMAT_ACCELERATORS = {
    "ncnn": "gpu",
    "onnx": "gpu",
    "openvino": "npu",
    "tflite-edgetpu": "tpu",
}


class PreparationError(ValueError):
    """Stable preparation failure."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class PreparedArtifact:
    """One model file, described exactly as a manifest must declare it."""

    reference: ArtifactReference
    source: Path
    size_bytes: int

    def manifest_fragment(self) -> dict:
        """The `requirements.model` block a workload manifest needs.

        Emitted rather than written into the manifest: which profile adopts a
        model is a decision for whoever maintains that profile, and silently
        editing a manifest would make a tool the author of a contract.
        """
        return {
            "id": self.reference.id,
            "version": self.reference.version,
            "format": self.reference.format,
            "sha256": self.reference.sha256,
        }

    def document(self) -> dict:
        return {
            "artifact": self.manifest_fragment(),
            "accelerator": FORMAT_ACCELERATORS[self.reference.format],
            "sourcePath": str(self.source),
            "sizeBytes": self.size_bytes,
        }


def file_digest(path: Path) -> str:
    """The sha256 of a file, read in bounded chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(READ_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_artifact(
    source: Path | str,
    *,
    artifact_id: str,
    version: str,
    model_format: str,
) -> PreparedArtifact:
    """Describe a model file as the artifact contract requires."""
    path = Path(source)
    if model_format not in FORMAT_ACCELERATORS:
        raise PreparationError(
            "format-unsupported",
            f"{model_format} is not one of {', '.join(SUPPORTED_FORMATS)}",
        )
    if not isinstance(artifact_id, str) or not 1 <= len(artifact_id) <= 120:
        raise PreparationError("identity-invalid", "artifact id must be a bounded string")
    if not isinstance(version, str) or not version:
        raise PreparationError("identity-invalid", "artifact version is required")
    try:
        if not path.is_file():
            raise PreparationError("source-invalid", f"not a regular file: {path}")
        size = path.stat().st_size
        digest = file_digest(path)
    except OSError as error:
        raise PreparationError("source-invalid", f"cannot read {path}: {error}") from error
    if size == 0:
        raise PreparationError("source-invalid", "an empty file is not a model")
    return PreparedArtifact(
        ArtifactReference(artifact_id, version, model_format, digest), path, size
    )


def install_prepared(prepared: PreparedArtifact, root: Path | str) -> Path:
    """Install a prepared artifact into the store dispatch reads from."""
    installer = ArtifactInstaller(Path(root))
    installation = installer.install(prepared.reference, prepared.source)
    return Path(getattr(installation, "path", root))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a pretrained model file as an OmniTensor artifact"
    )
    parser.add_argument("source", help="the model file to prepare")
    parser.add_argument("--id", required=True, dest="artifact_id")
    parser.add_argument("--version", required=True)
    parser.add_argument("--format", required=True, dest="model_format", choices=SUPPORTED_FORMATS)
    parser.add_argument("--install-root", default=None)
    arguments = parser.parse_args(argv)

    try:
        prepared = prepare_artifact(
            arguments.source,
            artifact_id=arguments.artifact_id,
            version=arguments.version,
            model_format=arguments.model_format,
        )
        document = prepared.document()
        if arguments.install_root:
            document["installedAt"] = str(install_prepared(prepared, arguments.install_root))
    except (PreparationError, OSError, ValueError) as error:
        print(f"preparation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(document, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
