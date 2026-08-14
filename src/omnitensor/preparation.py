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
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .plugins.artifact_installation import (
    ArtifactInstallationError as _ArtifactInstallationError,
)
from .plugins.artifact_installation import ArtifactInstaller
from .plugins.artifact_trust import ArtifactProvenance
from .plugins.artifacts import (
    COMPANION_SOURCE_SUFFIXES,
    ArtifactReference,
    companion_filenames,
)
from .plugins.artifacts import (
    artifact_reference_error as _artifact_reference_error,
)

ArtifactInstallationError = _ArtifactInstallationError
artifact_reference_error = _artifact_reference_error

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


class FileDigestTooLargeError(OSError):
    """A bounded digest observed at least one byte beyond its limit."""

    def __init__(self, max_bytes: int, observed_bytes: int):
        self.max_bytes = max_bytes
        self.observed_bytes = observed_bytes
        super().__init__(f"file exceeds {max_bytes} bytes")


class ArtifactTrustDecision(Protocol):
    """Result shape consumed by artifact trust publication."""

    trusted: bool
    reason: str


class ArtifactTrustVerifier(Protocol):
    """Structural producer boundary accepted by the canonical artifact store."""

    def verify(
        self,
        reference: ArtifactReference,
        provenance: ArtifactProvenance | None,
    ) -> ArtifactTrustDecision: ...

    def verify_installed(
        self,
        reference: ArtifactReference,
        version_root: Path,
    ) -> ArtifactTrustDecision: ...


@dataclass(frozen=True, slots=True)
class PreparedArtifact:
    """One model file, described exactly as a manifest must declare it."""

    reference: ArtifactReference
    source: Path
    size_bytes: int
    companions: Mapping[str, Path] = field(default_factory=dict)

    def manifest_fragment(self) -> dict:
        """The `requirements.model` block a workload manifest needs.

        Emitted rather than written into the manifest: which profile adopts a
        model is a decision for whoever maintains that profile, and silently
        editing a manifest would make a tool the author of a contract.
        """
        fragment = {
            "id": self.reference.id,
            "version": self.reference.version,
            "format": self.reference.format,
            "sha256": self.reference.sha256,
        }
        if self.reference.companions:
            # Printed rather than left to the reader: a format that keeps its
            # weights in a companion refuses to dispatch without these, so a
            # block that omitted them would be a documented way to build a
            # profile that cannot run.
            fragment["companions"] = self.reference.declared_companions
        return fragment

    def document(self) -> dict:
        return {
            "artifact": self.manifest_fragment(),
            "accelerator": FORMAT_ACCELERATORS[self.reference.format],
            "sourcePath": str(self.source),
            "sizeBytes": self.size_bytes,
            "companions": {
                name: str(path) for name, path in sorted(self.companions.items())
            },
        }


def file_digest(path: Path, max_bytes: int | None = None) -> str:
    """Return a complete SHA-256, explicitly refusing bounded overflow."""
    if max_bytes is not None and (
        isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1
    ):
        raise ValueError("max_bytes must be a positive integer")
    digest = hashlib.sha256()
    total = 0
    with Path(path).open("rb") as stream:
        while True:
            read_bytes = READ_CHUNK_BYTES
            if max_bytes is not None:
                read_bytes = min(read_bytes, max_bytes - total + 1)
            block = stream.read(read_bytes)
            if not block:
                break
            total += len(block)
            if max_bytes is not None and total > max_bytes:
                raise FileDigestTooLargeError(max_bytes, total)
            digest.update(block)
    return digest.hexdigest()


LABELS_FILENAME = "labels.txt"


def _labels_companion(labels: Path | str | None) -> dict[str, Path]:
    """A labels file named by the caller, staged like any other companion.

    Named rather than discovered, unlike a weights file: an exporter always
    writes the ``.bin`` beside the ``.param``, and nobody ships a labels file
    at a predictable path. Staging it as a companion means the same digest
    machinery re-verifies it on every resolve, so a label list swapped after
    installation makes the artifact unresolvable instead of quietly renaming
    every result.
    """
    if labels is None:
        return {}
    path = Path(labels)
    if not path.is_file():
        raise PreparationError("labels-invalid", f"not a regular file: {path}")
    return {LABELS_FILENAME: path}


def prepare_artifact(
    source: Path | str,
    *,
    artifact_id: str,
    version: str,
    model_format: str,
    labels: Path | str | None = None,
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
    companions = {**_discover_companions(path, model_format), **_labels_companion(labels)}
    return PreparedArtifact(
        ArtifactReference(
            artifact_id,
            version,
            model_format,
            digest,
            tuple(sorted((name, file_digest(source)) for name, source in companions.items())),
        ),
        path,
        size,
        companions,
    )


def _discover_companions(source: Path, model_format: str) -> dict[str, Path]:
    """Find the files this format needs beside the primary one.

    Discovered rather than asked for: an ncnn ".param" is meaningless without
    the ".bin" the exporter wrote next to it, and requiring the caller to name
    a file that always sits in the same place is a step they can only get
    wrong.
    """
    suffixes = COMPANION_SOURCE_SUFFIXES.get(model_format, {})
    found: dict[str, Path] = {}
    for name in companion_filenames(model_format):
        suffix = suffixes.get(name)
        candidate = source.with_suffix(suffix) if suffix else None
        if candidate is None or not candidate.is_file():
            raise PreparationError(
                "companion-missing",
                f"{model_format} needs {name}; expected it beside the model at "
                f"{candidate if candidate is not None else name}",
            )
        found[name] = candidate
    return found


def install_prepared(prepared: PreparedArtifact, root: Path | str) -> Path:
    """Install a prepared artifact into the store dispatch reads from."""
    installer = ArtifactInstaller(Path(root))
    installation = installer.install(
        prepared.reference, prepared.source, companions=dict(prepared.companions)
    )
    return Path(getattr(installation, "path", root))


def trusted_prepared_installer(
    root: Path | str,
    *,
    trust_verifier: ArtifactTrustVerifier,
) -> Callable[[PreparedArtifact, ArtifactProvenance], Path]:
    """Bind the canonical signed store once for a producer publication batch."""
    destination = Path(root)
    installer = ArtifactInstaller(destination, trust_verifier=trust_verifier)

    def install(prepared: PreparedArtifact, provenance: ArtifactProvenance) -> Path:
        installation = installer.install(
            prepared.reference,
            prepared.source,
            companions=dict(prepared.companions),
            provenance=provenance,
        )
        return Path(getattr(installation, "path", destination))

    return install


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a pretrained model file as an OmniTensor artifact"
    )
    parser.add_argument("source", help="the model file to prepare")
    parser.add_argument("--id", required=True, dest="artifact_id")
    parser.add_argument("--version", required=True)
    parser.add_argument("--format", required=True, dest="model_format", choices=SUPPORTED_FORMATS)
    parser.add_argument("--install-root", default=None)
    parser.add_argument(
        "--labels",
        default=None,
        help="a labels file to install beside the model, one label per line",
    )
    arguments = parser.parse_args(argv)

    try:
        prepared = prepare_artifact(
            arguments.source,
            artifact_id=arguments.artifact_id,
            version=arguments.version,
            model_format=arguments.model_format,
            labels=arguments.labels,
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
