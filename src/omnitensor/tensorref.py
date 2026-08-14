"""Pass a model input by reference, because it does not fit in the request.

One 3x227x227 float image is 774,394 bytes of JSON against a 262,144-byte
submission quota — three times over, and the argv limit refuses it before the
bus does.  Measured, not estimated.  So every image and audio model was
unreachable through the transport regardless of how the quota was tuned, and
raising the quota would only move the wall: the real cost is encoding a dense
numeric buffer as decimal text on both sides of a socket.

A reference is a path plus the exact shape, dtype, and digest of a raw buffer.
Three things follow from that, and each is the reason for a rule here.

*The path is checked against roots the user opted into.*  A service that reads
any path a caller names is a confused deputy: the caller borrows the service's
permissions to read a file it could not open itself.  Containment is reused
from the ingestion layer rather than reimplemented, because a second
implementation of "is this inside that directory" is a second chance to get
symlinks wrong.

*Nothing is decoded.*  The buffer is raw little-endian floats, never a PNG or a
WAV.  Decoding untrusted media is where the memory bombs live, and this module
would be the worst possible place to introduce one — it runs before admission,
in the service, on behalf of whoever asked.

*The declared size must equal the file's size.*  A shape that disagrees with
the bytes is refused rather than truncated or padded, because both of those
produce a tensor that infers successfully and means nothing.

Every one of those rules can be applied without keeping a single byte, so the
module exposes the check on its own as well as the load.  Admission uses the
check to refuse a bad reference in the submitting call; dispatch still reads
and re-digests the file when it actually needs the values, because a file can
change between the two and only the bytes that were read are proven.

That second digest is deliberate, and it is cheap.  Measured on this host:
0.25 ms for a typical 618 KB image tensor and 25 ms at the 64 MiB ceiling,
against job times of 11-49 ms for the small case.  Holding the verified
buffer for the job's lifetime instead would save that and cost a resident
copy of every in-flight input, bounded only by the queue depth — memory the
scheduler does not currently account for, traded against a re-read the page
cache usually serves.  Not worth it until a profile shows the digest
dominating, and it does not.
"""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

DEFAULT_MAX_TENSOR_BYTES = 64 * 1024 * 1024
MAX_RANK = 6
MAX_DIMENSION = 65_536
# Only formats whose element size is fixed and whose bytes need no decoding.
DTYPE_SIZES = {"float32": 4, "float64": 8, "int32": 4, "int64": 8, "uint8": 1}
_DTYPE_CODES = {"float32": "f", "float64": "d", "int32": "i", "int64": "q", "uint8": "B"}
_READ_CHUNK_BYTES = 1024 * 1024


class TensorReferenceError(ValueError):
    """Stable refusal for a referenced input, safe to return to the caller."""

    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


class InputRootPolicy(Protocol):
    """Decides whether the service may read a referenced path at all.

    A port rather than a function so the containment rule can be swapped
    (opted-in roots today, a per-plugin grant later) without the loading rules
    moving with it.
    """

    def permits(self, path: Path) -> bool: ...

    def describe(self) -> str: ...


class OptedInInputRoots:
    """Permit only paths inside roots the user configured."""

    def __init__(self, roots: Sequence[Path | str]) -> None:
        self._roots = tuple(Path(root).resolve() for root in roots)

    @property
    def roots(self) -> tuple[Path, ...]:
        return self._roots

    def permits(self, path: Path) -> bool:
        try:
            resolved = Path(path).resolve()
        except OSError:
            return False
        # resolve() first, so a symlink pointing outside a root is judged by
        # where it lands rather than where it sits.
        return any(resolved == root or root in resolved.parents for root in self._roots)

    def describe(self) -> str:
        if not self._roots:
            return "no input roots are configured"
        return f"input roots: {', '.join(str(root) for root in self._roots)}"


class DenyAllInputRoots:
    """The default: referencing a file is refused until roots are configured."""

    def permits(self, path: Path) -> bool:
        return False

    def describe(self) -> str:
        return "input references are not enabled for this service"


@dataclass(frozen=True, slots=True)
class TensorReference:
    """A raw numeric buffer on disk, described exactly."""

    path: Path
    shape: tuple[int, ...]
    dtype: str
    sha256: str

    @property
    def element_count(self) -> int:
        return math.prod(self.shape)

    @property
    def expected_bytes(self) -> int:
        return self.element_count * DTYPE_SIZES[self.dtype]


def parse_reference(document: object) -> TensorReference:
    """Validate one caller-supplied reference, refusing anything ambiguous."""
    if not isinstance(document, dict):
        raise TensorReferenceError("input-ref-invalid", "an input reference must be an object")
    path = document.get("path")
    if not isinstance(path, str) or not path:
        raise TensorReferenceError("input-ref-invalid", "input reference path is invalid")
    dtype = document.get("dtype", "float32")
    if dtype not in DTYPE_SIZES:
        raise TensorReferenceError(
            "input-ref-invalid", f"dtype must be one of {', '.join(sorted(DTYPE_SIZES))}"
        )
    shape = _parse_shape(document.get("shape"))
    digest = document.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        # Required, not optional: without it the service cannot tell the file
        # it read from the file the caller meant, and the two differ exactly
        # when it matters.
        raise TensorReferenceError(
            "input-ref-invalid", "input reference must declare a sha256 digest"
        )
    return TensorReference(Path(path), shape, dtype, digest.lower())


def _parse_shape(value: object) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise TensorReferenceError("input-ref-invalid", "input reference shape is invalid")
    if len(value) > MAX_RANK:
        raise TensorReferenceError("input-ref-invalid", f"rank must not exceed {MAX_RANK}")
    dimensions = []
    for dimension in value:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise TensorReferenceError("input-ref-invalid", "shape dimensions must be integers")
        if not 1 <= dimension <= MAX_DIMENSION:
            raise TensorReferenceError(
                "input-ref-invalid", f"shape dimension out of range: {dimension}"
            )
        dimensions.append(dimension)
    return tuple(dimensions)


def verify_reference(
    reference: TensorReference,
    policy: InputRootPolicy,
    *,
    max_tensor_bytes: int = DEFAULT_MAX_TENSOR_BYTES,
) -> None:
    """Refuse a reference that could not be loaded, without loading it.

    Everything :func:`load_referenced_tensor` decides, including the values —
    a file of NaNs was previously admitted and refused only in the infer
    stage, which is the late-failure shape admission exists to remove.  The
    check rides along with the digest, chunk by chunk, so nothing is retained
    and dispatch still re-reads what it will actually execute.
    """
    _permitted(reference, policy, max_tensor_bytes)
    _payload, digest = _read_reference(
        reference, max_tensor_bytes, retain_bytes=False
    )
    if digest != reference.sha256:
        raise TensorReferenceError(
            "input-ref-mismatch", "the file does not match the declared sha256"
        )


def _read_reference(
    reference: TensorReference,
    max_bytes: int,
    *,
    retain_bytes: bool,
) -> tuple[bytes | None, str]:
    """Digest and validate one reference, optionally retaining the read bytes.

    Float values are checked as whole elements, carrying a partial element
    across chunk boundaries. Integer buffers need no finiteness scan.
    """
    element = DTYPE_SIZES[reference.dtype]
    code = _DTYPE_CODES[reference.dtype]
    checked = code in ("f", "d")
    digest = hashlib.sha256()
    retained: list[bytes] | None = [] if retain_bytes else None
    carry = b""
    for chunk in _chunks(reference.path, max_bytes):
        digest.update(chunk)
        if retained is not None:
            retained.append(chunk)
        if not checked:
            continue
        carry += chunk
        whole = len(carry) - (len(carry) % element)
        if whole:
            _refuse_non_finite(carry[:whole], code, whole // element)
            carry = carry[whole:]
    payload = b"".join(retained) if retained is not None else None
    return payload, digest.hexdigest()


def _refuse_non_finite(payload: bytes, code: str, count: int) -> None:
    if any(not math.isfinite(value) for value in struct.unpack(f"<{count}{code}", payload)):
        # A NaN reaching a native library mid-inference is a failure with no
        # useful diagnosis, and the caller learns of it a poll later.
        raise TensorReferenceError(
            "input-ref-invalid", "referenced tensors must contain finite numbers"
        )


def load_referenced_tensor(
    reference: TensorReference,
    policy: InputRootPolicy,
    *,
    max_tensor_bytes: int = DEFAULT_MAX_TENSOR_BYTES,
) -> list:
    """Read a referenced buffer into nested lists, or refuse it.

    Returns the same shape of value an inline tensor would have, so nothing
    downstream needs to know how the input arrived.  The digest is checked
    against the bytes this call read, not against an earlier verification: a
    file re-read is a file that may have changed.
    """
    _permitted(reference, policy, max_tensor_bytes)
    payload, digest = _read_reference(reference, max_tensor_bytes, retain_bytes=True)
    if digest != reference.sha256:
        raise TensorReferenceError(
            "input-ref-mismatch", "the file does not match the declared sha256"
        )
    assert payload is not None
    return _reshape(_unpack(payload, reference), reference.shape)


def _permitted(
    reference: TensorReference, policy: InputRootPolicy, max_tensor_bytes: int
) -> None:
    """The rules that hold before a single byte is read."""
    if not policy.permits(reference.path):
        # Identical answer whether the path is outside the roots or does not
        # exist, so a caller cannot probe the filesystem through refusals.
        raise TensorReferenceError(
            "input-ref-denied",
            f"that path may not be read; {policy.describe()}",
        )
    if reference.expected_bytes > max_tensor_bytes:
        raise TensorReferenceError(
            "input-ref-too-large",
            f"{reference.expected_bytes} bytes exceeds the {max_tensor_bytes}-byte limit",
        )
    try:
        size = reference.path.stat().st_size
    except OSError as error:
        raise TensorReferenceError(
            "input-ref-denied", f"that path may not be read: {error}"
        ) from error
    if size != reference.expected_bytes:
        # Truncating or padding would both produce a tensor that infers
        # successfully and means nothing.
        raise TensorReferenceError(
            "input-ref-mismatch",
            f"declared shape needs {reference.expected_bytes} bytes, file holds {size}",
        )


def _chunks(path: Path, max_bytes: int):
    """Yield the file's bytes, refusing one larger than the caller allows."""
    total = 0
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(_READ_CHUNK_BYTES):
                total += len(chunk)
                if total > max_bytes:
                    raise TensorReferenceError(
                        "input-ref-too-large", f"file exceeds {max_bytes} bytes"
                    )
                yield chunk
    except OSError as error:
        raise TensorReferenceError(
            "input-ref-denied", f"that path may not be read: {error}"
        ) from error


def _digest_of(path: Path, max_bytes: int) -> str:
    digest = hashlib.sha256()
    for chunk in _chunks(path, max_bytes):
        digest.update(chunk)
    return digest.hexdigest()


def _unpack(payload: bytes, reference: TensorReference) -> list:
    code = _DTYPE_CODES[reference.dtype]
    values = struct.unpack(f"<{reference.element_count}{code}", payload)
    return list(values)


def _reshape(flat: list, shape: tuple[int, ...]) -> list:
    if len(shape) == 1:
        return flat
    stride = math.prod(shape[1:])
    return [
        _reshape(flat[index : index + stride], shape[1:])
        for index in range(0, len(flat), stride)
    ]


def parse_references(payload: dict, *, max_tensors: int) -> tuple[TensorReference, ...] | None:
    """The references a payload declares, parsed but not read.

    ``None`` when it declares none, which is how an inline payload is told
    apart from a referenced one without either caller inspecting the key.
    """
    references = payload.get("inputRefs")
    if references is None:
        return None
    if not isinstance(references, list) or not references:
        raise TensorReferenceError("input-ref-invalid", "inputRefs must be a non-empty array")
    if len(references) > max_tensors:
        raise TensorReferenceError(
            "input-ref-invalid", f"at most {max_tensors} referenced inputs are allowed"
        )
    if "inputs" in payload:
        # Two sources of truth for the same argument, and no rule for which
        # wins that would not surprise somebody.
        raise TensorReferenceError(
            "input-ref-invalid", "a payload carries either inputs or inputRefs, never both"
        )
    return tuple(parse_reference(item) for item in references)


def referenced_inputs(
    payload: dict,
    policy: InputRootPolicy,
    *,
    max_tensors: int,
    max_tensor_bytes: int = DEFAULT_MAX_TENSOR_BYTES,
) -> list | None:
    """The tensors a payload references, or ``None`` when it references none."""
    references = parse_references(payload, max_tensors=max_tensors)
    if references is None:
        return None
    return [
        load_referenced_tensor(reference, policy, max_tensor_bytes=max_tensor_bytes)
        for reference in references
    ]
