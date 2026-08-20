"""Shared, policy-driven mechanics for operational acceptance workflows."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from re import Pattern

from ..atomicio import JsonTooLargeError, read_bytes_bounded
from .generation import ProviderGenerationError

AcceptanceError = Callable[[str, str], Exception]
DocumentValidator = Callable[[str, object], Sequence[str]]
ReportWriter = Callable[..., None]


class JsonDigestMismatchError(ValueError):
    """The bounded bytes do not match an expected SHA-256 digest."""


@dataclass(frozen=True, slots=True)
class BoundedJsonDocument:
    """JSON, raw bytes, and digest obtained from one bounded descriptor read."""

    document: object
    raw: bytes
    sha256: str


@dataclass(frozen=True, slots=True)
class NativeLoadReport:
    backend: str
    device: str
    total_model_layers: int
    accelerator_layers: int
    cpu_fallback: bool


def read_bounded_json(
    path: Path | str,
    max_bytes: int,
    *,
    expected_sha256: str | None = None,
) -> BoundedJsonDocument:
    """Read, hash, and parse one bounded byte snapshot.

    When an expected digest is supplied, mismatch wins over malformed JSON so
    callers can retain their existing changed-artifact error precedence.
    """
    raw = read_bytes_bounded(Path(path), max_bytes)
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise JsonDigestMismatchError("bounded JSON digest does not match")
    return BoundedJsonDocument(json.loads(raw), raw, digest)


def require_mapping(
    value: object,
    *,
    error_type: AcceptanceError,
    code: str,
    detail: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise error_type(code, detail)
    return value


def require_sequence(
    value: object,
    *,
    error_type: AcceptanceError,
    code: str,
    detail: str,
    allow_empty: bool,
) -> Sequence[object]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or (not allow_empty and not value)
    ):
        raise error_type(code, detail)
    return value


def require_text(
    value: object,
    *,
    error_type: AcceptanceError,
    code: str,
    detail: str,
    maximum: int,
    allow_whitespace: bool,
) -> str:
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= maximum
        or (not allow_whitespace and not value.strip())
    ):
        raise error_type(code, detail)
    return value


def require_match(
    value: object,
    pattern: Pattern[str],
    *,
    error_type: AcceptanceError,
    code: str,
    detail: str,
) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise error_type(code, detail)
    return value


@dataclass(frozen=True, slots=True)
class CaseScore:
    """What one observation earned, with each number named.

    Both acceptance modules returned five positional numbers from their
    scorers — one of them six — so a call site could not be read without
    opening the callee, and the two shapes had quietly diverged.

    `matched`, `extra` and `missing` are the counts the precision and recall
    gates divide; the modules differ in what they count, which is why this
    holds counts rather than ratios.
    """

    matched: int
    extra: int
    missing: int
    latency_ms: int
    peak_memory_bytes: int
    matched_terms: int = 0


# What a digest and an identifier are does not vary by workload, and both
# acceptance modules had defined these identically. A second copy of a regular
# expression is a second place for it to drift.
DIGEST = re.compile(r"^[a-f0-9]{64}$")
IDENTIFIER = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def require_digest(
    value: object,
    *,
    error_type: AcceptanceError,
    code: str,
    detail: str,
) -> str:
    """A lower-case SHA-256, which is the only form a digest is written in here."""
    return require_match(value, DIGEST, error_type=error_type, code=code, detail=detail)


def require_identifier(
    value: object,
    *,
    error_type: AcceptanceError,
    code: str,
    detail: str,
) -> str:
    """A lower-case, hyphen-separated name: a workload id, a case id, a model id."""
    return require_match(value, IDENTIFIER, error_type=error_type, code=code, detail=detail)


def require_integer(
    value: object,
    *,
    error_type: AcceptanceError,
    code: str,
    detail: str,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise error_type(code, detail)
    return value


def require_boolean(
    value: object,
    *,
    error_type: AcceptanceError,
    code: str,
    detail: str,
) -> bool:
    if not isinstance(value, bool):
        raise error_type(code, detail)
    return value


class BoundChecks:
    """The ten bound checks an acceptance module makes, bound to its refusals.

    Each qualified workload had its own copy of these — `_bounded_bytes`,
    `_json_object`, `_mapping`, `_sequence`, `_bounded_text`, `_identifier`,
    `_digest`, `_integer`, `_boolean` — differing only in the error type they
    raise and in a handful of words. Ten wrappers per module is ten places to
    edit whenever a primitive above changes, and the copies had already
    drifted: one allows whitespace inside bounded text and one does not, one
    refuses an empty sequence and one does not, and the same oversize refusal
    reads "exceeds its byte limit" in one and "is oversized" in the other.

    Those differences are real and are contracts a client reads, so they are
    parameters here rather than something to unify by fiat. What they are not
    is a reason to keep two sets of wrappers: the disagreement is now one
    visible constructor call instead of two files that have to be diffed to
    find it.
    """

    # Deliberately not a dataclass: the mutation inventory reads undecorated
    # classes only, and a decorator here would take every check below out of
    # mutation coverage the moment the wrappers moved in.
    __slots__ = (
        "allow_whitespace",
        "digest_detail",
        "error_type",
        "identifier_text_limit",
        "integer_detail",
        "optional_sequence_detail",
        "oversize_detail",
        "sequence_detail",
    )

    def __init__(
        self,
        error_type: AcceptanceError,
        *,
        # The words each module's refusals use: stable text a client may match.
        oversize_detail: str = "exceeds its byte limit",
        sequence_detail: str = "must be a non-empty sequence",
        optional_sequence_detail: str = "must be a sequence",
        digest_detail: str = "must be lowercase SHA-256",
        integer_detail: str = "is outside its bound",
        # Whether bounded text may be blank once stripped.
        allow_whitespace: bool = True,
        # When set, an identifier is first required to be bounded text of this
        # length, so an unbounded value is refused as text, not as a name.
        identifier_text_limit: int | None = None,
    ) -> None:
        self.error_type = error_type
        self.oversize_detail = oversize_detail
        self.sequence_detail = sequence_detail
        self.optional_sequence_detail = optional_sequence_detail
        self.digest_detail = digest_detail
        self.integer_detail = integer_detail
        self.allow_whitespace = allow_whitespace
        self.identifier_text_limit = identifier_text_limit

    def bounded_bytes(self, path: Path | str, maximum: int, label: str) -> bytes:
        try:
            return read_bytes_bounded(Path(path), maximum)
        except JsonTooLargeError as error:
            raise self.error_type(f"{label}-invalid", f"{label} {self.oversize_detail}") from error
        except OSError as error:
            raise self.error_type(f"{label}-invalid", f"cannot read {label}") from error

    def json_object(self, raw: bytes, label: str) -> Mapping[str, object]:
        try:
            value = json.loads(raw)
        except (UnicodeError, ValueError) as error:
            raise self.error_type(f"{label}-invalid", f"{label} is not JSON") from error
        return self.mapping(value, label, f"{label}-invalid")

    def mapping(
        self, value: object, label: str, code: str = "evidence-invalid"
    ) -> Mapping[str, object]:
        return require_mapping(
            value,
            error_type=self.error_type,
            code=code,
            detail=f"{label} must be an object",
        )

    def sequence(
        self,
        value: object,
        label: str,
        code: str = "evidence-invalid",
        *,
        allow_empty: bool = False,
    ) -> Sequence[object]:
        detail = self.optional_sequence_detail if allow_empty else self.sequence_detail
        return require_sequence(
            value,
            error_type=self.error_type,
            code=code,
            detail=f"{label} {detail}",
            allow_empty=allow_empty,
        )

    def text(self, value: object, label: str, maximum: int, code: str) -> str:
        return require_text(
            value,
            error_type=self.error_type,
            code=code,
            detail=f"{label} must be bounded text",
            maximum=maximum,
            allow_whitespace=self.allow_whitespace,
        )

    def identifier(self, value: object, label: str, code: str) -> str:
        if self.identifier_text_limit is not None:
            value = self.text(value, label, self.identifier_text_limit, code)
        return require_identifier(
            value,
            error_type=self.error_type,
            code=code,
            detail=f"{label} must be a kebab-case identifier",
        )

    def digest(self, value: object, label: str, code: str = "evidence-invalid") -> str:
        return require_digest(
            value,
            error_type=self.error_type,
            code=code,
            detail=f"{label} {self.digest_detail}",
        )

    def integer(
        self,
        value: object,
        label: str,
        minimum: int,
        maximum: int | None = None,
        code: str = "evidence-invalid",
    ) -> int:
        return require_integer(
            value,
            error_type=self.error_type,
            code=code,
            detail=f"{label} {self.integer_detail}",
            minimum=minimum,
            maximum=maximum,
        )

    def boolean(self, value: object, label: str, code: str = "evidence-invalid") -> bool:
        return require_boolean(
            value,
            error_type=self.error_type,
            code=code,
            detail=f"{label} must be boolean",
        )


def discover_resource(*ordered_candidates: Path | str) -> Path:
    """Return the first existing candidate, or the final fallback path."""
    if not ordered_candidates:
        raise ValueError("at least one resource candidate is required")
    candidates = tuple(Path(candidate) for candidate in ordered_candidates)
    return next((candidate for candidate in candidates[:-1] if candidate.is_file()), candidates[-1])


def validate_acceptance_report(
    schema_name: str,
    document: dict[str, object],
    *,
    validator: DocumentValidator,
    error_type: AcceptanceError,
) -> dict[str, object]:
    """Fail closed on the first canonical acceptance-schema violation."""
    violations = validator(schema_name, document)
    if violations:
        raise error_type("report-invalid", violations[0])
    return document


def run_acceptance_cli(
    invoke: Callable[[], dict[str, object]],
    output_path: Path | str,
    *,
    writer: ReportWriter,
    output_prefix: str,
    failure_prefix: str,
    error_types: tuple[type[BaseException], ...],
    ensure_ascii: bool,
) -> int:
    """Execute shared acceptance publication without changing parser behavior."""
    try:
        report = invoke()
        writer(Path(output_path), report, prefix=output_prefix)
    except error_types as error:
        print(f"{failure_prefix} failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=ensure_ascii))
    return 0


def validate_gpu_load(report: NativeLoadReport) -> None:
    """Require complete llama.cpp Vulkan model-layer offload."""
    if (
        not isinstance(report, NativeLoadReport)
        or report.backend != "llama.cpp-vulkan"
        or report.device != "Vulkan"
        or report.cpu_fallback
        or report.total_model_layers < 1
        or report.accelerator_layers != report.total_model_layers
    ):
        raise ProviderGenerationError(
            "model-load-failed",
            "llama.cpp did not prove complete Vulkan model-layer offload",
            generation_started=False,
        )


def validate_npu_load(report: NativeLoadReport) -> None:
    """Require OpenVINO GenAI NPU-only execution evidence."""
    if (
        not isinstance(report, NativeLoadReport)
        or report.backend != "openvino-genai"
        or report.device != "NPU"
        or report.cpu_fallback
    ):
        raise ProviderGenerationError(
            "model-load-failed",
            "OpenVINO GenAI did not prove NPU-only execution",
            generation_started=False,
        )


def native_load_report_document(report: NativeLoadReport) -> dict[str, object]:
    """Return the canonical camel-case load evidence document."""
    if not isinstance(report, NativeLoadReport):
        raise TypeError("report must be NativeLoadReport")
    return {
        "backend": report.backend,
        "device": report.device,
        "totalModelLayers": report.total_model_layers,
        "acceleratorLayers": report.accelerator_layers,
        "cpuFallback": report.cpu_fallback,
    }


def parse_native_load_report(
    value: object,
    *,
    error_type: AcceptanceError,
    code: str,
    object_detail: str,
    fields_detail: str,
) -> NativeLoadReport:
    """Parse a closed camel-case load report with family-owned errors."""
    item = require_mapping(
        value,
        error_type=error_type,
        code=code,
        detail=object_detail,
    )
    if set(item) != {
        "backend",
        "device",
        "totalModelLayers",
        "acceleratorLayers",
        "cpuFallback",
    }:
        raise error_type(code, fields_detail)
    return NativeLoadReport(
        require_text(
            item["backend"],
            error_type=error_type,
            code=code,
            detail="backend must be bounded text",
            maximum=120,
            allow_whitespace=False,
        ),
        require_text(
            item["device"],
            error_type=error_type,
            code=code,
            detail="device must be bounded text",
            maximum=120,
            allow_whitespace=False,
        ),
        require_integer(
            item["totalModelLayers"],
            error_type=error_type,
            code=code,
            detail="total layers is invalid",
            minimum=1,
            maximum=65_535,
        ),
        require_integer(
            item["acceleratorLayers"],
            error_type=error_type,
            code=code,
            detail="accelerator layers is invalid",
            minimum=1,
            maximum=65_535,
        ),
        require_boolean(
            item["cpuFallback"],
            error_type=error_type,
            code=code,
            detail="CPU fallback must be boolean",
        ),
    )


__all__ = [
    "BoundedJsonDocument",
    "JsonDigestMismatchError",
    "NativeLoadReport",
    "discover_resource",
    "native_load_report_document",
    "parse_native_load_report",
    "read_bounded_json",
    "require_boolean",
    "require_integer",
    "require_mapping",
    "require_digest",
    "require_identifier",
    "require_match",
    "require_sequence",
    "require_text",
    "run_acceptance_cli",
    "validate_acceptance_report",
    "validate_gpu_load",
    "validate_npu_load",
]
