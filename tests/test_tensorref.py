from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import pytest

from omnitensor.tensorref import (
    MAX_RANK,
    DenyAllInputRoots,
    OptedInInputRoots,
    TensorReferenceError,
    load_referenced_tensor,
    parse_reference,
    parse_references,
    referenced_inputs,
    verify_reference,
)


def buffer_file(tmp_path, values, name="input.f32"):
    path = tmp_path / name
    payload = struct.pack(f"<{len(values)}f", *values)
    path.write_bytes(payload)
    return path, hashlib.sha256(payload).hexdigest()


def reference(path, digest, shape, dtype="float32"):
    return {"path": str(path), "shape": list(shape), "dtype": dtype, "sha256": digest}


def roots(tmp_path):
    return OptedInInputRoots([tmp_path])


def test_a_referenced_buffer_loads_with_the_declared_shape(tmp_path):
    path, digest = buffer_file(tmp_path, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    loaded = load_referenced_tensor(
        parse_reference(reference(path, digest, (2, 3))), roots(tmp_path)
    )

    assert loaded == [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]


def test_a_reference_carries_an_input_too_large_to_inline(tmp_path):
    """The whole point: 3x227x227 does not fit in a submission."""
    values = [0.5] * (3 * 227 * 227)
    path, digest = buffer_file(tmp_path, values)

    loaded = load_referenced_tensor(
        parse_reference(reference(path, digest, (3, 227, 227))), roots(tmp_path)
    )

    assert len(loaded) == 3
    assert len(loaded[0]) == 227
    assert len(loaded[0][0]) == 227
    assert path.stat().st_size < 700_000  # raw, versus 774,394 bytes of JSON


def test_a_path_outside_the_opted_in_roots_is_refused(tmp_path):
    """Reading any path a caller names makes the service a confused deputy."""
    outside = tmp_path.parent / "elsewhere.f32"
    outside.write_bytes(struct.pack("<2f", 1.0, 2.0))
    digest = hashlib.sha256(outside.read_bytes()).hexdigest()
    permitted = tmp_path / "permitted"
    permitted.mkdir()

    with pytest.raises(TensorReferenceError) as failure:
        load_referenced_tensor(
            parse_reference(reference(outside, digest, (2,))), OptedInInputRoots([permitted])
        )

    assert failure.value.code == "input-ref-denied"


def test_a_symlink_escaping_a_root_is_judged_by_where_it_lands(tmp_path):
    outside = tmp_path.parent / "secret.f32"
    outside.write_bytes(struct.pack("<2f", 1.0, 2.0))
    permitted = tmp_path / "permitted"
    permitted.mkdir()
    link = permitted / "innocent.f32"
    link.symlink_to(outside)
    digest = hashlib.sha256(outside.read_bytes()).hexdigest()

    with pytest.raises(TensorReferenceError, match="input-ref-denied"):
        load_referenced_tensor(
            parse_reference(reference(link, digest, (2,))), OptedInInputRoots([permitted])
        )


def test_a_missing_file_and_a_forbidden_one_answer_alike(tmp_path):
    """Otherwise refusals become a way to probe the filesystem."""
    permitted = tmp_path / "permitted"
    permitted.mkdir()
    absent = permitted / "absent.f32"
    forbidden = tmp_path.parent / "forbidden.f32"

    codes = []
    for candidate in (absent, forbidden):
        with pytest.raises(TensorReferenceError) as failure:
            load_referenced_tensor(
                parse_reference(reference(candidate, "a" * 64, (2,))),
                OptedInInputRoots([permitted]),
            )
        codes.append(failure.value.code)

    assert codes == ["input-ref-denied", "input-ref-denied"]


def test_referencing_is_denied_until_roots_are_configured(tmp_path):
    path, digest = buffer_file(tmp_path, [1.0, 2.0])

    with pytest.raises(TensorReferenceError, match="not enabled"):
        load_referenced_tensor(parse_reference(reference(path, digest, (2,))), DenyAllInputRoots())


def test_a_shape_that_disagrees_with_the_file_is_refused(tmp_path):
    """Truncating or padding both yield a tensor that infers and means nothing."""
    path, digest = buffer_file(tmp_path, [1.0, 2.0, 3.0, 4.0])

    with pytest.raises(TensorReferenceError) as failure:
        load_referenced_tensor(
            parse_reference(reference(path, digest, (3, 3))), roots(tmp_path)
        )

    assert failure.value.code == "input-ref-mismatch"
    assert "36 bytes" in failure.value.detail


def test_a_file_that_does_not_match_its_digest_is_refused(tmp_path):
    path, _digest = buffer_file(tmp_path, [1.0, 2.0])

    with pytest.raises(TensorReferenceError, match="declared sha256"):
        load_referenced_tensor(
            parse_reference(reference(path, "b" * 64, (2,))), roots(tmp_path)
        )


def test_a_digest_is_required(tmp_path):
    """Without it the service cannot tell the file it read from the one meant."""
    path, _digest = buffer_file(tmp_path, [1.0, 2.0])
    document = {"path": str(path), "shape": [2], "dtype": "float32"}

    with pytest.raises(TensorReferenceError, match="sha256"):
        parse_reference(document)


def test_a_non_finite_referenced_value_is_refused(tmp_path):
    path, digest = buffer_file(tmp_path, [float("nan"), 1.0])

    with pytest.raises(TensorReferenceError, match="finite"):
        load_referenced_tensor(parse_reference(reference(path, digest, (2,))), roots(tmp_path))


def test_an_oversized_reference_is_refused_before_it_is_read(tmp_path):
    path, digest = buffer_file(tmp_path, [1.0, 2.0])

    with pytest.raises(TensorReferenceError) as failure:
        load_referenced_tensor(
            parse_reference(reference(path, digest, (1000, 1000))),
            roots(tmp_path),
            max_tensor_bytes=1024,
        )

    assert failure.value.code == "input-ref-too-large"


@pytest.mark.parametrize(
    "document",
    [
        "not an object",
        {"shape": [2], "sha256": "a" * 64},
        {"path": "", "shape": [2], "sha256": "a" * 64},
        {"path": "/x", "shape": [], "sha256": "a" * 64},
        {"path": "/x", "shape": "2", "sha256": "a" * 64},
        {"path": "/x", "shape": [0], "sha256": "a" * 64},
        {"path": "/x", "shape": [True], "sha256": "a" * 64},
        {"path": "/x", "shape": [99999], "sha256": "a" * 64},
        {"path": "/x", "shape": [1] * (MAX_RANK + 1), "sha256": "a" * 64},
        {"path": "/x", "shape": [2], "dtype": "complex128", "sha256": "a" * 64},
        {"path": "/x", "shape": [2], "sha256": "short"},
    ],
)
def test_a_malformed_reference_is_refused(document):
    with pytest.raises(TensorReferenceError, match="input-ref-invalid"):
        parse_reference(document)


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64", "uint8"])
def test_every_supported_dtype_round_trips(tmp_path, dtype):
    import struct as s

    codes = {"float32": "f", "float64": "d", "int32": "i", "int64": "q", "uint8": "B"}
    payload = s.pack(f"<2{codes[dtype]}", 1, 2)
    path = tmp_path / f"input.{dtype}"
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    loaded = load_referenced_tensor(
        parse_reference(reference(path, digest, (2,), dtype)), roots(tmp_path)
    )

    assert loaded == [1, 2]


def test_a_payload_referencing_nothing_is_left_alone():
    assert referenced_inputs({"inputs": [[1.0]]}, DenyAllInputRoots(), max_tensors=4) is None


def test_a_payload_carrying_both_forms_is_refused(tmp_path):
    """Two sources of truth for one argument, and no unsurprising winner."""
    path, digest = buffer_file(tmp_path, [1.0, 2.0])
    payload = {"inputs": [[1.0]], "inputRefs": [reference(path, digest, (2,))]}

    with pytest.raises(TensorReferenceError, match="never both"):
        referenced_inputs(payload, roots(tmp_path), max_tensors=4)


def test_the_number_of_referenced_tensors_is_bounded(tmp_path):
    path, digest = buffer_file(tmp_path, [1.0, 2.0])
    payload = {"inputRefs": [reference(path, digest, (2,))] * 5}

    with pytest.raises(TensorReferenceError, match="at most 2"):
        referenced_inputs(payload, roots(tmp_path), max_tensors=2)


@pytest.mark.parametrize("value", [[], {}, "refs", None])
def test_an_unusable_inputrefs_field_is_refused(tmp_path, value):
    if value is None:
        pytest.skip("absent inputRefs is the no-reference case")
    with pytest.raises(TensorReferenceError, match="input-ref-invalid"):
        referenced_inputs({"inputRefs": value}, roots(tmp_path), max_tensors=4)


def test_referenced_inputs_load_as_a_list_of_tensors(tmp_path):
    path, digest = buffer_file(tmp_path, [1.0, 2.0, 3.0, 4.0])

    loaded = referenced_inputs(
        {"inputRefs": [reference(path, digest, (2, 2))]}, roots(tmp_path), max_tensors=4
    )

    assert loaded == [[[1.0, 2.0], [3.0, 4.0]]]


def test_the_policy_describes_what_it_permits(tmp_path):
    assert str(tmp_path) in OptedInInputRoots([tmp_path]).describe()
    assert "no input roots" in OptedInInputRoots([]).describe()
    assert "not enabled" in DenyAllInputRoots().describe()
    assert OptedInInputRoots([tmp_path]).roots == (Path(tmp_path).resolve(),)


def test_an_unreadable_file_is_refused_as_denied(tmp_path, monkeypatch):
    path, digest = buffer_file(tmp_path, [1.0, 2.0])
    original = Path.open

    def explode(self, *args, **kwargs):
        if self.name == path.name:
            raise OSError("device error")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", explode)

    with pytest.raises(TensorReferenceError, match="input-ref-denied"):
        load_referenced_tensor(parse_reference(reference(path, digest, (2,))), roots(tmp_path))


def test_an_unresolvable_path_is_denied_rather_than_raising(tmp_path, monkeypatch):
    """A path that cannot even be resolved must not escape the policy."""
    policy = OptedInInputRoots([tmp_path])

    def explode(self, *args, **kwargs):
        raise OSError("too many levels of symbolic links")

    monkeypatch.setattr(Path, "resolve", explode)

    assert policy.permits(tmp_path / "x.f32") is False


def test_a_file_that_changed_size_after_the_check_is_refused(tmp_path):
    """A stat only describes the file as it was, so the read is bounded too."""
    path, digest = buffer_file(tmp_path, [1.0, 2.0])
    document = parse_reference(reference(path, digest, (2,)))
    path.write_bytes(b"x" * 5000)

    with pytest.raises(TensorReferenceError, match="input-ref-mismatch"):
        load_referenced_tensor(document, roots(tmp_path))


def test_a_file_that_grows_between_the_check_and_the_read_is_still_bounded(
    tmp_path, monkeypatch
):
    """Simulates the race the size check alone cannot close."""
    path, digest = buffer_file(tmp_path, [1.0, 2.0])
    document = parse_reference(reference(path, digest, (2,)))
    real_stat = Path.stat

    def stale_stat(self, *args, **kwargs):
        result = real_stat(self, *args, **kwargs)
        if self.name == path.name:
            # Report the size the caller declared while the file is larger.
            return type("Stat", (), {"st_size": document.expected_bytes})()
        return result

    path.write_bytes(b"x" * 5000)
    monkeypatch.setattr(Path, "stat", stale_stat)

    with pytest.raises(TensorReferenceError, match="input-ref-too-large"):
        load_referenced_tensor(document, roots(tmp_path), max_tensor_bytes=64)


def test_verification_accepts_a_reference_without_reading_it_into_a_tensor(tmp_path):
    """Admission needs the verdict, not the buffer."""
    path, digest = buffer_file(tmp_path, [1.0, 2.0, 3.0, 4.0])

    assert (
        verify_reference(parse_reference(reference(path, digest, (2, 2))), roots(tmp_path))
        is None
    )


def test_verification_refuses_a_digest_that_does_not_match(tmp_path):
    """A caller must not learn at result time what submission could have said."""
    path, _digest = buffer_file(tmp_path, [1.0, 2.0, 3.0, 4.0])

    with pytest.raises(TensorReferenceError) as failure:
        verify_reference(parse_reference(reference(path, "0" * 64, (4,))), roots(tmp_path))

    assert failure.value.code == "input-ref-mismatch"
    assert "sha256" in failure.value.detail


def test_verification_refuses_a_shape_that_disagrees_with_the_file(tmp_path):
    path, digest = buffer_file(tmp_path, [1.0, 2.0, 3.0, 4.0])

    with pytest.raises(TensorReferenceError) as failure:
        verify_reference(parse_reference(reference(path, digest, (3, 2))), roots(tmp_path))

    assert failure.value.code == "input-ref-mismatch"
    assert "24 bytes" in failure.value.detail


def test_verification_refuses_a_path_outside_the_roots(tmp_path):
    path, digest = buffer_file(tmp_path, [1.0, 2.0])

    with pytest.raises(TensorReferenceError, match="input-ref-denied"):
        verify_reference(parse_reference(reference(path, digest, (2,))), DenyAllInputRoots())


def test_verification_refuses_a_tensor_larger_than_the_limit(tmp_path):
    path, digest = buffer_file(tmp_path, [1.0, 2.0, 3.0, 4.0])

    with pytest.raises(TensorReferenceError, match="input-ref-too-large"):
        verify_reference(
            parse_reference(reference(path, digest, (2, 2))),
            roots(tmp_path),
            max_tensor_bytes=8,
        )


def test_verification_refuses_a_missing_file(tmp_path):
    with pytest.raises(TensorReferenceError, match="input-ref-denied"):
        verify_reference(
            parse_reference(reference(tmp_path / "absent.f32", "0" * 64, (2,))),
            roots(tmp_path),
        )


def test_parsing_references_reads_nothing_from_disk(tmp_path):
    """Parsing is separate from reading so admission can bound the work it does."""
    payload = {"inputRefs": [reference(tmp_path / "never-created.f32", "0" * 64, (2,))]}

    [parsed] = parse_references(payload, max_tensors=4)

    assert parsed.shape == (2,)
    assert parsed.expected_bytes == 8


def test_parsing_returns_none_for_an_inline_payload():
    assert parse_references({"inputs": [[1.0]]}, max_tensors=4) is None


def test_parsing_refuses_a_payload_carrying_both_forms(tmp_path):
    path, digest = buffer_file(tmp_path, [1.0])
    payload = {"inputs": [[1.0]], "inputRefs": [reference(path, digest, (1,))]}

    with pytest.raises(TensorReferenceError, match="never both"):
        parse_references(payload, max_tensors=4)
