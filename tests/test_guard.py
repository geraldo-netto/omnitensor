from __future__ import annotations

import json

import pytest

from omnitensor.guard import (
    GUARD_ERROR_VERSION,
    BusGuard,
    GuardRefusedError,
    MethodQuota,
    asserted_identity_field,
    guarded,
)


class Clock:
    def __init__(self):
        self.now = 1_000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def guard(quotas=None, **changes):
    clock = Clock()
    return BusGuard(quotas, clock=clock, **changes), clock


def test_a_call_within_quota_is_admitted():
    subject, _clock = guard()
    subject.admit("SubmitJob", "uid:1000", "{}")
    subject.release("SubmitJob", "uid:1000")


def test_a_caller_over_its_rate_is_refused_with_a_stable_code():
    subject, _clock = guard({"SubmitJob": MethodQuota(max_calls=2, max_concurrent=99)})

    subject.admit("SubmitJob", "uid:1000", "{}")
    subject.admit("SubmitJob", "uid:1000", "{}")

    with pytest.raises(GuardRefusedError) as refusal:
        subject.admit("SubmitJob", "uid:1000", "{}")
    assert refusal.value.code == "rate-limit-exceeded"
    assert refusal.value.method == "SubmitJob"


def test_one_noisy_caller_does_not_lock_out_another():
    """A global limit turns protection into a denial-of-service primitive."""
    subject, _clock = guard({"SubmitJob": MethodQuota(max_calls=1, max_concurrent=99)})

    subject.admit("SubmitJob", "uid:1000", "{}")
    with pytest.raises(GuardRefusedError):
        subject.admit("SubmitJob", "uid:1000", "{}")

    subject.admit("SubmitJob", "uid:1001", "{}")


def test_a_quota_is_per_method_not_shared_across_them():
    subject, _clock = guard({"SubmitJob": MethodQuota(max_calls=1, max_concurrent=99)})
    subject.admit("SubmitJob", "uid:1000", "{}")
    subject.admit("CancelJob", "uid:1000", "{}")


def test_the_rate_window_slides():
    subject, clock = guard({"SubmitJob": MethodQuota(max_calls=1, window_seconds=10)})
    subject.admit("SubmitJob", "uid:1000", "{}")

    clock.advance(11)

    subject.admit("SubmitJob", "uid:1000", "{}")


def test_calls_still_inside_the_window_keep_counting():
    subject, clock = guard({"SubmitJob": MethodQuota(max_calls=1, window_seconds=10)})
    subject.admit("SubmitJob", "uid:1000", "{}")

    clock.advance(9)

    with pytest.raises(GuardRefusedError, match="rate-limit-exceeded"):
        subject.admit("SubmitJob", "uid:1000", "{}")


def test_concurrent_calls_are_bounded_and_released():
    subject, _clock = guard({"SubmitJob": MethodQuota(max_concurrent=2, max_calls=99)})
    subject.admit("SubmitJob", "uid:1000", "{}")
    subject.admit("SubmitJob", "uid:1000", "{}")

    with pytest.raises(GuardRefusedError, match="concurrency-limit-exceeded"):
        subject.admit("SubmitJob", "uid:1000", "{}")

    subject.release("SubmitJob", "uid:1000")
    subject.admit("SubmitJob", "uid:1000", "{}")


def test_releasing_more_than_was_admitted_cannot_go_negative():
    subject, _clock = guard({"SubmitJob": MethodQuota(max_concurrent=1, max_calls=99)})
    subject.release("SubmitJob", "uid:1000")
    subject.release("SubmitJob", "uid:1000")
    subject.admit("SubmitJob", "uid:1000", "{}")

    with pytest.raises(GuardRefusedError, match="concurrency-limit-exceeded"):
        subject.admit("SubmitJob", "uid:1000", "{}")


def test_releasing_a_method_never_admitted_is_harmless():
    subject, _clock = guard()
    subject.release("SubmitJob", "uid:1000")


def test_an_oversized_payload_is_refused_before_it_is_parsed():
    """Parsing to discover it is too large has already paid the cost."""
    subject, _clock = guard({"SubmitJob": MethodQuota(max_bytes=32)})

    with pytest.raises(GuardRefusedError) as refusal:
        subject.admit("SubmitJob", "uid:1000", json.dumps({"payload": "x" * 100}))
    assert refusal.value.code == "payload-too-large"


def test_payload_size_is_measured_in_bytes_not_characters():
    subject, _clock = guard({"SubmitJob": MethodQuota(max_bytes=10)})
    with pytest.raises(GuardRefusedError, match="payload-too-large"):
        subject.admit("SubmitJob", "uid:1000", "é" * 6)


def test_a_payload_that_is_not_text_is_refused():
    subject, _clock = guard()
    with pytest.raises(GuardRefusedError, match="payload-invalid"):
        subject.admit("SubmitJob", "uid:1000", {"not": "text"})


@pytest.mark.parametrize(
    "field", ["owner", "ownerToken", "uid", "callerUid", "sender", "uniqueName", "identity"]
)
def test_a_caller_asserted_identity_is_refused_not_ignored(field):
    """The daemon already stamped the sender; naming one is asking to be believed."""
    subject, _clock = guard()

    with pytest.raises(GuardRefusedError) as refusal:
        subject.admit("SubmitJob", "uid:1000", json.dumps({"version": 1, field: "uid:0"}))
    assert refusal.value.code == "identity-asserted"
    assert field in refusal.value.detail


def test_an_ordinary_document_is_not_mistaken_for_an_assertion():
    subject, _clock = guard()
    subject.admit(
        "SubmitJob",
        "uid:1000",
        json.dumps({"version": 1, "requestId": "r-1", "workloadId": "visual-library"}),
    )


@pytest.mark.parametrize("text", ["not json", "[]", '"a string"', "", "null"])
def test_a_non_object_payload_is_left_to_the_method_schema(text):
    assert asserted_identity_field(text, 1024) == ""


def test_an_oversized_document_is_not_parsed_to_look_for_identity():
    assert asserted_identity_field(json.dumps({"owner": "uid:0"}), 4) == ""


def test_an_undeclared_method_is_refused_rather_than_unlimited():
    """The safe reading of "no quota was written" is "not exposed"."""
    subject, _clock = guard()

    with pytest.raises(GuardRefusedError) as refusal:
        subject.admit("DeleteEverything", "uid:1000", "{}")
    assert refusal.value.code == "method-unknown"


def test_the_default_methods_are_all_declared():
    subject, _clock = guard()
    for method in ("ApplyCommand", "SubmitJob", "CancelJob", "DescribePlugins"):
        assert subject.quota_for(method).max_calls >= 1


def test_the_caller_table_is_bounded_because_names_are_attacker_influenced():
    subject, _clock = guard(max_tracked_callers=2)

    for index in range(10):
        subject.admit("SubmitJob", f"name::1.{index}", "{}")

    assert len(subject._windows) == 2


def test_a_caller_cannot_evict_its_own_record_and_reset_its_limit():
    """The victim is always some other, less recently seen caller."""
    subject, _clock = guard(
        {"SubmitJob": MethodQuota(max_calls=1, max_concurrent=99)}, max_tracked_callers=1
    )
    subject.admit("SubmitJob", "uid:1000", "{}")

    for _attempt in range(5):
        with pytest.raises(GuardRefusedError, match="rate-limit-exceeded"):
            subject.admit("SubmitJob", "uid:1000", "{}")
        assert list(subject._windows) == [("SubmitJob", "uid:1000")]


def test_eviction_forgets_an_idle_caller_which_uid_keys_make_unexploitable():
    """One attacker is one key however many connections it opens."""
    subject, _clock = guard(
        {"SubmitJob": MethodQuota(max_calls=1, max_concurrent=99)}, max_tracked_callers=1
    )
    subject.admit("SubmitJob", "uid:1000", "{}")

    subject.admit("SubmitJob", "uid:1001", "{}")

    assert list(subject._windows) == [("SubmitJob", "uid:1001")]
    subject.admit("SubmitJob", "uid:1000", "{}")


def test_a_refusal_is_a_stable_versioned_document():
    refusal = GuardRefusedError("rate-limit-exceeded", "too fast", method="SubmitJob")

    document = json.loads(refusal.text())

    assert document == {
        "version": GUARD_ERROR_VERSION,
        "status": "rejected",
        "code": "rate-limit-exceeded",
        "message": "too fast",
        "method": "SubmitJob",
    }


@pytest.mark.parametrize(
    "quota",
    [
        MethodQuota(max_bytes=0),
        MethodQuota(max_calls=0),
        MethodQuota(max_concurrent=True),
        MethodQuota(window_seconds=0),
        MethodQuota(window_seconds=10_000),
        MethodQuota(window_seconds=True),
    ],
)
def test_an_impossible_quota_is_refused(quota):
    with pytest.raises(GuardRefusedError, match="quota-invalid"):
        BusGuard({"SubmitJob": quota})


def test_a_quota_that_is_not_a_quota_is_refused():
    with pytest.raises(GuardRefusedError, match="quota-invalid"):
        BusGuard({"SubmitJob": {"max_calls": 1}})


@pytest.mark.parametrize("bound", [0, True, 99_999])
def test_the_caller_table_bound_is_validated(bound):
    with pytest.raises(GuardRefusedError, match="quota-invalid"):
        BusGuard(max_tracked_callers=bound)


def test_the_context_manager_releases_the_slot_even_when_the_call_raises():
    subject, _clock = guard({"SubmitJob": MethodQuota(max_concurrent=1, max_calls=99)})

    with pytest.raises(RuntimeError), guarded(subject, "SubmitJob", "uid:1000", "{}"):
        raise RuntimeError("the method failed")

    with guarded(subject, "SubmitJob", "uid:1000", "{}") as held:
        assert held is not None


def test_every_refusal_validates_against_the_published_contract():
    """The applet parses replies by schema, so a refusal needs one of its own."""
    from omnitensor.registry import validate_document

    subject, _clock = guard({"SubmitJob": MethodQuota(max_calls=1, max_bytes=64)})
    subject.admit("SubmitJob", "uid:1000", "{}")

    refusals = []
    for method, payload in (
        ("SubmitJob", "{}"),
        ("SubmitJob", json.dumps({"padding": "x" * 200})),
        ("SubmitJob", json.dumps({"owner": "uid:0"})),
        ("SubmitJob", 7),
        ("Unlisted", "{}"),
    ):
        with pytest.raises(GuardRefusedError) as refusal:
            subject.admit(method, "uid:1000", payload)
        refusals.append(refusal.value)

    codes = set()
    for refusal in refusals:
        document = json.loads(refusal.text())
        assert validate_document("runtime-refusal.schema.json", document) == []
        codes.add(document["code"])

    assert codes == {
        "rate-limit-exceeded",
        "payload-too-large",
        "identity-asserted",
        "payload-invalid",
        "method-unknown",
    }


def test_a_refusal_is_not_mistaken_for_a_job_acknowledgement():
    from omnitensor.registry import validate_document

    document = json.loads(GuardRefusedError("rate-limit-exceeded", "slow down").text())

    assert validate_document("runtime-job-acknowledgement.schema.json", document) != []


def test_a_full_reconciliation_burst_is_admitted():
    """A client reconciling a large catalogue must not be refused partway.

    Measured live: nine profiles cost 23 ApplyCommand calls in one burst, and
    a refusal midway leaves the client and the runtime divergent with nothing
    shown to the user.
    """
    subject, _clock = guard()

    for index in range(120):
        subject.admit("ApplyCommand", "uid:1000", "{}")
        subject.release("ApplyCommand", "uid:1000")
        assert index >= 0


def test_apply_command_is_still_bounded():
    """Sized for reconciliation, not unlimited."""
    subject, _clock = guard()
    quota = subject.quota_for("ApplyCommand")

    for _ in range(quota.max_calls):
        subject.admit("ApplyCommand", "uid:1000", "{}")
        subject.release("ApplyCommand", "uid:1000")

    with pytest.raises(GuardRefusedError, match="rate-limit-exceeded"):
        subject.admit("ApplyCommand", "uid:1000", "{}")
