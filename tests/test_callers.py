"""Caller identity from control-socket peer credentials."""

from __future__ import annotations

import pytest

from omnitensor.callers import (
    ANONYMOUS,
    ANONYMOUS_OWNER,
    CallerIdentity,
    CallerIdentityResolver,
    bind_sender,
    current_sender,
    valid_sender,
)


@pytest.fixture(autouse=True)
def _clean_sender():
    bind_sender("")
    yield
    bind_sender("")


@pytest.mark.parametrize("name", ["peer:0:1", "peer:1000:1", "peer:1000:42", "peer:65534:999"])
def test_a_token_the_transport_could_mint_is_accepted(name):
    assert valid_sender(name)


@pytest.mark.parametrize(
    "name",
    [
        "",
        "peer:1000",  # no serial
        "peer::1",  # no uid
        "peer:01:1",  # leading zero could alias two spellings of one uid
        "peer:1000:01",
        "peer:-1:1",
        "peer:1000:1:extra",
        "uid:1000",
        ":1.42",  # the retired D-Bus unique-name form
        "org.cinnamon.OmniTensor1",
        None,
        7,
    ],
)
def test_a_token_the_transport_would_not_mint_is_refused(name):
    assert not valid_sender(name)


def test_binding_a_sender_makes_it_the_current_one():
    bind_sender("peer:1000:1")
    assert current_sender() == "peer:1000:1"


@pytest.mark.parametrize("sender", ["", None, 7])
def test_binding_nothing_clears_the_previous_sender(sender):
    """A connection without credentials must not inherit the last caller."""
    bind_sender("peer:1000:1")
    bind_sender(sender)
    assert current_sender() == ""


def test_a_peer_token_resolves_to_the_uid_it_carries():
    subject = CallerIdentityResolver()
    identity = subject.resolve("peer:1000:7")

    assert identity == CallerIdentity("peer:1000:7", 1000)
    assert subject.owner_token("peer:1000:7") == "uid:1000"


def test_the_bound_sender_is_used_when_none_is_given():
    subject = CallerIdentityResolver()
    bind_sender("peer:1000:7")

    assert subject.owner_token() == "uid:1000"


@pytest.mark.parametrize("name", ["", "not a token", ":1.7", "peer:x:1"])
def test_an_unmintable_token_is_anonymous_not_someone_else(name):
    subject = CallerIdentityResolver()

    assert subject.resolve(name) is ANONYMOUS
    assert subject.owner_token(name) == ANONYMOUS_OWNER


def test_an_owner_token_survives_the_reconnect_the_applet_performs():
    """The one property the applet relies on, pinned.

    It submits a job, is handed an id, and polls; between those the Cinnamon
    process can reload and reconnect, and it cannot re-submit because the
    input has already been staged and consumed. Scoping by the connection
    serial would leave that job running, holding a device, and owned by a
    token nobody can present again.
    """
    submitted = CallerIdentity("peer:1000:7", 1000)
    after_reload = CallerIdentity("peer:1000:404", 1000)

    assert submitted.owner_token == after_reload.owner_token

    # And a different user is still a different owner, which is the whole of
    # what the boundary does separate.
    assert CallerIdentity("peer:1001:7", 1001).owner_token != submitted.owner_token


def test_the_identity_document_carries_sender_uid_and_owner():
    assert CallerIdentity("peer:1000:7", 1000).document() == {
        "sender": "peer:1000:7",
        "uid": 1000,
        "owner": "uid:1000",
    }


def test_the_anonymous_identity_claims_nothing():
    assert ANONYMOUS.owner_token == ANONYMOUS_OWNER
    assert ANONYMOUS.document() == {"sender": "", "uid": None, "owner": ANONYMOUS_OWNER}
