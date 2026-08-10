from __future__ import annotations

import asyncio

import pytest

from omnitensor.callers import (
    ANONYMOUS,
    ANONYMOUS_OWNER,
    CallerError,
    CallerIdentity,
    CallerIdentityResolver,
    bind_sender,
    caller_capture_handler,
    current_sender,
    unix_user_lookup,
    valid_bus_name,
)


class Message:
    def __init__(self, sender):
        self.sender = sender


def resolver(uids=None, **changes):
    lookups = []

    async def unix_user(name):
        lookups.append(name)
        if uids is None:
            raise RuntimeError("no credentials")
        return uids[name]

    subject = CallerIdentityResolver(unix_user, **changes)
    return subject, lookups


def test_an_identity_is_scoped_by_uid_so_it_survives_a_reconnect():
    """A token nobody can present again is a bug that looks like security."""
    first = CallerIdentity(":1.7", 1000)
    reconnected = CallerIdentity(":1.94", 1000)

    assert first.owner_token == reconnected.owner_token == "uid:1000"


def test_an_identity_without_credentials_falls_back_to_the_connection():
    assert CallerIdentity(":1.7").owner_token == "name::1.7"
    assert ANONYMOUS.owner_token == ANONYMOUS_OWNER


def test_an_identity_describes_itself():
    assert CallerIdentity(":1.7", 1000).document() == {
        "uniqueName": ":1.7",
        "uid": 1000,
        "owner": "uid:1000",
    }


def test_the_sender_is_bound_from_the_message_and_dispatch_continues():
    handler = caller_capture_handler()

    assert handler(Message(":1.42")) is None
    assert current_sender() == ":1.42"


def test_a_message_without_a_sender_clears_rather_than_inherits():
    """Leaving the last value would attribute one caller's call to another."""
    handler = caller_capture_handler()
    handler(Message(":1.42"))

    handler(Message(None))

    assert current_sender() == ""


@pytest.mark.parametrize("sender", ["", "not a bus name", "..", 7, ":1", "x" * 300])
def test_an_implausible_sender_is_not_bound(sender):
    handler = caller_capture_handler()
    handler(Message(":1.42"))

    handler(Message(sender))

    assert current_sender() == ""


def test_the_capture_handler_can_report_what_it_bound():
    seen = []
    handler = caller_capture_handler(seen.append)

    handler(Message(":1.9"))

    assert seen == [":1.9"]


@pytest.mark.parametrize("name", [":1.0", ":1.42", "org.cinnamon.OmniTensor1", "a.b"])
def test_a_well_formed_bus_name_is_accepted(name):
    assert valid_bus_name(name)


@pytest.mark.parametrize("name", ["", ":1", "1.2", ".leading", "has space", None, 7])
def test_a_malformed_bus_name_is_refused(name):
    assert not valid_bus_name(name)


def test_a_caller_resolves_to_its_uid():
    subject, _lookups = resolver({":1.7": 1000})
    identity = asyncio.run(subject.resolve(":1.7"))

    assert identity == CallerIdentity(":1.7", 1000)
    assert asyncio.run(subject.owner_token(":1.7")) == "uid:1000"


def test_an_unresolvable_caller_is_anonymous_not_someone_else():
    """The daemon refuses for a peer that already disconnected."""
    subject, _lookups = resolver(None)
    identity = asyncio.run(subject.resolve(":1.7"))

    assert identity.uid is None
    assert identity.owner_token == "name::1.7"


@pytest.mark.parametrize("uid", [True, -1, "1000", None])
def test_an_implausible_uid_is_discarded(uid):
    async def unix_user(_name):
        return uid

    subject = CallerIdentityResolver(unix_user)
    assert asyncio.run(subject.resolve(":1.7")).uid is None


def test_an_absent_sender_resolves_as_anonymous():
    subject, lookups = resolver({":1.7": 1000})
    assert asyncio.run(subject.resolve("")) is ANONYMOUS
    assert asyncio.run(subject.resolve("not a name")) is ANONYMOUS
    assert lookups == []


def test_the_bound_sender_is_used_when_none_is_given():
    subject, _lookups = resolver({":1.7": 1000})
    bind_sender(":1.7")

    assert asyncio.run(subject.owner_token()) == "uid:1000"


def test_a_resolved_caller_is_cached_so_every_call_is_not_a_round_trip():
    subject, lookups = resolver({":1.7": 1000})

    asyncio.run(subject.resolve(":1.7"))
    asyncio.run(subject.resolve(":1.7"))

    assert lookups == [":1.7"]


def test_the_cache_is_bounded_because_a_peer_can_mint_names():
    subject, lookups = resolver({f":1.{index}": 1000 for index in range(10)}, max_cached_callers=2)

    for index in range(10):
        asyncio.run(subject.resolve(f":1.{index}"))
    asyncio.run(subject.resolve(":1.0"))

    assert len(subject._cache) == 2
    assert lookups.count(":1.0") == 2


def test_a_cached_caller_can_be_forgotten():
    subject, lookups = resolver({":1.7": 1000})
    asyncio.run(subject.resolve(":1.7"))

    subject.forget(":1.7")
    subject.forget(":1.99")
    asyncio.run(subject.resolve(":1.7"))

    assert lookups == [":1.7", ":1.7"]


def test_without_a_lookup_every_caller_is_anonymous():
    subject = CallerIdentityResolver()
    assert asyncio.run(subject.resolve(":1.7")).uid is None


def test_attaching_the_lookup_discards_identities_resolved_without_it():
    """Those were anonymous only because there was no bus to ask yet."""
    subject = CallerIdentityResolver()
    assert asyncio.run(subject.resolve(":1.7")).uid is None

    async def unix_user(_name):
        return 1000

    subject.attach(unix_user)

    assert asyncio.run(subject.resolve(":1.7")).uid == 1000


def test_an_uncallable_lookup_is_refused():
    with pytest.raises(CallerError, match="lookup-invalid"):
        CallerIdentityResolver().attach("not callable")


@pytest.mark.parametrize("bound", [0, -1, True])
def test_the_cache_bound_is_validated(bound):
    with pytest.raises(CallerError, match="bounds-invalid"):
        CallerIdentityResolver(max_cached_callers=bound)


def test_the_credential_lookup_asks_the_bus_daemon():
    class Reply:
        body = [1000]

    class Bus:
        def __init__(self):
            self.calls = []

        async def call(self, message):
            self.calls.append(message)
            return Reply()

    bus = Bus()
    assert asyncio.run(unix_user_lookup(bus)(":1.7")) == 1000
    [message] = bus.calls
    assert message.member == "GetConnectionUnixUser"
    assert message.destination == "org.freedesktop.DBus"
    assert message.body == [":1.7"]


@pytest.mark.parametrize("reply", [None, type("Empty", (), {"body": []})()])
def test_a_bus_that_does_not_describe_the_caller_raises(reply):
    class Bus:
        async def call(self, _message):
            return reply

    with pytest.raises(CallerError, match="caller-unknown"):
        asyncio.run(unix_user_lookup(Bus())(":1.7"))


def test_an_owner_token_survives_the_reconnect_the_applet_performs():
    """The one property the applet relies on, pinned.

    It submits a job, is handed an id, and polls; between those the Cinnamon
    process can reconnect to the bus, and it cannot re-submit because the
    input has already been staged and consumed. Scoping by the unique name
    would leave that job running, holding a device, and owned by a token
    nobody can present again — see docs/bus-boundary.md.
    """
    submitted = CallerIdentity(":1.7", 1000)
    after_reload = CallerIdentity(":1.404", 1000)

    assert submitted.owner_token == after_reload.owner_token

    # And a different user is still a different owner, which is the whole of
    # what the boundary does separate.
    assert CallerIdentity(":1.7", 1001).owner_token != submitted.owner_token
