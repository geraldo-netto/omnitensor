"""What this service speaks, answerable before anything else is called.

A client had no way to ask.  It could only call a method and read the failure,
and "this service is older than you" arrives looking exactly like "this service
is broken" — an ``UnknownMethod`` error, or a refusal code — which is the wrong
diagnosis to show a user and the wrong one to act on.  D-Bus introspection is
not the answer either: it enumerates method *names*, and a name says nothing
about which version of the document body behind it the service accepts.  Two
services can both export ``SubmitJob`` and disagree completely about what a
submission looks like.

So the handshake reports both halves: the methods that exist, and the version
of each named wire contract.  A client compares that against what it can send
and decides what to offer before it offers it.

The lists are not written here by hand.  ``methods`` is derived from the bus
interface and ``schemas`` from the shipped schema files, so a method or a
contract added on one side cannot quietly go unannounced on the other.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable

from .registry import load_schema, schema_names, validate_document

RUNTIME_CONTRACT_VERSION = 1
CONTRACT_SCHEMA = "runtime-contract.schema.json"
# Only the documents that cross the bus.  A workload manifest and a plugin
# manifest are read from disk by the service itself; announcing them would
# describe an agreement this handshake is not part of.
_WIRE_SCHEMA_PREFIXES = ("runtime-", "plugin-inventory")


def wire_schema_names(names: Iterable[str]) -> tuple[str, ...]:
    """The schema files that describe documents crossing the bus."""
    return tuple(sorted(name for name in names if name.startswith(_WIRE_SCHEMA_PREFIXES)))


def _declared_version(schema: dict) -> int | None:
    """The version a document must carry, as the schema itself pins it."""
    declared = schema.get("properties", {}).get("version", {}).get("const")
    return declared if isinstance(declared, int) and not isinstance(declared, bool) else None


def schema_versions(loader: Callable[[str], dict] = load_schema) -> dict[str, int]:
    """Map each wire contract to the version its own schema pins.

    Read from the schema rather than declared beside it: a hand-kept table is
    a second place for the number to live, and the copy that drifts is always
    the one a client is told.  A schema that pins no version is omitted rather
    than guessed at, because an announced version nobody enforces is worse
    than none.
    """
    versions: dict[str, int] = {}
    for name in wire_schema_names(schema_names()):
        declared = _declared_version(loader(name))
        if declared is not None:
            versions[name.removesuffix(".schema.json")] = declared
    return versions


def build_contract_document(
    methods: Iterable[str], loader: Callable[[str], dict] = load_schema
) -> dict:
    """The handshake document for an interface exporting ``methods``."""
    return {
        "version": RUNTIME_CONTRACT_VERSION,
        "methods": sorted(methods),
        "schemas": schema_versions(loader),
    }


def contract_document_text(
    methods: Iterable[str], loader: Callable[[str], dict] = load_schema
) -> str:
    """The handshake as the bus returns it, validated before it is sent.

    Validated here rather than trusted: the document is assembled from what
    the interface and the schema directory happen to contain, so a malformed
    one is a reachable state rather than a typo somebody would have caught.
    """
    document = build_contract_document(methods, loader)
    errors = validate_document(CONTRACT_SCHEMA, document)
    if errors:
        raise ValueError(f"contract description is invalid: {'; '.join(errors)}")
    return json.dumps(document, separators=(",", ":"))
