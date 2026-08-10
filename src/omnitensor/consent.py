"""Issuing and withdrawing plugin consent, from the machine that owns it.

The ledger has always been able to record a grant.  Nothing could ever create
one: it was never constructed by the service, and no command, method, or tool
reached it — so every declared permission read as ungranted, permanently, and
the refusal a user saw had no remedy.

This is that remedy, and it is a command-line tool rather than a bus method on
purpose.  A grant authorises a plugin to reach something outside itself, which
is a larger thing to hand out than enabling a profile.  On a session bus every
peer runs as the same user and the service cannot tell one from another
(OMNI-0174), so a bus method would accept consent from anything already able to
talk to it and record it as the user's.  A tool run from a shell has an
authority the bus cannot offer: it required access to this account's files
before it ran at all.

What a grant means here:

*Per plugin and permission.*  Granting ``files:read`` to one plugin says
nothing about any other.

*Only what the plugin declared.*  A permission absent from the manifest cannot
be granted, so consent cannot be given for something the publisher never asked
for and a manifest that later drops a permission stops it being usable.

*Persistent until revoked.*  Re-consenting every boot trains people to say yes
without reading, which is worse than the grant lasting.

*Withdrawn immediately.*  Enforcement re-reads the ledger before it checks, so
a revoked grant stops a job already sitting in a queue.

Every change is recorded with who made it, when, and why, and the ledger's
revision has to match — two shells racing cannot both write.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
import time
from pathlib import Path

from .plugins.grants import (
    GrantError,
    GrantLedger,
    GrantOrigin,
    GrantProvenance,
)
from .plugins.loading import (
    discover_plugin_metadata,
    resolve_plugin_compatibility,
    resolve_plugin_identities,
)
from .registry import bundled_workloads_path
from .service import DEFAULT_GRANTS_PATH

MAX_REASON_CHARS = 200


def _ledger_path(argument: str | None) -> Path:
    configured = argument or os.environ.get("OMNITENSOR_GRANTS_PATH") or DEFAULT_GRANTS_PATH
    return Path(configured).expanduser()


def declared_permissions(plugin_id: str) -> set[str]:
    """What this plugin's manifest asks for, read without importing its code.

    Asking a plugin what it requires must never mean running it, and the
    declaration is the only thing that bounds what may be granted.
    """
    candidates = discover_plugin_metadata(bundled_root=bundled_workloads_path())
    catalog = resolve_plugin_compatibility(resolve_plugin_identities(candidates))
    for plugin in catalog.plugins:
        if plugin.plugin_id == plugin_id:
            return set(plugin.manifest["plugin"]["permissions"])
    raise GrantError("plugin-unknown", f"no installed plugin declares itself as {plugin_id}")


def _provenance(reason: str, request_id: str) -> GrantProvenance:
    return GrantProvenance(
        actor_id=_actor(),
        origin=GrantOrigin.USER,
        reason=reason[:MAX_REASON_CHARS],
        recorded_at_ms=int(time.time() * 1000),
        request_id=request_id,
    )


def _actor() -> str:
    """Who is consenting, as this machine knows them."""
    try:
        return getpass.getuser()[:120]
    except (KeyError, OSError):  # pragma: no cover - no passwd entry
        return f"uid-{os.getuid()}"


def _request_id() -> str:
    return f"cli-{int(time.time() * 1000)}-{os.getpid()}"


def _report(snapshot) -> dict:
    return {
        "pluginId": snapshot.plugin_id,
        "revision": snapshot.revision,
        "granted": sorted(grant.permission for grant in snapshot.active),
    }


def list_grants(ledger: GrantLedger, plugin_id: str) -> dict:
    declared = declared_permissions(plugin_id)
    snapshot = ledger.snapshot(plugin_id, declared)
    granted = {grant.permission for grant in snapshot.active}
    return {
        "pluginId": plugin_id,
        "revision": snapshot.revision,
        "permissions": [
            {"name": permission, "granted": permission in granted}
            for permission in sorted(declared)
        ],
    }


def grant(ledger: GrantLedger, plugin_id: str, permission: str, reason: str) -> dict:
    declared = declared_permissions(plugin_id)
    return _report(
        ledger.grant(
            plugin_id,
            permission,
            declared,
            _provenance(reason, _request_id()),
            expected_revision=ledger.revision,
        )
    )


def revoke(ledger: GrantLedger, plugin_id: str, permission: str, reason: str) -> dict:
    declared = declared_permissions(plugin_id)
    return _report(
        ledger.revoke(
            plugin_id,
            permission,
            declared,
            _provenance(reason, _request_id()),
            expected_revision=ledger.revision,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnitensor-grant",
        description="Show, issue, and withdraw the consent a plugin needs to run.",
    )
    parser.add_argument("--grants-path", default=None, help="ledger to read and write")
    subcommands = parser.add_subparsers(dest="command", required=True)

    listing = subcommands.add_parser("list", help="what a plugin declares and holds")
    listing.add_argument("plugin_id")

    for name, help_text in (
        ("grant", "consent to one declared permission"),
        ("revoke", "withdraw consent for one permission"),
    ):
        action = subcommands.add_parser(name, help=help_text)
        action.add_argument("plugin_id")
        action.add_argument("permission")
        action.add_argument(
            "--reason",
            default="",
            help="recorded with the change, for whoever reads the ledger later",
        )
    return parser


def run(argv: list[str]) -> tuple[int, str]:
    """Execute one command and return its exit status and output."""
    arguments = build_parser().parse_args(argv)
    ledger = GrantLedger(_ledger_path(arguments.grants_path))
    try:
        if arguments.command == "list":
            document = list_grants(ledger, arguments.plugin_id)
        elif arguments.command == "grant":
            document = grant(
                ledger, arguments.plugin_id, arguments.permission, arguments.reason
            )
        else:
            document = revoke(
                ledger, arguments.plugin_id, arguments.permission, arguments.reason
            )
    except GrantError as error:
        # Named rather than raised: "permission-undeclared" tells somebody the
        # manifest never asked for it, which a traceback does not.
        return 1, json.dumps({"error": error.code, "detail": error.detail}, indent=2)
    return 0, json.dumps(document, indent=2)


def main() -> None:  # pragma: no cover - process entry point
    status, output = run(sys.argv[1:])
    print(output)
    raise SystemExit(status)
