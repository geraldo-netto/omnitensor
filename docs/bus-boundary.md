# What the bus boundary is worth

The control service scopes everything a caller owns — its jobs, its quota
window, its results — by a uid token. This document says what that separates,
what it does not, and why per-connection scoping is deliberately not offered
as an alternative.

Read it before relying on the boundary for anything, and before adding a
method that returns one caller's data to another.

## The guarantee

A caller's owner token is `uid:<n>`, resolved from the bus daemon's
`GetConnectionUnixUser` for the sender of each message. Two consequences
follow, and they are the whole of the guarantee:

- **A different uid cannot read or cancel your jobs.** If the socket is
  reachable by another user — a shared machine, a bus socket with loose
  permissions, a system bus deployment — that user is a different owner and
  is refused.
- **The same uid can, from any connection.** Reconnecting keeps your jobs.

## What it does not separate

On a session bus every peer is normally the same user. Between two programs
you started yourself, this boundary separates nothing at all. It is not a
sandbox, and no part of the service treats it as one:

- Any process running as you can submit jobs, read your results, and cancel
  them.
- Plugin confinement is a separate mechanism (seccomp and bubblewrap, see
  `docs/plugin-workers.md`) and does not depend on this.
- Consent for plugin permissions is deliberately **not** granted over the
  bus, for exactly this reason — it is a command-line tool, because access
  to the account's files is an authority the bus cannot offer. See the
  module docstring in `omnitensor/consent.py`.

Where uid scoping earns its keep is the case it was written for: a socket
reachable by another uid. It costs one cached round trip per caller and it
never claims isolation it does not have.

## Why per-connection scoping is not offered

The obvious alternative is to scope by the sender's unique bus name
(`:1.42`), so that a caller's jobs are visible only to the connection that
submitted them. It is narrower, and narrower sounds safer. It is not offered,
for two reasons.

**It would separate nothing for the caller that matters.** The Cinnamon
applet calls through `Gio.DBus.session`, which is a per-process singleton in
GIO. Every applet, extension, and desklet loaded into the Cinnamon process
shares that one connection and therefore that one unique name. Scoping by it
would give the applet a token it shares with all of its neighbours — the same
non-separation as uid scoping, with none of the durability.

**It would break the caller that matters.** A unique name lasts exactly as
long as one connection. Cinnamon reconnects on a bus restart, and the applet
is reloaded whenever it is upgraded or the user reloads the desktop. Under
connection scoping every job in flight at that moment becomes unreachable:
still running, still consuming a device, and owned by a token nobody can
present any more. An owner token that cannot be produced by its owner is not
a boundary, it is a leak that reports itself as a permission error.

The fallback path already covers the case where connection scoping would be
the honest answer. When the bus daemon will not say who the caller is, the
token degrades to `name:<unique>` — narrower than uid, never wider — because
without credentials the connection is the most that can truthfully be
claimed. That is a fallback, not a mode: it is chosen by what the bus can
prove, not by what a caller asks for.

## What the applet relies on

Exactly one property: **that a job survives the applet's own reconnect.** It
submits a job, receives an id, and polls for the result; between the
submission and the result the Cinnamon process may reconnect to the bus, and
the applet has no way to re-submit because the input has already been staged
and consumed. Uid scoping gives it that. Nothing else in the applet depends
on the boundary — it never asks for another caller's jobs, and it would not
be refused if it did, because it is the same user.

## If this ever needs to be stronger

The thing to change is not the token. It is the transport: a per-user socket
with restrictive permissions, or a system-bus deployment with a policy file
naming which uids may call which methods. Both make `uid:<n>` mean something
the session bus cannot make it mean. Tightening the token alone would move
the boundary somewhere it is easier to reason about and no harder to cross.
