# What the control boundary is worth

The control service scopes everything a caller owns — its jobs, its quota
window, its results — by a uid token. This document says what that separates,
what it does not, and why per-connection scoping is deliberately not offered
as an alternative.

Read it before relying on the boundary for anything, and before adding a
method that returns one caller's data to another.

## Refusals are method replies

A guarded method returns a version-one `runtime-refusal` document when the
call is over quota, oversized, asserts an identity, exceeds concurrency, or
names an unknown method. That reply is deliberately not a job acknowledgement.
Consumers must try the refusal contract before parsing the method's success
contract; treating every text reply as an acknowledgement hides actionable
policy failures behind a generic parse error.

The transport keeps its own failures apart from these: an envelope that does
not validate, or a method that does not exist at dispatch, is answered with a
`control-reply` **error**, while a method-level refusal is a **result**,
because the method answered and its answer is the refusal document. The
Cinnamon applet follows that order in `runtime-control-gateway.js` and maps
every published refusal code to deterministic recovery text. Its
`guard-refusal-regression.test.js` drives all codes through the real gateway
and manager boundary. Unknown or malformed envelopes still fail closed.

## The guarantee

A caller's owner token is `uid:<n>`, read from `SO_PEERCRED` when the control
socket accepts the connection — the kernel stamps it, no daemon is asked, and
nothing the peer sends can change it. Two consequences follow, and they are
the whole of the guarantee:

- **A different uid cannot read or cancel your jobs.** If the socket is ever
  reachable by another user, that user is a different owner and is refused.
- **The same uid can, from any connection.** Reconnecting keeps your jobs.

## What it does not separate

The socket lives in `$XDG_RUNTIME_DIR/omnitensor`, which is mode 0700, so
every peer that can reach it is normally the same user. Between two programs
you started yourself, this boundary separates nothing at all. It is not a
sandbox, and no part of the service treats it as one:

- Any process running as you can submit jobs, read your results, and cancel
  them.
- Plugin confinement is a separate mechanism (seccomp and bubblewrap, see
  `docs/plugin-workers.md`) and does not depend on this.
- Consent for plugin permissions is deliberately **not** granted over the
  socket, for exactly this reason — it is a command-line tool, because access
  to the account's files is an authority the socket cannot offer. See the
  module docstring in `omnitensor/consent.py`.

Where uid scoping earns its keep is the case it was written for: a socket
reachable by another uid. It costs a parse of the bound sender token and it
never claims isolation it does not have.

## Why per-connection scoping is not offered

The obvious alternative is to scope by the connection — each accepted socket
gets a serial, and the sender token carries it — so that a caller's jobs are
visible only to the connection that submitted them. It is narrower, and
narrower sounds safer. It is not offered, for two reasons.

**It would separate nothing for the caller that matters.** Every applet,
extension, and desklet loaded into the Cinnamon process runs as the same
user; scoping by connection would give the applet a token that any of its
neighbours could equally mint by connecting — the same non-separation as uid
scoping, with none of the durability.

**It would break the caller that matters.** A connection lasts exactly as
long as one socket. The applet is reloaded whenever it is upgraded or the
user reloads the desktop, and its adapter opens a fresh connection per
request. Under connection scoping every job in flight at that moment becomes
unreachable: still running, still consuming a device, and owned by a token
nobody can present any more. An owner token that cannot be produced by its
owner is not a boundary, it is a leak that reports itself as a permission
error.

The serial that does appear in the sender token (`peer:<uid>:<serial>`)
exists for audit — "which connection did this" — and never widens or narrows
what a caller may touch. A connection whose credentials cannot be read is
anonymous rather than guessed at: without credentials, nothing is the most
that can truthfully be claimed.

## What the applet relies on

Exactly one property: **that a job survives the applet's own reconnect.** It
submits a job, receives an id, and polls for the result; between the
submission and the result the applet opens and closes many connections, and
it has no way to re-submit because the input has already been staged and
consumed. Uid scoping gives it that. Nothing else in the applet depends on
the boundary — it never asks for another caller's jobs, and it would not be
refused if it did, because it is the same user.

## If this ever needs to be stronger

The thing to change is not the token. It is the transport: today's socket
already sits in a per-user directory with 0600 permissions, so the next step
would be a system service with a socket policy naming which uids may call
which methods. That makes `uid:<n>` mean something a per-user session cannot
make it mean. Tightening the token alone would move the boundary somewhere it
is easier to reason about and no harder to cross.
