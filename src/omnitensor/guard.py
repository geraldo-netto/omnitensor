"""Quotas and refusals at the control boundary, before a request means anything.

The control socket is the one surface any local client can reach, and every
method behind it costs something: parsing, a schema validation, a scheduler
slot.  So admission happens here, on the raw text, before a request is parsed —
parsing a ten-megabyte document to discover it is too large has already paid
the cost the limit exists to avoid.

Quotas are per caller, not global.  A global rate limit lets one noisy client
lock everyone else out, which turns a limit meant to protect the service into a
denial-of-service primitive aimed at its users.  The table of callers is itself
bounded, because an unbounded table is its own denial-of-service; the key is
the caller's uid token, so one peer is one entry however many connections it
opens.

Identity is never taken from the request body.  The kernel stamps the
sender at accept time; a document that also *claims* an identity is refused rather than
ignored, because the only reason to send one is to be believed.

Every refusal carries a stable, versioned code.  A caller that cannot tell
"you are going too fast" from "that workload does not exist" has to guess, and
retrying the wrong one is how a client turns a refusal into an outage.
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from .registry import validate_document

GUARD_ERROR_VERSION = 1
RUNTIME_REFUSAL_SCHEMA = "runtime-refusal.schema.json"
DEFAULT_MAX_TRACKED_CALLERS = 256
MAX_TRACKED_CALLERS_LIMIT = 4096
# Fields that would mean the caller is asserting who it is.
ASSERTED_IDENTITY_FIELDS = frozenset(
    {"owner", "ownerToken", "uid", "callerUid", "sender", "uniqueName", "identity"}
)


class GuardRefusedError(Exception):
    """A request refused at the boundary, with a code a client can branch on."""

    def __init__(self, code: str, detail: str, *, method: str = ""):
        self.code = code
        self.detail = detail
        self.method = method
        super().__init__(f"{code}: {detail}")

    def document(self) -> dict:
        return {
            "version": GUARD_ERROR_VERSION,
            "status": "rejected",
            "code": self.code,
            "message": self.detail,
            "method": self.method,
        }

    def text(self) -> str:
        document = self.document()
        violations = validate_document(RUNTIME_REFUSAL_SCHEMA, document)
        if violations:
            raise RuntimeError(f"runtime refusal violates contract: {violations}")
        return json.dumps(document, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class MethodQuota:
    """What one method allows one caller to do."""

    max_bytes: int = 256 * 1024
    max_calls: int = 60
    window_seconds: float = 10.0
    max_concurrent: int = 8

    def validate(self, method: str) -> None:
        for name in ("max_bytes", "max_calls", "max_concurrent"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise GuardRefusedError(
                    "quota-invalid", f"{method}.{name} must be a positive integer"
                )
        if (
            isinstance(self.window_seconds, bool)
            or not isinstance(self.window_seconds, (int, float))
            or not 0 < self.window_seconds <= 3600
        ):
            raise GuardRefusedError(
                "quota-invalid", f"{method}.window_seconds must be in (0, 3600]"
            )


DEFAULT_QUOTAS: Mapping[str, MethodQuota] = {
    # apply-command is one call per profile per setting, so any bulk change —
    # a user working through a catalogue, or a client applying several
    # settings — costs many calls in a short window, and a refusal partway
    # through leaves policy half-applied with no error the caller can act on.
    # The allowance is sized for that rather than for the steady state.
    "apply-command": MethodQuota(max_bytes=64 * 1024, max_calls=240, max_concurrent=4),
    # Jobs are larger and burstier; describing plugins is read-only and cheap
    # but trivially spammable.
    "submit-job": MethodQuota(max_bytes=256 * 1024, max_calls=60, max_concurrent=8),
    "cancel-job": MethodQuota(max_bytes=16 * 1024, max_calls=60, max_concurrent=8),
    # Polled while a job runs, so the rate is higher and the payload tiny.
    "get-job-result": MethodQuota(max_bytes=16 * 1024, max_calls=240, max_concurrent=8),
    "describe-plugins": MethodQuota(max_bytes=1, max_calls=30, max_concurrent=4),
    # A handshake is issued once per client per connection, so the allowance is
    # the smallest of any method — but it must exist, because an unlisted
    # method is refused, and a client refused the handshake cannot tell an
    # older service from a busy one, which is the confusion it exists to end.
    "describe-contract": MethodQuota(max_bytes=1, max_calls=20, max_concurrent=4),
}


def asserted_identity_field(text: str, max_bytes: int) -> str:
    """The identity field a document claims, or ``""``.

    Only well-formed JSON objects are inspected: anything else is refused
    later by the method's own schema, with a better message than this could
    give.
    """
    if len(text.encode("utf-8", "surrogatepass")) > max_bytes:
        return ""
    try:
        document = json.loads(text)
    except (ValueError, RecursionError):
        return ""
    if not isinstance(document, dict):
        return ""
    claimed = sorted(set(document) & ASSERTED_IDENTITY_FIELDS)
    return claimed[0] if claimed else ""


class _CallerWindow:
    """One caller's recent calls and in-flight count for one method."""

    __slots__ = ("calls", "in_flight")

    def __init__(self) -> None:
        self.calls: deque[float] = deque()
        self.in_flight = 0


class ControlGuard:
    """Per-caller admission for the control surface."""

    def __init__(
        self,
        quotas: Mapping[str, MethodQuota] | None = None,
        *,
        max_tracked_callers: int = DEFAULT_MAX_TRACKED_CALLERS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if (
            isinstance(max_tracked_callers, bool)
            or not isinstance(max_tracked_callers, int)
            or not 1 <= max_tracked_callers <= MAX_TRACKED_CALLERS_LIMIT
        ):
            raise GuardRefusedError(
                "quota-invalid",
                f"max_tracked_callers must be an integer in [1, {MAX_TRACKED_CALLERS_LIMIT}]",
            )
        resolved = dict(DEFAULT_QUOTAS)
        resolved.update(dict(quotas or {}))
        for method, quota in resolved.items():
            if not isinstance(quota, MethodQuota):
                raise GuardRefusedError("quota-invalid", f"{method} quota must be a MethodQuota")
            quota.validate(method)
        self._quotas = resolved
        self._max_tracked_callers = max_tracked_callers
        self._clock = clock
        self._windows: OrderedDict[tuple[str, str], _CallerWindow] = OrderedDict()

    def quota_for(self, method: str) -> MethodQuota:
        quota = self._quotas.get(method)
        if quota is None:
            # An unlisted method is not silently unlimited: the safe reading of
            # "no quota was written for this" is that it is not exposed.
            raise GuardRefusedError(
                "method-unknown", f"no quota is declared for {method}", method=method
            )
        return quota

    def admit(self, method: str, owner: str, payload: str = "") -> _CallerWindow:
        """Charge one call against ``owner``'s quota, or refuse it.

        Raises :class:`GuardRefusedError`; the caller turns that into
        whatever the transport's error shape is. The charged window is
        returned so the caller can give the slot back to *that* window even if
        the LRU has since evicted its key.
        """
        quota = self.quota_for(method)
        self._check_payload(method, quota, payload)
        window = self._window(method, owner)
        now = self._clock()
        self._expire(window, now, quota.window_seconds)
        if len(window.calls) >= quota.max_calls:
            raise GuardRefusedError(
                "rate-limit-exceeded",
                f"{method} allows {quota.max_calls} calls per {quota.window_seconds:g}s",
                method=method,
            )
        if window.in_flight >= quota.max_concurrent:
            raise GuardRefusedError(
                "concurrency-limit-exceeded",
                f"{method} allows {quota.max_concurrent} concurrent calls",
                method=method,
            )
        window.calls.append(now)
        window.in_flight += 1
        return window

    def release(self, method: str, owner: str) -> None:
        """Mark one admitted call finished; never lets the count go negative."""
        window = self._windows.get((method, owner))
        if window is not None:
            self.release_admitted(window)

    @staticmethod
    def release_admitted(window: _CallerWindow) -> None:
        """Give back the slot charged on this exact window.

        By key, an eviction between admit and release loses the decrement and
        the readmitted window under-counts its in-flight calls for good.
        """
        if window.in_flight > 0:
            window.in_flight -= 1

    def _check_payload(self, method: str, quota: MethodQuota, payload: object) -> None:
        if not isinstance(payload, str):
            raise GuardRefusedError("payload-invalid", f"{method} takes a string", method=method)
        size = len(payload.encode("utf-8", "surrogatepass"))
        if size > quota.max_bytes:
            raise GuardRefusedError(
                "payload-too-large",
                f"{method} accepts {quota.max_bytes} bytes, received {size}",
                method=method,
            )
        claimed = asserted_identity_field(payload, quota.max_bytes)
        if claimed:
            # The daemon already stamped the sender.  A document that also
            # names one is trying to be believed instead.
            raise GuardRefusedError(
                "identity-asserted",
                f"{method} must not carry a caller-asserted {claimed}",
                method=method,
            )

    def _window(self, method: str, owner: str) -> _CallerWindow:
        key = (method, owner)
        window = self._windows.get(key) or _CallerWindow()
        self._windows[key] = window
        self._windows.move_to_end(key)
        while len(self._windows) > self._max_tracked_callers:
            # The current caller was just moved to the end, so the victim is
            # always some *other*, less recently seen caller: a caller can
            # never evict its own record and reset the limit it is hitting.
            #
            # Eviction does forget an idle caller's history, which would matter
            # if a peer could mint keys at will.  It cannot: the key is the uid
            # token, so one attacker is one key however many connections it
            # opens, and filling this table takes that many distinct real users.
            self._windows.popitem(last=False)
        return window

    @staticmethod
    def _expire(window: _CallerWindow, now: float, window_seconds: float) -> None:
        horizon = now - window_seconds
        while window.calls and window.calls[0] <= horizon:
            window.calls.popleft()


class guarded:  # noqa: N801 - used as a context manager, not a type
    """Hold one admitted call for the duration of a block."""

    __slots__ = ("_guard", "_method", "_owner", "_payload", "_window")

    def __init__(self, guard: ControlGuard, method: str, owner: str, payload: str = "") -> None:
        self._guard = guard
        self._method = method
        self._owner = owner
        self._payload = payload
        self._window: _CallerWindow | None = None

    def __enter__(self) -> guarded:
        # Charged here rather than in __init__: one constructed and never
        # entered used to hold a concurrency slot no __exit__ would ever give
        # back, locking that caller out with concurrency-limit-exceeded.
        self._window = self._guard.admit(self._method, self._owner, self._payload)
        return self

    def __exit__(self, *_exception) -> None:
        window = self._window
        self._window = None
        if window is not None:
            self._guard.release_admitted(window)
