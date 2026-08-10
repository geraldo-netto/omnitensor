"""Per-plugin flow control: how much work is accepted, and how often.

A plugin that is slower than its trigger source will queue work faster than it
finishes it, and the failure is not a crash — it is unbounded memory, a device
monopolised by one profile, and duplicate side effects when a retry delivers a
result the first attempt already delivered.

Four rules, all per plugin so one profile cannot spend another's capacity:

* **Backpressure** bounds work in flight; past the bound submissions are
  dropped with a stable reason rather than queued forever.
* **Coalescing** joins a submission to work already running under the same
  idempotency key instead of starting a second copy of it.
* **Retention** answers a repeated key with the result the first attempt
  produced, so a retrying caller cannot cause a second delivery.
* **Deadlines and bounded retries** stop an operation that will not finish,
  and retry it a fixed number of times — never delivering more than once,
  because only the attempt that succeeds produces a result at all.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum

DEFAULT_MAX_IN_FLIGHT = 4
DEFAULT_DEADLINE_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 1
DEFAULT_RESULT_RETENTION = 64
DEFAULT_RESULT_TTL_SECONDS = 300.0
MAX_IN_FLIGHT_LIMIT = 256
MAX_IDEMPOTENCY_KEY_CHARS = 120


class FlowRefusal(StrEnum):
    """Why a submission was not accepted, stable enough to report to a caller."""

    BACKPRESSURE = "plugin-backpressure"
    DEADLINE = "deadline-exceeded"
    RETRIES_EXHAUSTED = "retries-exhausted"
    INVALID_KEY = "idempotency-key-invalid"


class FlowRefusedError(RuntimeError):
    """A submission was dropped or abandoned by flow control."""

    def __init__(self, refusal: FlowRefusal, detail: str):
        self.refusal = refusal
        self.code = str(refusal)
        self.detail = detail
        super().__init__(f"{self.code}: {detail}")


@dataclass(frozen=True, slots=True)
class FlowSnapshot:
    """Counters an operator needs to see whether a plugin is keeping up."""

    in_flight: int
    accepted: int
    coalesced: int
    replayed: int
    dropped: int
    retried: int
    failed: int


class PluginFlowController:
    """Admission, coalescing, retention, and retry for one plugin."""

    def __init__(
        self,
        plugin_id: str,
        *,
        max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
        deadline_seconds: float = DEFAULT_DEADLINE_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        result_retention: int = DEFAULT_RESULT_RETENTION,
        result_ttl_seconds: float = DEFAULT_RESULT_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        _positive_integer(max_in_flight, "max_in_flight", MAX_IN_FLIGHT_LIMIT)
        _positive_number(deadline_seconds, "deadline_seconds")
        _non_negative_integer(max_retries, "max_retries")
        _positive_integer(result_retention, "result_retention", 4096)
        _positive_number(result_ttl_seconds, "result_ttl_seconds")
        self._plugin_id = plugin_id
        self._max_in_flight = max_in_flight
        self._deadline_seconds = deadline_seconds
        self._max_retries = max_retries
        self._result_retention = result_retention
        self._result_ttl_seconds = result_ttl_seconds
        self._clock = clock
        self._in_flight: dict[str, asyncio.Future] = {}
        self._completed: OrderedDict[str, tuple[float, object]] = OrderedDict()
        self._accepted = 0
        self._coalesced = 0
        self._replayed = 0
        self._dropped = 0
        self._retried = 0
        self._failed = 0

    def snapshot(self) -> FlowSnapshot:
        return FlowSnapshot(
            len(self._in_flight),
            self._accepted,
            self._coalesced,
            self._replayed,
            self._dropped,
            self._retried,
            self._failed,
        )

    async def submit(
        self,
        idempotency_key: str,
        operation: Callable[[], Awaitable],
    ) -> object:
        """Run ``operation`` once for ``idempotency_key`` and return its result."""
        self._validate_key(idempotency_key)
        if not callable(operation):
            raise TypeError("operation must be callable")

        existing = self._existing(idempotency_key)
        if existing is not None:
            return await existing

        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._in_flight[idempotency_key] = future
        self._accepted += 1
        try:
            result = await self._run_with_retries(operation)
        except BaseException as error:
            if not future.done():
                future.set_exception(error)
            # A future nobody awaited would be reported as never-retrieved.
            future.exception()
            raise
        else:
            if not future.done():
                future.set_result(result)
            self._retain(idempotency_key, result)
            return result
        finally:
            self._in_flight.pop(idempotency_key, None)

    def _existing(self, idempotency_key: str):
        """Answer from retention or from work already running, or admit.

        Returns an awaitable when this submission is not new work, and ``None``
        when the caller should run it.  Admission is refused here too, so the
        three ways a submission avoids starting live in one place.
        """
        retained = self._retained(idempotency_key)
        if retained is not _MISSING:
            # The caller is retrying something that already succeeded.  Running
            # it again would deliver a second time for one logical request.
            self._replayed += 1
            return _completed(retained)

        running = self._in_flight.get(idempotency_key)
        if running is not None:
            self._coalesced += 1
            # Shielded so one impatient caller cannot cancel work another is
            # still waiting on.
            return asyncio.shield(running)

        if len(self._in_flight) >= self._max_in_flight:
            self._dropped += 1
            raise FlowRefusedError(
                FlowRefusal.BACKPRESSURE,
                f"{self._plugin_id} already has {self._max_in_flight} operations in flight",
            )
        return None

    async def _run_with_retries(self, operation: Callable[[], Awaitable]) -> object:
        attempts = self._max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                return await asyncio.wait_for(operation(), timeout=self._deadline_seconds)
            except TimeoutError:
                if attempt >= attempts:
                    self._failed += 1
                    raise FlowRefusedError(
                        FlowRefusal.RETRIES_EXHAUSTED,
                        f"{self._plugin_id} exceeded {self._deadline_seconds:g}s "
                        f"on {attempts} attempts",
                    ) from None
                self._retried += 1
        raise AssertionError("unreachable")  # pragma: no cover - loop always returns

    def _retained(self, key: str) -> object:
        entry = self._completed.get(key)
        if entry is None:
            return _MISSING
        stored_at, result = entry
        if self._clock() - stored_at > self._result_ttl_seconds:
            # Expiry is what keeps idempotency from becoming an unbounded log
            # of every request the plugin ever served.
            del self._completed[key]
            return _MISSING
        self._completed.move_to_end(key)
        return result

    def _retain(self, key: str, result: object) -> None:
        self._completed[key] = (self._clock(), result)
        self._completed.move_to_end(key)
        while len(self._completed) > self._result_retention:
            self._completed.popitem(last=False)

    def _validate_key(self, key: str) -> None:
        if not isinstance(key, str) or not 1 <= len(key) <= MAX_IDEMPOTENCY_KEY_CHARS:
            raise FlowRefusedError(
                FlowRefusal.INVALID_KEY,
                f"idempotency key must contain 1-{MAX_IDEMPOTENCY_KEY_CHARS} characters",
            )


async def _completed(result: object) -> object:
    return result


class _Missing:
    __slots__ = ()


_MISSING = _Missing()


def _positive_integer(value: object, name: str, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")


def _non_negative_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _positive_number(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a positive number")
