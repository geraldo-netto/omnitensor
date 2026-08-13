"""Run one job through its stages with policy, flow, and cancellation applied.

Each of those three exists on its own and each is useless alone: a policy gate
nobody asks decides nothing, a flow controller nothing submits through bounds
nothing, and a cancellation token no stage watches stops nothing.  This is the
one place that holds all three around the same job, so a stage author cannot
forget one of them.

The order is deliberate.  Flow control decides whether the job runs *at all*
(it may be a duplicate of one already running, or a replay of one already
delivered) before any stage starts, because the cheapest work is the work not
repeated.  Policy is then re-checked before every stage, because the answer
legitimately changes while a job waits.  Cancellation wraps each stage, so a
stop signal reaches the stage that is actually executing rather than being
noticed after it finishes.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .cancellation import (
    CancellationReason,
    CancellationRegistry,
    JobCancelledError,
)
from .flow import FlowRefusedError, PluginFlowController
from .pipeline import (
    PipelineStage,
    PipelineStateMachine,
    StageOutput,
)
from .policy import PipelinePolicyGate, PolicyRefusedError
from .protocol import PluginResult

StageCallable = Callable[[object], Awaitable[StageOutput]]

# The stage order the state machine enforces; the runner walks the same list so
# a stage cannot be silently skipped by a caller supplying a shorter mapping.
STAGE_ORDER = (
    PipelineStage.COLLECT,
    PipelineStage.PREPROCESS,
    PipelineStage.RESOLVE,
    PipelineStage.INFER,
    PipelineStage.POSTPROCESS,
    PipelineStage.DELIVER,
)


@dataclass(frozen=True, slots=True)
class StageFailure:
    """Why a job ended before delivering, in a form a consumer can show."""

    code: str
    detail: str


class PipelineRunner:
    """Drive one plugin's jobs through every stage under all three controls."""

    def __init__(
        self,
        plugin_id: str,
        stages: dict[PipelineStage, StageCallable],
        *,
        policy: PipelinePolicyGate,
        flow: PluginFlowController,
        cancellations: CancellationRegistry,
        clock_ms: Callable[[], int] = lambda: int(time.time() * 1000),
    ) -> None:
        missing = [stage for stage in STAGE_ORDER if stage not in stages]
        if missing:
            raise ValueError(
                f"every stage must be supplied; missing {', '.join(missing)}"
            )
        self._plugin_id = plugin_id
        self._stages = dict(stages)
        self._policy = policy
        self._flow = flow
        self._cancellations = cancellations
        self._clock_ms = clock_ms

    async def run(self, job_id: str, request: object, *, idempotency_key: str = "") -> PluginResult:
        """Produce exactly one terminal result for ``job_id``."""
        key = idempotency_key or job_id
        try:
            return await self._flow.submit(key, lambda: self._run_stages(job_id, request))
        except FlowRefusedError as error:
            # Flow refusals happen before any stage runs, so there is no
            # machine to finish; the result is synthesised at the same shape.
            return self._failed(job_id, error.code, error.detail)

    async def _run_stages(self, job_id: str, request: object) -> PluginResult:
        machine = PipelineStateMachine(job_id)
        token = self._cancellations.track(job_id, self._plugin_id)
        try:
            machine.start()
            carried: object = request
            for stage in STAGE_ORDER:
                self._policy.require(stage)
                token.raise_if_cancelled()
                carried = await token.guard(
                    lambda stage=stage, carried=carried: self._stages[stage](carried)
                )
                machine.advance(carried, completed_at_ms=self._clock_ms())
            return machine.terminal_result
        except JobCancelledError as error:
            machine.cancel(error.detail, completed_at_ms=self._clock_ms())
            return machine.terminal_result
        except PolicyRefusedError as error:
            # Policy is a refusal, not a fault: the job stops because the user
            # or the runtime said so, so it is reported as cancelled.
            machine.cancel(error.detail, completed_at_ms=self._clock_ms())
            return machine.terminal_result
        except Exception as error:  # noqa: BLE001 - one plugin fault ends one job
            machine.fail(
                f"{type(error).__name__}: {error}"[:200],
                completed_at_ms=self._clock_ms(),
            )
            return machine.terminal_result
        finally:
            self._cancellations.release(job_id)

    def cancel(self, job_id: str, detail: str = "") -> bool:
        """Withdraw a running job on behalf of its caller."""
        return self._cancellations.cancel(job_id, CancellationReason.CALLER, detail)

    def _failed(self, job_id: str, code: str, detail: str) -> PluginResult:
        machine = PipelineStateMachine(job_id)
        machine.fail(f"{code}: {detail}"[:200], completed_at_ms=self._clock_ms())
        return machine.terminal_result
