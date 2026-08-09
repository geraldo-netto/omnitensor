# Local plugin runner

`omnitensor.sdk` includes deterministic test doubles for triggers, collectors,
artifact resolution, inference executors, permission grants, clocks, progress,
and terminal result sinks. `LocalPipelineRunner` composes those boundaries and
runs collector, preprocessing, artifact readiness, inference, postprocessing,
and delivery without importing accelerator libraries or service internals.

All inputs are replay fixtures. Time advances only when the test advances the
fake clock. Missing fixtures, undeclared permissions, artifact failures, and
adapter exceptions become stable failed results; cancellation and deadline
expiry become cancelled results. Recorded progress and results have explicit
cardinality bounds, making the runner suitable for deterministic CI and local
plugin development rather than performance qualification.
