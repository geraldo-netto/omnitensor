# Plugin conformance kit

`omnitensor.sdk` ships reusable checks for the public plugin boundary.
`run_plugin_contract()` validates lifecycle, health, bounded progress, result
identity, JSON output, and terminal timestamps while guaranteeing shutdown.
`run_cancellation_contract()` requires a pre-cancelled request to return a
cancelled result within an explicit deadline.

The deterministic JSON corpus generator supports property and fuzz tests with
explicit depth, width, string, seed, and case bounds. `FailureInjector`
provides named one-shot or repeated failures at lifecycle and pipeline
boundaries. `changed_function_targets()` maps changed lines to the narrowest
AST function, and `mutmut_function_patterns()` produces selectors so mutation
jobs can remain limited to changed functions instead of the full project.

These helpers establish contract behavior; use-case acceptance still requires
its own representative fixtures, safety evidence, model metrics, latency, and
named-hardware results.
