# Runtime snapshot evolution

The top-level runtime snapshot remains contract version `1`. Existing fields
and their meanings are unchanged. Callers that do not supply plugin telemetry
still receive the original six-field version-one document, so stored fixtures
and readers of the original contract keep working.

The optional `pluginTelemetry` extension has its own version. Version `1`
contains at most 128 plugin entries, sorted by plugin ID. Every entry has a
fixed allowlist of health, current-stage, artifact-readiness, queue/activity,
timestamp, and saturating-counter fields. Free-form diagnostic detail is not
published. IDs, error codes, timestamps, counters, and cardinality are bounded
by the canonical schema.

The service publishes the extension and registers every installed version-two
plugin manifest. Consumers must inspect `pluginTelemetry.version` before using
its entries and ignore the optional extension when they only implement the
original snapshot view.
