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

## Plugin inventory (`DescribePlugins`)

The snapshot answers "what is the runtime doing"; it deliberately does not
answer "why is this profile doing nothing". That question needs what a plugin
*declares* against what it *has*, which is what `DescribePlugins` returns,
validated against `plugin-inventory.schema.json`.

Each entry carries the plugin's identity and source, its negotiated protocol
range and capabilities, its triggers, the readiness of every declared artifact
with the reason when one is not ready, and every declared permission with
whether it is currently granted. Declared-but-ungranted is reported explicitly
because that is the case a user can act on.

Two omissions are deliberate. Configuration **values** never appear — only the
declared schema — because a value can be a resolved secret; fields marked
`x-omnitensor-secret` are listed by name in `secretConfigurationKeys` so a
consumer renders a control rather than a value. And the document is built
entirely from installed metadata and artifact-store state: asking what a plugin
requires must never mean importing and running its code.

The method is read-only and takes no arguments, so it cannot be used to change
policy or to probe for a plugin that is not installed.

## Contract handshake (`DescribeContract`)

Every other method assumes the caller already knows what this service speaks.
Before this method existed there was no way to ask: a client called something
and read the failure, and "this service is older than you" arrives looking
exactly like "this service is broken" — a D-Bus `UnknownMethod`, or a refusal
code. Introspection does not close the gap either, because it enumerates
method *names*, and two services can both export `SubmitJob` while disagreeing
completely about what a submission looks like.

`DescribeContract` takes no arguments and returns a document validated against
`runtime-contract.schema.json`:

```json
{
  "version": 1,
  "methods": ["ApplyCommand", "CancelJob", "DescribeContract", "…"],
  "schemas": {"runtime-job-submit": 1, "runtime-snapshot": 1, "…": 1}
}
```

`version` is the version of this description, not of the contracts it
describes. `methods` is every method the bus interface exports, including this
one — a handshake a client must already know another method to reach is not a
handshake. `schemas` maps each wire contract to the version its own schema
pins; a name absent from the map is a document this service does not speak.

Neither list is written by hand. `methods` is asserted against the decorated
methods on the bus interface and `schemas` is read from the shipped schema
files, so a method or a contract added on one side cannot go unannounced on
the other. A schema pinning no version is omitted rather than guessed at: an
announced version nobody enforces is worse than no announcement.

The method is quota-guarded like every other, with the smallest allowance of
any of them, because a handshake is issued once per client per connection.
