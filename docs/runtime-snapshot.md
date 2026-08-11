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

## Cinnamon compatibility

Snapshot evolution is exercised from both ends. OmniTensor validates every
published document against `schemas/runtime-snapshot.schema.json`; the Cinnamon
applet's `tests/helpers/runtime-snapshot-fixtures.js` supplies the corresponding
old and extended documents to both its generated schema validator and its
runtime gateway regression tests.

| Producer document | Cinnamon result |
| --- | --- |
| Version-one fields only | Accepted and rendered |
| Optional `pluginTelemetry` version 1 | Validated, then safely ignored by the current view |
| Alert with optional `resultRef` | Accepted without claiming the referenced result was fetched |
| Unknown extension version or malformed reference | Rejected as a contract failure |

This compatibility matrix is additive: an old snapshot stays usable, while a
new field is never silently accepted without a schema and a deterministic
consumer outcome. The applet regression
`runtime-schema-validation-regression.test.js` proves the combined live shape
(`pluginTelemetry` plus `resultRef`) as well as the original shape.

## Kernel telemetry (`kernelTelemetry`)

The optional version-one block publishes only aggregate run-queue and block-I/O
latency histograms plus monotonic counters. Series, bucket counts, names,
timestamps, details, and integer magnitudes are all bounded. Process, user,
cgroup, path, and payload identity is rejected by the helper reader before a
document can reach this contract.

`state` distinguishes a ready sample from an absent, unreachable, or invalid
helper. Those states carry empty measurements rather than invented zeroes: an
idle kernel and an unavailable probe are not the same observation. The
Cinnamon applet validates this optional block and safely ignores it until a
view needs it, so either producer shape remains usable.

The `resource-scheduler` collector consumes the same aggregate. It exposes
sample counts plus P50 and P95 log2-bucket upper bounds for the declared
run-queue and block-I/O histograms. This keeps model features fixed and bounded
while deterministic scheduling enforcement remains outside accelerator code.

## Input roots (`inputs`)

The optional `inputs` block states where a caller may stage a referenced input
buffer and how large one may be. It exists because a caller cannot discover
those paths any other way: a path outside the configured roots is refused with
the same answer as a path that does not exist, deliberately, so that refusals
cannot be used to probe the filesystem — and that also makes trial and error
useless for finding the right directory.

`roots` is bounded at eight entries and `maxBytes` restates the per-reference
byte ceiling the service enforces. Empty roots mean the service will read no
referenced file at all, which is the default: referencing a file is a
capability, not something that is on unless configured off. An absent block
means a runtime older than the field. Both refuse every reference, so a
consumer that treats them identically is correct.

The block is republished on every tick rather than fixed at startup, so a
consumer that cached it would keep offering a directory the service has stopped
reading.

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
