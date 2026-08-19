# Writing and shipping an OmniTensor plugin

This guide is the end-to-end route for a plugin author: package a
distribution, get it discovered, declare what it needs, publish the model
artifacts it runs, test it locally, and then install, upgrade, and roll it
back without losing state. The reference material for each layer lives in its
own document; this guide is the order to read them in and the decisions that
tie them together.

Start from `examples/omnitensor-plugin-template`, which is an independently
buildable distribution that already satisfies everything below.

## 1. Packaging

A plugin is an ordinary Python distribution. It must:

- declare exactly one `omnitensor.workloads` entry point whose **name is the
  plugin id** and whose value is the import target;
- ship exactly one manifest-v2 `omnitensor-plugin.json`;
- depend on `omnitensor` and import only `omnitensor.sdk`.

```toml
[project.entry-points."omnitensor.workloads"]
template-workload = "omnitensor_template.plugin:TemplatePlugin"
```

Put the manifest **inside the package directory**, so it installs as
`your_package/omnitensor-plugin.json` rather than at the wheel root. Two
plugins that both install a top-level `omnitensor-plugin.json` overwrite each
other's manifest. Discovery matches the file by name anywhere in the
distribution, so nesting it costs nothing.

Importing service internals instead of `omnitensor.sdk` is not a shortcut, it
is an upgrade hazard: only the SDK carries the compatibility promise in
[plugin-sdk.md](plugin-sdk.md).

## 2. Discovery and identity

Discovery is metadata-only — it never imports plugin code — and identity
validation runs before anything is loaded. The entry-point name, its value,
and the distribution name must all agree with the manifest, and they are
re-checked inside the worker process before the plugin is constructed, so a
distribution swapped after discovery is rejected rather than run.

See [plugin-discovery.md](plugin-discovery.md) and
[plugin-identity.md](plugin-identity.md).

## 3. Schemas and configuration

The canonical JSON schemas in `schemas/` are the contract with the Cinnamon
applet. A plugin never emits a document that has not been validated against
them, and a schema is never weakened without coordinating both repositories.

Plugin configuration has its own schema, declared in the manifest and enforced
on load and on every update. Configuration is revisioned: an update supplies
the revision it read, and a mismatch is rejected rather than silently
overwriting a concurrent change. Ship migrations for every version edge you
expect to upgrade across; a version with no migration path fails closed
instead of guessing.

Secrets never live in configuration. Mark the field `x-omnitensor-secret` and
persist a host secret reference; see [plugin-sdk.md](plugin-sdk.md).

## 4. Permissions

Declare every permission in the manifest. Declaring is not granting: the host
intersects declared with granted, and the worker receives only that
intersection. Undeclared grants are refused outright, so a permission cannot be
smuggled in from the host side.

Consent is revocable while work runs, not only between runs. Use the public
`PermissionView.require()` at each permission-dependent operation boundary and
let cancellation propagate promptly. The installed host monitors the live
grant ledger, cancels active work, and stops the worker when its grant set
changes; plugin packages must not import the service's ledger or supervisor
internals. See [plugin-workers.md](plugin-workers.md) for that enforcement
boundary.

Filesystem, device, network, and execution restrictions are derived from the
granted set. Free-form permissions do not imply filesystem access; see
[plugin-workers.md](plugin-workers.md).

## 5. Artifacts

Models are immutable, content-addressed, and verified. Declare each artifact in
the manifest with its id, version, format, and SHA-256, plus the optional
`sourceUri` and `licenseSpdx` origin fields, which are carried through to the
worker as artifact provenance. The host resolves an
artifact only through the digest gate, so a file that no longer matches its
declared digest is never executed.

Publish an artifact with provenance: a signature over the reference plus its
publisher and key id. Installation verifies the digest while copying, verifies
trust before activation, and only then swaps the active pointer, keeping the
previous version as the single rollback target. Storage is bounded by a quota
whose collection never removes the active or rollback version.

There is deliberately no CPU backend: an artifact must be compiled for GPU,
NPU, or TPU. Ship the compiler report that proves the mapping.

## 6. Testing locally

Before installing anything:

- `omnitensor-plugin-smoke` runs the SDK conformance and smoke checks against
  your plugin without a bus or hardware;
- the local plugin runner ([local-plugin-runner.md](local-plugin-runner.md))
  executes a full collect/infer/deliver cycle from replay fixtures;
- the conformance suite ([plugin-conformance.md](plugin-conformance.md))
  checks the contracts the host relies on.

Use replayable fixtures rather than live sources. A collector that can only be
tested against real hardware cannot be tested in CI, and its failure modes —
a missing sensor, a device removed mid-scan — are exactly the ones that matter.

## 7. Installing

Install the distribution into an environment the service can import, then
restart the service so discovery runs. The service starts one worker process
per active plugin, in plugin-id order, and a plugin that fails to start becomes
an isolated status record rather than stopping the others.

The handshake and plugin startup are bounded separately: protocol negotiation
must complete in milliseconds, while `start` may take much longer to load a
model or open a device. A worker that never reports ready fails with its own
reason and does not spend the restart budget on a deadline it will miss again.

## 8. Upgrading and rolling back

Upgrade is: install the new distribution, restart the service, let identity and
compatibility validation run. Three things are versioned independently and
each has its own rollback:

| What | Rollback |
| --- | --- |
| Plugin distribution | reinstall the previous version and restart |
| Model artifact | `rollback` swaps the active pointer back to the retained previous version |
| Configuration | migrations move forward; keep the previous revision to restore |

The artifact store keeps exactly one rollback target, so validate a new
artifact before publishing a second one over it. Executor caches revalidate a
model against its file, so replacing an artifact in place is observed rather
than served from a stale entry — but replacing in place still defeats the
digest gate and rollback, so publish a new version instead.

## 9. Compatibility policy

The manifest declares a protocol range; the host negotiates the highest common
version and refuses a plugin with no overlap. Additive SDK exports are backward
compatible. Removals and signature changes require a negotiated protocol or SDK
major version — never an import of service internals to work around a gap.

The same rule governs the applet contract: a snapshot the applet already
renders must keep rendering after an upgrade, so schema changes are additive
until both repositories agree to a version bump.
