# Plugin SDK

External plugins import their supported API from `omnitensor.sdk`. The package
exports lifecycle and execution contracts, collector and pipeline hooks,
artifact references, configuration specifications, permission checks,
cancellation, bounded progress, and terminal-result factories. It does not
export the service, scheduler, control socket, registry, executor, or persistence
implementations.

Subclass `ManagedPlugin` for identity-checked, idempotent startup and shutdown.
Use `ConfigurationView` for defensive typed reads, `PermissionView.require()`
before accessing a declared host resource, `ArtifactView.require_ready()` for
verified artifacts, and `CancellationController` in local composition tests.
`ProgressEmitter` enforces bounded monotonic progress. The result factories
copy output values and validate terminal timestamps and details before they
cross IPC.

Configuration secrets never belong in plugin settings. Mark string fields
with `x-omnitensor-secret: true` and persist only a host secret reference:

```json
{"$secretRef":{"provider":"secret-service","key":"plugins/example/token","version":"v1"}}
```

OmniTensor resolves references through an injected provider immediately
before plugin execution. Plugins receive the resolved value through their
normal `ConfigurationView`; persisted state, snapshots, results, and logs use
the reference or a redacted value. Missing providers and invalid values fail
closed without returning secret material.

The SDK follows the manifest protocol compatibility policy. Additive exports
are backward compatible; removals or signature changes require a negotiated
protocol/SDK major version rather than importing service internals as a
workaround.

## Consent that keeps applying

`PermissionView` is the public, immutable view of the grants supplied to one
worker. Call `PermissionView.require()` immediately before each
permission-dependent unit of work and let cancellation propagate promptly; do
not cache a successful check as authority for a later operation.

The installed host owns continuous revocation. It checks the live grant ledger
before dispatch and while a request runs. If the active set changes, the host
cancels the request, stops the isolated worker, returns `consent-revoked`, and
refuses later work until the worker is started with the current grants. Plugin
packages do not read the host ledger or import service-side grant monitors.

This split keeps persistence and process supervision outside the public SDK
while ensuring consent withdrawn between queueing and dispatch stops the
current work, not merely the next request. See
[plugin-workers.md](plugin-workers.md) for the host enforcement boundary.
