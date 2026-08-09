# Plugin SDK

External plugins import their supported API from `omnitensor.sdk`. The package
exports lifecycle and execution contracts, collector and pipeline hooks,
artifact references, configuration specifications, permission checks,
cancellation, bounded progress, and terminal-result factories. It does not
export the service, scheduler, D-Bus, registry, executor, or persistence
implementations.

Subclass `ManagedPlugin` for identity-checked, idempotent startup and shutdown.
Use `ConfigurationView` for defensive typed reads, `PermissionView.require()`
before accessing a declared host resource, `ArtifactView.require_ready()` for
verified artifacts, and `CancellationController` in local composition tests.
`ProgressEmitter` enforces bounded monotonic progress. The result factories
copy output values and validate terminal timestamps and details before they
cross IPC.

The SDK follows the manifest protocol compatibility policy. Additive exports
are backward compatible; removals or signature changes require a negotiated
protocol/SDK major version rather than importing service internals as a
workaround.
