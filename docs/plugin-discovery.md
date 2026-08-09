# Plugin discovery

OmniTensor discovers external workload plugins through the
`omnitensor.workloads` Python entry-point group. Each entry-point name is the
plugin identity and its value is the import target used only after metadata,
identity, compatibility, and policy checks succeed. A distribution should
ship one `omnitensor-plugin.json` file containing its manifest v2 document.

Discovery is metadata-only: it reads installed distribution metadata and the
bundled manifest catalog, but never calls `EntryPoint.load()` or imports plugin
modules. Results are deterministic: validated bundled manifests are sorted by
entry-point name first, followed by external entry points sorted by name,
distribution, and target. Duplicate and broken candidates remain visible for
the identity-validation layer to reject independently.
