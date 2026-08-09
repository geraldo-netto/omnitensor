# Plugin identity validation

Discovery candidates are validated without importing their Python target.
Bundled identities come from OmniTensor's validated catalog. External
distributions must expose exactly one `omnitensor-plugin.json` manifest v2,
valid distribution metadata, and a syntactically valid import target. The
manifest `id`, manifest `plugin.entryPoint`, and installed entry-point name
must be identical.

Validation is candidate-local. Missing metadata, unreadable or invalid
manifests, incompatible versions, and identity mismatches become bounded
rejection records while the rest of the catalog remains available. If an
external ID conflicts with a bundled ID, the bundled plugin remains visible
and the external candidate is rejected. Ambiguous external duplicates are all
rejected rather than resolved by installation order.
