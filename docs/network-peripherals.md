# Network and peripheral collection

`network-peripherals` collects bounded metadata for later anomaly inference. It
does not capture packets, addresses, SSIDs, Bluetooth names, serial numbers,
or user content. No model or hardware acceptance evidence ships yet, so these
collectors do not make the profile operational inference.

## Network metadata

`NetworkMetadataCollector` implements the standard async collector contract.
Before source access it requires the declared `read:network-metadata` grant.
Its injected host source supplies typed aggregate NetworkManager and kernel
link observations: opaque stable identity, link kind/state, connectivity,
carrier, metering, default-route status, signal percentage, monotonic byte,
error, and drop counters, source health, and timestamps.

Output is sorted by stable identity and capped at 64 links by default (256 hard maximum).
Extra links are counted, not emitted. Duplicate or malformed
identities, invalid counters, and future timestamps fail the collection. The
trigger payload is never copied into collector output.

`ReplayNetworkMetadataSource` replays the same typed snapshots used by a live
host adapter. Tests exercise it through `TriggerCoordinator`, providing a
deterministic source-health and collection path without claiming access to
real NetworkManager state or network hardware.
