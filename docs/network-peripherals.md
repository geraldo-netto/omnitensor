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

## USB and Bluetooth peripheral metadata

`PeripheralMetadataCollector` requires `read:peripheral-metadata` before any
source access. A device is emitted only when its opaque stable identity is in
the collector's explicit allowlist and the matching
`read:peripheral-device/<stable-id>` grant is active. Revoked, unlisted, and
ungranted devices are neither emitted nor counted.

The typed source contract contains only bus, broad device class, health,
connection/authorization/pairing/trust flags, battery percentage, an aggregate
error count, and timestamps. It excludes USB descriptors, vendor/product IDs,
serials, Bluetooth addresses and names, HID events, file contents, and traffic.

Eligible output is sorted by stable identity and capped at 32 devices by
default (64 hard maximum); source input is rejected above 256 devices. Churn is
computed over the bounded emitted set: additions, removals, and changed public
fields are stable-sorted. Poll timestamps alone do not create changes, and a
repeated replay snapshot produces empty churn.

`ReplayPeripheralMetadataSource` exercises hotplug, removal, health changes,
and source degradation without claiming real USB or Bluetooth hardware
evidence. This collector adds no model and does not make the profile
operational inference.
