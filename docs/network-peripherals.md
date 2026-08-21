# Network and peripheral collection

`network-peripherals` collects bounded metadata for later anomaly inference. It
does not capture packets, addresses, SSIDs, Bluetooth names, serial numbers,
or user content. No model or hardware acceptance evidence ships yet, so these
collectors do not make the profile operational inference.

## Network metadata

`NetworkMetadataCollector` implements the standard async collector contract.
Before source access it requires the declared `read:network-metadata` grant.
Each link also requires an explicit stable-identity allowlist entry and the
matching `read:network-link/<stable-id>` grant. Unlisted and ungranted links
are neither emitted nor counted.
Its injected host source supplies typed aggregate NetworkManager and kernel
link observations: opaque stable identity, link kind/state, connectivity,
carrier, metering, default-route status, signal percentage, monotonic byte,
error, and drop counters, source health, and timestamps.

Output is sorted by stable identity and contains every allowlisted, granted
link; the allowlist itself accepts at most 256 identities, so nothing a person
selected is silently omitted. Churn is computed over the same eligible links,
using only stable identity
and link kind for additions and removals. Changes name public aggregate fields;
poll timestamps alone do not create changes, and a repeated replay snapshot
produces empty churn. Duplicate or malformed
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

Eligible output is sorted by stable identity and contains every allowlisted,
granted device; the allowlist itself accepts at most 64 identities, and source
input is rejected above 256 devices. Churn is
computed over the same eligible devices:
additions, removals, and changed public fields are stable-sorted. Poll timestamps
alone do not create changes, and a
repeated replay snapshot produces empty churn.

`ReplayPeripheralMetadataSource` exercises hotplug, removal, health changes,
and source degradation without claiming real USB or Bluetooth hardware
evidence. This collector adds no model and does not make the profile
operational inference.
