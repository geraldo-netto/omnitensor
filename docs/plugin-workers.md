# Plugin worker supervision

## Filesystem and device isolation

External workers run under Bubblewrap in private user, mount, PID, UTS, IPC,
and cgroup namespaces. The sandbox starts with a fresh `/proc`, `/dev`, and
`/tmp`; it never bind-mounts the host root. Trusted Python/runtime roots are
read-only. A plugin's other mounts are derived only from permissions that are
both declared in its manifest and active in the grant ledger:

- `read:/absolute/path` or `read:/absolute/directory/*` becomes read-only;
- `write:/absolute/path` becomes writable;
- `device:/dev/apex_N`, `device:/dev/accel/accelN`, or
  `device:/dev/dri/renderDN` exposes only that named accelerator node.

Relative/non-canonical paths, broad device globs, the host `/proc` or `/dev`,
kernel debug/tracing/BPF namespaces, and undeclared grants fail worker startup.
Free-form plugin permissions do not imply filesystem access. The same active
permission set is passed into the worker's `PluginContext`.

Each active plugin runs in one child process launched without a shell, without
inheritable file descriptors, and in a separate process session. The service
starts workers in plugin-ID order and accepts a worker only after the bounded
IPC handshake confirms its plugin identity and protocol range. A failed or
incompatible worker becomes an isolated status record and does not prevent
later plugins from starting.

The supervisor monitors every accepted child for exit. Shutdown proceeds in
reverse startup order: send a bounded cancellation frame, close the IPC input,
wait for graceful exit, then terminate and finally kill after bounded waits.
This lifecycle is the process foundation; permission-derived filesystem,
device, network, and execution restrictions are layered on by the sandbox
policy items.
