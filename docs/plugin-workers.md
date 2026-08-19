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

`files:read-selected` is implemented by a broker, not by mounting a caller's
directory. The service copies only the explicitly submitted, canonical regular
files into a private per-request directory already mounted read-only in that
worker. Each copy lives below a broker-owned ordinal directory, so safe original
basenames remain available for user-visible citations without allowing a name
collision or path traversal. Originals and sibling files stay outside the
namespace, and the staged copies are removed at the terminal boundary.
The system timezone database is mounted read-only as trusted runtime metadata,
so named-zone validation works without exposing unrelated `/usr/share` data.

Each active plugin runs in one child process launched without a shell, without
inheritable file descriptors, and in a separate process session. The service
starts workers in plugin-ID order and accepts a worker only after the bounded
IPC handshake confirms its plugin identity and protocol range. A failed or
incompatible worker becomes an isolated status record and does not prevent
later plugins from starting.

The installed service also creates one delegated cgroup per worker before
launch. A trusted exec shim joins that cgroup before Bubblewrap or plugin code
can fork, and the kernel enforces its process and aggregate-memory ceilings.
The supervisor polls cgroup membership, memory, and descriptor use while each
call runs, and separately bounds call duration and concurrency. Result size is
not bounded: an answer larger than one IPC frame is carried as ordered result
chunks and reassembled by the supervisor, because refusing it discarded the
whole answer and cost a model reload on the next request. A non-concurrency
violation stops the worker before its channel can be reused. External-worker startup fails closed when the systemd-delegated cgroup
is unavailable; embedders may explicitly select procfs accounting for tests,
but that fallback cannot contain a daemonised descendant.

Executable workers negotiate `execute`, `progress`, and `cancel` explicitly.
`SubmitJob` sends one schema-validated request over the authenticated worker
channel, relays bounded redacted progress, and accepts exactly one correlated
terminal result, whether it arrives in one frame or as ordered chunks. Cancellation remains readable while plugin code is running;
if a worker does not return a terminal cancellation result within the bound,
the supervisor stops it before releasing the channel. An installed identity
without a ready worker or negotiated `execute` capability remains visible for
diagnosis but cannot receive a job.

Handshake and startup are separate phases with separate deadlines. The worker
answers the `hello` as soon as the protocol is settled, then runs the plugin's
`start` and sends a `ready` frame when it completes. Protocol negotiation is
therefore bounded in milliseconds while a plugin that loads a model or opens a
device gets the far longer startup budget. A worker that never reports ready
fails with its own reason rather than as a handshake failure, and because the
same deadline would be missed on every attempt, a startup timeout ends recovery
instead of spending the restart budget on identical retries.

The supervisor monitors every accepted child for exit. Shutdown proceeds in
reverse startup order: send a bounded cancellation frame, close the IPC input,
wait for graceful exit, then terminate and finally kill after bounded waits.
This lifecycle is the process foundation; permission-derived filesystem,
device, network, and execution restrictions are layered on by the sandbox
policy items.

## Network and capabilities

A worker gets its own empty network namespace unless its manifest declares and
is granted a network permission, so a plugin cannot reach a socket, a name
server, or the session bus regardless of what its code attempts. Every
capability is dropped explicitly rather than assumed absent.

Two network scopes are recognised, and each maps to something the sandbox can
actually enforce:

| Permission | Effect |
| --- | --- |
| *(none)* | private network namespace; loopback only |
| `network:localhost` | same — a new namespace gets its own loopback |
| `network:outbound` | the host network namespace is shared |

There is no narrower scope, because bwrap offers no partial sharing and a
finer-grained claim would be one the sandbox could not enforce. An unrecognised
scope is refused rather than ignored: treating it as "no network" would hide a
policy the author intended, and treating it as "network" would grant more than
was declared.

`tests/test_plugin_sandbox.py` runs the real `bwrap` and checks what actually
happens rather than trusting the constructed argv.
