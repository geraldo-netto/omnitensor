# Plugin worker supervision

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
