"""A seccomp filter denying a worker the ability to become another program.

The sandbox already denies networking and drops every capability, but a worker
could still `execve` something else or fork a child that outlives its
supervision — and a plugin that can exec is a plugin whose declared behaviour
means nothing, because the process being supervised is no longer the process
that was reviewed.  ``bwrap`` cannot express that; only a seccomp filter can.

The filter is installed by the worker on itself, not through ``bwrap``'s
``--add-seccomp-fd``.  That was the obvious route and it does not work: bwrap
applies its filter *before* executing the payload, so a filter denying
``execve`` denies bwrap's own exec and nothing ever starts — confirmed by
running it.  Self-installation also draws the boundary better, covering exactly
the phase in which untrusted plugin code runs.

There are no seccomp Python bindings on this host, and adding libseccomp as a
runtime dependency would make the sandbox weaker on any machine that lacks it —
the failure mode would be "no filter" exactly where it matters.  So the classic
BPF is emitted directly: about a dozen instructions, no dependency, and the
tests install it in a real process and check that exec and fork are actually
refused rather than asserting on the bytes and hoping.

Three decisions worth keeping:

*Unknown architecture kills.*  A filter written for x86-64 syscall numbers says
nothing about another ABI's numbering, and applying it anyway would allow
whatever happens to share a number with something harmless.

*Thread creation stays allowed.*  Denying ``clone`` outright stops the runtime
from starting a thread, which breaks ordinary workers; the filter inspects the
flags and denies only the process-creating calls.

*``clone3`` returns ENOSYS, not EPERM.*  Its arguments live behind a pointer
seccomp cannot follow, so it cannot be inspected — but glibc treats ENOSYS as
"this kernel is too old" and falls back to ``clone``, which this filter *can*
inspect.  EPERM would instead surface as an unexplained failure.
"""

from __future__ import annotations

import ctypes
import os
import platform
import struct
from pathlib import Path

# Classic BPF, as the kernel's seccomp(2) accepts it.
_BPF_LD_W_ABS = 0x20
_BPF_JEQ_K = 0x15
_BPF_JSET_K = 0x45
_BPF_RET_K = 0x06

# struct seccomp_data
_OFFSET_NR = 0
_OFFSET_ARCH = 4
_OFFSET_ARG0_LOW = 16

SECCOMP_RET_KILL_PROCESS = 0x80000000
SECCOMP_RET_ALLOW = 0x7FFF0000
_SECCOMP_RET_ERRNO = 0x00050000

EPERM = 1
ENOSYS = 38
CLONE_THREAD = 0x00010000

AUDIT_ARCH_X86_64 = 0xC000003E
AUDIT_ARCH_AARCH64 = 0xC00000B7

# Only the architectures whose numbering is written down here can be filtered.
SYSCALLS: dict[str, dict[str, int]] = {
    "x86_64": {
        "execve": 59,
        "execveat": 322,
        "fork": 57,
        "vfork": 58,
        "clone": 56,
        "clone3": 435,
    },
    "aarch64": {
        # No fork or vfork: glibc implements both through clone.
        "execve": 221,
        "execveat": 281,
        "clone": 220,
        "clone3": 435,
    },
}
_AUDIT_ARCH = {"x86_64": AUDIT_ARCH_X86_64, "aarch64": AUDIT_ARCH_AARCH64}
_DENIED_OUTRIGHT = ("execve", "execveat", "fork", "vfork")


#: Architectures whose syscall numbering is written down here, and therefore
#: the only ones where a plugin worker can be confined.  A table is not
#: guesswork that can be extended from documentation alone: applying a filter
#: built from the wrong ABI's numbers denies unrelated syscalls, so an entry
#: belongs here only once somebody has run the suite on that machine.
SUPPORTED_MACHINES: tuple[str, ...] = ("aarch64", "x86_64")


class SeccompUnsupportedError(RuntimeError):
    """This architecture's syscall numbering is not written down here."""

    def __init__(self, machine: str):
        self.machine = machine
        super().__init__(
            f"no seccomp syscall table for {machine}; refusing to apply a filter"
            " written for another ABI. Confinement is implemented for "
            f"{', '.join(SUPPORTED_MACHINES)}; on any other machine a plugin "
            "worker cannot be sandboxed and will not start unless --no-seccomp "
            "is passed deliberately"
        )


def _instruction(code: int, jt: int, jf: int, k: int) -> bytes:
    return struct.pack("=HBBI", code, jt, jf, k)


def _return(value: int) -> bytes:
    return _instruction(_BPF_RET_K, 0, 0, value)


def _errno(number: int) -> int:
    return _SECCOMP_RET_ERRNO | (number & 0xFFFF)


def current_machine() -> str:
    return platform.machine()


def worker_filter(machine: str | None = None) -> bytes:
    """The BPF program a worker runs under."""
    machine = machine or current_machine()
    table = SYSCALLS.get(machine)
    if table is None:
        raise SeccompUnsupportedError(machine)

    denied = [table[name] for name in _DENIED_OUTRIGHT if name in table]
    clone = table.get("clone")
    clone3 = table.get("clone3")

    # Classic BPF only jumps forward, so the returns sit after everything that
    # branches to them and every offset is computed from the final layout.
    length = 3 + len(denied) + (1 if clone3 else 0) + (3 if clone else 0)
    allow_at = length
    eperm_at = allow_at + 1
    enosys_at = allow_at + 2
    kill_at = allow_at + 3

    program: list[bytes] = []

    def offset(target: int) -> int:
        return target - len(program) - 1

    program.append(_instruction(_BPF_LD_W_ABS, 0, 0, _OFFSET_ARCH))
    program.append(_instruction(_BPF_JEQ_K, 0, offset(kill_at), _AUDIT_ARCH[machine]))
    program.append(_instruction(_BPF_LD_W_ABS, 0, 0, _OFFSET_NR))
    for number in denied:
        program.append(_instruction(_BPF_JEQ_K, offset(eperm_at), 0, number))
    if clone3 is not None:
        program.append(_instruction(_BPF_JEQ_K, offset(enosys_at), 0, clone3))
    if clone is not None:
        program.append(_instruction(_BPF_JEQ_K, 0, offset(allow_at), clone))
        program.append(_instruction(_BPF_LD_W_ABS, 0, 0, _OFFSET_ARG0_LOW))
        program.append(
            _instruction(_BPF_JSET_K, offset(allow_at), offset(eperm_at), CLONE_THREAD)
        )

    assert len(program) == length, "filter layout and computed offsets disagree"
    program.append(_return(SECCOMP_RET_ALLOW))
    program.append(_return(_errno(EPERM)))
    program.append(_return(_errno(ENOSYS)))
    program.append(_return(SECCOMP_RET_KILL_PROCESS))
    return b"".join(program)


def instruction_count(program: bytes) -> int:
    return len(program) // 8


PR_SET_NO_NEW_PRIVS = 38
SECCOMP_SET_MODE_FILTER = 1
# seccomp(2) is not in glibc on every release, so it is called by number.
SYS_SECCOMP = {"x86_64": 317, "aarch64": 277}


class SeccompInstallError(RuntimeError):
    """The filter could not be installed, so the worker must not proceed."""


class _SockFprog(ctypes.Structure):
    _fields_ = (("len", ctypes.c_ushort), ("filter", ctypes.c_void_p))


def confinement_error(machine: str | None = None) -> str:
    """Why a worker could not be confined on this host, or ``""``.

    A probe rather than an attempt: installing a filter is irreversible for
    the process that does it, so this answers from what can be read — the
    syscall table for this ABI, and the kernel's own statement that it
    supports the filter mode.

    It exists because failing closed is right and being undiagnosable is not.
    On a kernel without ``CONFIG_SECCOMP_FILTER`` every plugin worker refused
    to start, correctly, and nothing anywhere said that the kernel was the
    reason.
    """
    machine = machine or current_machine()
    if machine not in SYSCALLS:
        return (
            f"no seccomp syscall table for {machine}; confinement is implemented "
            f"for {', '.join(SUPPORTED_MACHINES)}"
        )
    try:
        status = Path("/proc/self/status").read_text(encoding="utf-8")
    except OSError:
        # Nothing readable to judge by, so claim nothing: the worker's own
        # attempt remains the authority.
        return ""
    if "Seccomp:" not in status:
        return (
            "this kernel does not report seccomp support (no Seccomp field in "
            "/proc/self/status); a plugin worker cannot be confined and will not start"
        )
    return ""


def install_filter(program: bytes | None = None, *, machine: str | None = None, libc=None) -> int:
    """Apply the filter to *this* process, and to everything it spawns.

    Installed by the worker on itself rather than by ``bwrap``: bwrap applies
    its filter before executing the payload, so a filter denying ``execve``
    denies bwrap's own exec of the worker and nothing ever starts.  Installing
    it here also fits the boundary better — the filter covers exactly the phase
    where untrusted plugin code runs, and nothing before it.

    Raises rather than returning a failure: a worker that believes it is
    confined and is not would run plugin code under a guarantee that does not
    exist.
    """
    machine = machine or current_machine()
    number = SYS_SECCOMP.get(machine)
    if number is None:
        raise SeccompUnsupportedError(machine)
    program = worker_filter(machine) if program is None else program
    library = libc if libc is not None else ctypes.CDLL("libc.so.6", use_errno=True)

    # Without no_new_privs an unprivileged process may not install a filter at
    # all, and it is also what stops a setuid binary from escaping one.
    if library.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise SeccompInstallError(
            f"could not set no_new_privs: {os.strerror(ctypes.get_errno())}"
        )
    buffer = ctypes.create_string_buffer(program, len(program))
    fprog = _SockFprog(instruction_count(program), ctypes.cast(buffer, ctypes.c_void_p))
    if library.syscall(number, SECCOMP_SET_MODE_FILTER, 0, ctypes.byref(fprog)) != 0:
        raise SeccompInstallError(
            f"could not install the seccomp filter: {os.strerror(ctypes.get_errno())}"
        )
    return instruction_count(program)
