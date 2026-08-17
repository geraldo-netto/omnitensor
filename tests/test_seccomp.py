from __future__ import annotations

import struct
import subprocess
import sys
import textwrap

import pytest

from omnitensor.plugins.seccomp import (
    AUDIT_ARCH_X86_64,
    CLONE_THREAD,
    SECCOMP_RET_ALLOW,
    SECCOMP_RET_KILL_PROCESS,
    SYSCALLS,
    SeccompInstallError,
    SeccompUnsupportedError,
    current_machine,
    install_filter,
    instruction_count,
    worker_filter,
)

INSTRUCTION = struct.Struct("=HBBI")


def decode(program):
    return [
        INSTRUCTION.unpack_from(program, offset)
        for offset in range(0, len(program), INSTRUCTION.size)
    ]


def run_child(body):
    """Run ``body`` in a real process so the filter applies to a real kernel."""
    source = textwrap.dedent(
        """
        import os, sys
        sys.path.insert(0, {path!r})
        from omnitensor.plugins.seccomp import install_filter
        {body}
        """
    ).format(path="src", body=textwrap.indent(textwrap.dedent(body), ""))
    return subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, timeout=60
    )


def test_the_filter_is_a_well_formed_bpf_program():
    program = worker_filter("x86_64")

    assert len(program) % 8 == 0
    assert instruction_count(program) == len(decode(program))


def test_every_jump_stays_inside_the_program():
    """Classic BPF only jumps forward; an offset past the end is a kernel reject."""
    for machine in SYSCALLS:
        instructions = decode(worker_filter(machine))
        for index, (code, jt, jf, _k) in enumerate(instructions):
            if code & 0x07 == 0x05:  # BPF_JMP
                assert index + 1 + jt < len(instructions)
                assert index + 1 + jf < len(instructions)


def test_the_architecture_is_checked_before_any_syscall_number():
    """A number means nothing until the ABI it belongs to is confirmed."""
    instructions = decode(worker_filter("x86_64"))

    assert instructions[0] == (0x20, 0, 0, 4)
    assert instructions[1][0] == 0x15
    assert instructions[1][3] == AUDIT_ARCH_X86_64
    assert instructions[2] == (0x20, 0, 0, 0)


def test_an_unexpected_architecture_kills_rather_than_allows():
    instructions = decode(worker_filter("x86_64"))
    _code, _jt, jf, _k = instructions[1]
    target = instructions[1 + 1 + jf]

    assert target == (0x06, 0, 0, SECCOMP_RET_KILL_PROCESS)


def test_the_program_ends_with_allow_and_the_refusals():
    returns = [item for item in decode(worker_filter("x86_64")) if item[0] == 0x06]

    assert returns[0][3] == SECCOMP_RET_ALLOW
    assert returns[-1][3] == SECCOMP_RET_KILL_PROCESS


def test_clone_is_inspected_rather_than_denied_outright():
    """Denying clone outright stops the runtime from starting a thread."""
    instructions = decode(worker_filter("x86_64"))
    jset = [item for item in instructions if item[0] == 0x45]

    assert len(jset) == 1
    assert jset[0][3] == CLONE_THREAD


def test_an_architecture_without_a_table_is_refused():
    with pytest.raises(SeccompUnsupportedError, match="riscv64"):
        worker_filter("riscv64")
    with pytest.raises(SeccompUnsupportedError, match="riscv64"):
        install_filter(machine="riscv64")


def test_an_architecture_without_fork_omits_it():
    """aarch64 has no fork or vfork; glibc implements both through clone."""
    assert "fork" not in SYSCALLS["aarch64"]
    assert instruction_count(worker_filter("aarch64")) < instruction_count(worker_filter("x86_64"))


@pytest.mark.skipif(
    current_machine() not in SYSCALLS, reason="no syscall table for this architecture"
)
def test_a_filtered_process_cannot_execute_another_program():
    """A plugin that can exec is a plugin whose declared behaviour means nothing."""
    result = run_child(
        """
        install_filter()
        try:
            os.execv("/bin/echo", ["echo", "REPLACED"])
        except PermissionError as error:
            print("EXEC-DENIED", error)
        """
    )

    assert "EXEC-DENIED" in result.stdout
    assert "REPLACED" not in result.stdout


@pytest.mark.skipif(
    current_machine() not in SYSCALLS, reason="no syscall table for this architecture"
)
def test_a_filtered_process_cannot_fork_a_child_that_outlives_supervision():
    result = run_child(
        """
        install_filter()
        try:
            os.fork()
            print("FORKED")
        except OSError as error:
            print("FORK-DENIED", error)
        """
    )

    assert "FORK-DENIED" in result.stdout
    assert "FORKED" not in result.stdout


@pytest.mark.skipif(
    current_machine() not in SYSCALLS, reason="no syscall table for this architecture"
)
def test_a_filtered_process_can_still_start_a_thread():
    result = run_child(
        """
        import threading
        install_filter()
        seen = []
        thread = threading.Thread(target=lambda: seen.append(1))
        thread.start()
        thread.join()
        print("THREAD-OK", seen)
        """
    )

    assert "THREAD-OK [1]" in result.stdout


@pytest.mark.skipif(
    current_machine() not in SYSCALLS, reason="no syscall table for this architecture"
)
def test_a_filtered_process_cannot_spawn_a_subprocess():
    result = run_child(
        """
        import subprocess
        install_filter()
        try:
            subprocess.run(["/bin/echo", "SPAWNED"], capture_output=True)
            print("SPAWN-ALLOWED")
        except (OSError, PermissionError) as error:
            print("SPAWN-DENIED", type(error).__name__)
        """
    )

    assert "SPAWN-DENIED" in result.stdout
    assert "SPAWN-ALLOWED" not in result.stdout


def test_a_worker_that_cannot_be_confined_refuses_to_run():
    """Believing it is confined while it is not is the worst of the outcomes."""

    class Libc:
        def __init__(self, prctl_result=0, seccomp_result=-1):
            self._prctl_result = prctl_result
            self._seccomp_result = seccomp_result

        def prctl(self, *_arguments):
            return self._prctl_result

        def syscall(self, *_arguments):
            return self._seccomp_result

    with pytest.raises(SeccompInstallError, match="seccomp filter"):
        install_filter(machine="x86_64", libc=Libc())
    with pytest.raises(SeccompInstallError, match="no_new_privs"):
        install_filter(machine="x86_64", libc=Libc(prctl_result=-1))


def test_installing_reports_how_large_the_applied_filter_was():
    class Libc:
        def prctl(self, *_arguments):
            return 0

        def syscall(self, *_arguments):
            return 0

    assert install_filter(machine="x86_64", libc=Libc()) == instruction_count(
        worker_filter("x86_64")
    )
