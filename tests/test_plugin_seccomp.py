

def test_the_supported_architectures_are_the_ones_with_a_table():
    """Stated so an operator on another machine learns it from the error and
    the packaging metadata rather than from every worker failing to start."""
    from omnitensor.plugins.seccomp import SUPPORTED_MACHINES, SYSCALLS

    assert sorted(SUPPORTED_MACHINES) == sorted(SYSCALLS)
    assert tuple(sorted(SUPPORTED_MACHINES)) == SUPPORTED_MACHINES, "stable order"


def test_an_unsupported_architecture_names_the_ones_that_work():
    from omnitensor.plugins.seccomp import SUPPORTED_MACHINES, SeccompUnsupportedError

    message = str(SeccompUnsupportedError("riscv64"))

    assert "riscv64" in message
    for machine in SUPPORTED_MACHINES:
        assert machine in message
    assert "--no-seccomp" in message, "the override is named where the failure is read"


def test_confinement_is_probed_before_it_is_attempted():
    """Failing closed is right; being undiagnosable is not.

    On a kernel without CONFIG_SECCOMP_FILTER every worker refused to start,
    correctly, and nothing said the kernel was why.
    """
    from omnitensor.plugins.seccomp import confinement_error

    assert confinement_error("riscv64").startswith("no seccomp syscall table")
    # This host runs the suite, so it must be confinable — otherwise the
    # isolation the tests assume is not the isolation that ships.
    assert confinement_error() == ""


def test_a_kernel_that_reports_no_seccomp_support_is_named(monkeypatch, tmp_path):
    from omnitensor.plugins import seccomp

    status = tmp_path / "status"
    status.write_text("Name:\tpython\nThreads:\t1\n", encoding="utf-8")
    monkeypatch.setattr(seccomp, "Path", lambda _p: status)

    reason = seccomp.confinement_error("x86_64")

    assert "does not report seccomp support" in reason


def test_an_unreadable_status_file_claims_nothing(monkeypatch):
    """The worker's own attempt stays the authority when nothing can be read."""
    from omnitensor.plugins import seccomp

    class Unreadable:
        def read_text(self, **_kwargs):
            raise OSError("no /proc here")

    monkeypatch.setattr(seccomp, "Path", lambda _p: Unreadable())

    assert seccomp.confinement_error("x86_64") == ""
