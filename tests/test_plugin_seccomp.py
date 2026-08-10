

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
