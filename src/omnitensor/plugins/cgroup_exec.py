"""Join a prepared worker cgroup, then replace this trusted shim with the worker."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from pathlib import Path

from .budgets import join_worker_cgroup


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if len(arguments) < 2:
        raise SystemExit("usage: cgroup_exec CGROUP COMMAND [ARG ...]")
    cgroup = Path(arguments[0])
    command = arguments[1:]
    join_worker_cgroup(cgroup)
    os.execvp(command[0], command)
    return 127  # pragma: no cover - a successful exec never returns


if __name__ == "__main__":  # pragma: no cover - exercised through the launcher
    raise SystemExit(main())
