"""Print one provider distribution's third-party dependencies, one per line.

The provider wheels depend on `omnitensor` and on each other by name and exact
version. Those are the working tree, not something to fetch: resolving them
from an index during CI would install a published copy over the code under
test. Everything else is a genuine third-party requirement and is exactly what
was missing when `providers/media-transcription/tests` failed 35 of 107 --
`av`, `cairosvg` and `defusedxml`, every one of them declared right here and
never installed.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

_LOCAL = re.compile(r"^omnitensor(-[a-z0-9-]+)?\b")


def requirements(pyproject: Path) -> list[str]:
    document = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = document.get("project", {})
    declared = list(project.get("dependencies", ()))
    for extra in project.get("optional-dependencies", {}).values():
        declared.extend(extra)
    return [item for item in declared if _LOCAL.match(item.strip()) is None]


if __name__ == "__main__":
    # CI reads this through a process substitution, so a traceback becomes an
    # empty requirements file and a silently under-installed environment. Say
    # what is wrong on stderr and exit non-zero instead.
    if len(sys.argv) != 2:
        raise SystemExit("usage: provider-requirements.py <provider-dir-or-pyproject.toml>")
    root = Path(sys.argv[1])
    path = root if root.name == "pyproject.toml" else root / "pyproject.toml"
    if not path.is_file():
        raise SystemExit(f"no pyproject.toml at {path}")
    print("\n".join(requirements(path)))
