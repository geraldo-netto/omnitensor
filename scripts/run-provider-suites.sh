#!/usr/bin/env bash
# Run every provider distribution's own test suite.
#
# The root `pytest` collects `tests/` and never descends into `providers/`, so
# each provider suite runs only if something runs it deliberately. Nothing did:
# `providers/media-transcription/tests` sat at 35 failures of 107 that no gate
# reported, and a receipt once described a task that did not exist because the
# suite proving otherwise was never invoked. An invariant nothing runs is not
# an invariant.
#
# Each provider is a separate distribution with its own dependencies, so the
# suites are run one directory at a time against that provider's pyproject
# rather than merged into the root run.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="${OMNI_PYTHON:-$root/.venv/bin/python}"
if [[ ! -x "$python" ]]; then
    python="$(command -v python3)"
fi

install=0
coverage=0
declare -a args=()
for argument in "$@"; do
    case "$argument" in
        # Coverage per distribution, against the floor that distribution
        # declares. `--cov-fail-under=80` in the root run measures
        # `src/omnitensor` only, so `providers/` — a sixth of the runtime
        # source, shipped as six installed distributions — was measured by
        # nothing, and a provider module no test loads read exactly like one
        # at 100%. The suites cannot be merged into the root run: each
        # provider has its own dependencies. Requires the distributions to be
        # installed, so pass `--install` too on a fresh environment.
        --cov) coverage=1 ;;
        # The install loop used to live only in `.github/workflows/quality.yml`,
        # so a developer running this script exactly as the header invites hit
        # the same missing dependencies that once hid 35 failures of 107. It
        # lives here now and CI calls the script.
        --install) install=1 ;;
        *) args+=("$argument") ;;
    esac
done

if command -v uv > /dev/null 2>&1; then
    declare -a installer=(uv pip)
else
    declare -a installer=("$python" -m pip)
fi

install_providers() {
    local provider requirements
    for provider in "$root"/providers/*/; do
        echo "== installing $(basename "$provider")"
        "${installer[@]}" install --no-deps -e "$provider"
        # Captured rather than piped through a process substitution: a failure
        # here must stop the run, not become an empty requirements file.
        requirements="$("$python" "$root/scripts/provider-requirements.py" "$provider")"
        if [[ -n "$requirements" ]]; then
            printf '%s\n' "$requirements" | "${installer[@]}" install -r -
        fi
    done
}

if (( install )); then
    install_providers
fi

declare -a ran=() skipped=() failed=()

# What a distribution installs, read from its own wheel packages, and the floor
# it declares. Both are derived: a hand-typed list here would be one more place
# to forget a sixth distribution.
coverage_args() {
    "$python" - "$1" <<'PYTHON'
import sys
import tomllib
from pathlib import Path

project = tomllib.loads((Path(sys.argv[1]) / "pyproject.toml").read_text(encoding="utf-8"))
packages = project["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
floor = project.get("tool", {}).get("coverage", {}).get("report", {}).get("fail_under")
if floor is None:
    raise SystemExit("declare [tool.coverage.report] fail_under in the provider pyproject")
for package in packages:
    print(f"--cov={Path(package).name}")
print("--cov-report=term-missing")
print(f"--cov-fail-under={floor}")
PYTHON
}

for provider in "$root"/providers/*/; do
    name="$(basename "$provider")"
    if ! compgen -G "$provider/tests/test_*.py" > /dev/null; then
        # Said out loud rather than passed over: four of the six providers ship
        # no suite at all, and silence there reads exactly like success.
        skipped+=("$name")
        continue
    fi
    echo "== $name"
    declare -a scoped=()
    if (( coverage )); then
        mapfile -t scoped < <(coverage_args "$provider")
    fi
    if (cd "$provider" && "$python" -m pytest -q \
        "${scoped[@]+"${scoped[@]}"}" "${args[@]+"${args[@]}"}"); then
        ran+=("$name")
    else
        failed+=("$name")
    fi
done

echo
echo "suites passed:  ${ran[*]:-none}"
echo "suites failed:  ${failed[*]:-none}"
echo "no suite:       ${skipped[*]:-none}"

if (( ${#failed[@]} > 0 )); then
    exit 1
fi
