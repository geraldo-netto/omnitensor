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
declare -a args=()
for argument in "$@"; do
    case "$argument" in
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

for provider in "$root"/providers/*/; do
    name="$(basename "$provider")"
    if ! compgen -G "$provider/tests/test_*.py" > /dev/null; then
        # Said out loud rather than passed over: four of the six providers ship
        # no suite at all, and silence there reads exactly like success.
        skipped+=("$name")
        continue
    fi
    echo "== $name"
    if (cd "$provider" && "$python" -m pytest -q "${args[@]+"${args[@]}"}"); then
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
