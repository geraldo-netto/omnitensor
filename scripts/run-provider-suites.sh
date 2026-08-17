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
    if (cd "$provider" && "$python" -m pytest -q "$@"); then
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
