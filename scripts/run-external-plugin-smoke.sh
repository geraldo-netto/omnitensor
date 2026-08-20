#!/usr/bin/env bash
# Build the two wheels, install them into an empty environment, and run one
# plugin job from outside the checkout.
#
# The steps lived only in `.github/workflows/external-plugin.yml`, so the one
# gate that proves an installed plugin works without the source tree could not
# be reproduced locally: a developer who broke the wheel found out from CI, and
# the workflow was free to drift from anything anybody could run. The same
# reason `run-provider-suites.sh` exists.
#
# The environment is deliberately built with `--no-deps` off and nothing from
# the checkout on `sys.path`: the point of the gate is that the wheels carry
# what they need. Run from anywhere; `--keep` leaves the environment behind to
# inspect.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="${OMNI_PYTHON:-$root/.venv/bin/python}"
if [[ ! -x "$python" ]]; then
    python="$(command -v python3)"
fi

template="$root/examples/omnitensor-plugin-template"
keep=0
for argument in "$@"; do
    case "$argument" in
        --keep) keep=1 ;;
        *) echo "usage: $(basename "$0") [--keep]" >&2; exit 2 ;;
    esac
done

work="$(mktemp -d)"
cleanup() {
    if (( keep )); then
        echo "environment kept at $work"
    else
        rm -rf "$work"
    fi
}
trap cleanup EXIT

echo "== building wheels"
"$python" -m build --wheel --no-isolation --outdir "$work/dist" "$root"
"$python" -m build --wheel --no-isolation --outdir "$work/dist" "$template"

echo "== installing into an empty environment"
"$python" -m venv "$work/venv"
# Globbed here rather than in the install line so an empty `dist` fails loudly
# instead of installing nothing and passing.
shopt -s nullglob
wheels=("$work"/dist/*.whl)
shopt -u nullglob
if (( ${#wheels[@]} < 2 )); then
    echo "expected the runtime and template wheels, found ${#wheels[@]}" >&2
    exit 1
fi
"$work/venv/bin/python" -m pip install --quiet --upgrade pip
"$work/venv/bin/python" -m pip install --quiet "${wheels[@]}"

echo "== executing one installed plugin job outside the checkout"
# `cd` out of the checkout so an accidental relative import of the source tree
# is a failure rather than a silent pass.
(
    cd "$work"
    "$work/venv/bin/omnitensor-plugin-smoke" \
        --plugin-id template-workload \
        --payload '{"values":[-1,0,1]}' \
        --configuration '{"scale":2}'
)
echo "external plugin smoke passed"
