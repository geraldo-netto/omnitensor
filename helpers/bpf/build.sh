#!/usr/bin/env bash
# Build the CO-RE object for this host's kernel.
#
# vmlinux.h is generated rather than committed: it is ~166k lines describing
# one kernel's exact types, and a stale copy would silently compile against
# layouts the running kernel does not have.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="${1:-$here/build}"
mkdir -p "$out"

command -v clang >/dev/null || { echo "clang is required to build the BPF object" >&2; exit 1; }
command -v bpftool >/dev/null || { echo "bpftool is required to generate vmlinux.h" >&2; exit 1; }
[ -r /sys/kernel/btf/vmlinux ] || { echo "kernel BTF is unavailable; CO-RE cannot be built" >&2; exit 1; }

bpftool btf dump file /sys/kernel/btf/vmlinux format c > "$out/vmlinux.h"

includes=()
for candidate in \
    /usr/include \
    /usr/src/linux-headers-*/tools/bpf/resolve_btfids/libbpf/include \
    /usr/src/linux-headers-*/tools/lib
do
    for path in $candidate; do
        [ -r "$path/bpf/bpf_helpers.h" ] && includes+=("-I$path")
    done
done
[ ${#includes[@]} -gt 0 ] || { echo "libbpf headers not found; install libbpf-dev" >&2; exit 1; }

arch="$(uname -m)"
case "$arch" in
    x86_64) target=__TARGET_ARCH_x86 ;;
    aarch64) target=__TARGET_ARCH_arm64 ;;
    *) echo "unsupported architecture for the BPF helper: $arch" >&2; exit 1 ;;
esac

clang -O2 -g -target bpf "-D$target" -I"$out" "${includes[@]}" \
    -Wno-missing-declarations -c "$here/runq_latency.bpf.c" -o "$out/runq_latency.bpf.o"
echo "built $out/runq_latency.bpf.o"
