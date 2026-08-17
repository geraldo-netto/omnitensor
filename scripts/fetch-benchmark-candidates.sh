#!/usr/bin/env bash
# Fetch the larger candidates the accuracy/speed question needs, into a root
# that is NOT the artifact store. Nothing here installs, qualifies, or changes
# a default: these are models to measure, and deciding on one is a separate act
# with its own commit and its own receipt.
#
# Each file is pinned by the sha256 its publisher serves and verified after the
# download; a mismatch deletes what arrived rather than leaving it to be
# measured. Re-running skips what is already present and verified.
#
#   scripts/fetch-benchmark-candidates.sh [root]
#
# Default root: ~/.local/share/omnitensor/benchmark-candidates
set -euo pipefail

ROOT="${1:-$HOME/.local/share/omnitensor/benchmark-candidates}"

# id · version · sha256 · url
CANDIDATES=(
  # 14B at the only quantization small enough to have a chance on an 8 GiB card.
  "qwen3-14b-ud-iq3-xxs|1.0.0|346f4dcf6bd85dfc6145eeb0da8efdfacae879cb9b0c8fe5cad85e0ae0ae80ce|https://huggingface.co/unsloth/Qwen3-14B-GGUF/resolve/main/Qwen3-14B-UD-IQ3_XXS.gguf"
  # 14B at the quantization the integrated lane has room for.
  "qwen3-14b-q4-k-m|1.0.0|500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0|https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/main/Qwen3-14B-Q4_K_M.gguf"
  # The same 8B as production, one quantization finer: is the loss q4 costs
  # worth more than the parameters 14B adds?
  "qwen3-8b-q5-k-m|1.0.0|068bae163faa96ad48032daf4e071a6a28fe67d8dcc95367609c2ff165e52738|https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q5_K_M.gguf"
)

for entry in "${CANDIDATES[@]}"; do
  IFS='|' read -r id version sha url <<<"$entry"
  target="$ROOT/$id/$version/model.gguf"
  if [ -f "$target" ] && [ "$(sha256sum "$target" | cut -d' ' -f1)" = "$sha" ]; then
    echo "$id: already here and verified"
    continue
  fi
  mkdir -p "$(dirname "$target")"
  echo "$id: fetching $(basename "$url")"
  curl -fL --retry 3 --retry-delay 5 -C - -o "$target.part" "$url"
  actual="$(sha256sum "$target.part" | cut -d' ' -f1)"
  if [ "$actual" != "$sha" ]; then
    rm -f "$target.part"
    echo "$id: digest mismatch — expected $sha, got $actual" >&2
    exit 1
  fi
  mv "$target.part" "$target"
  echo "$id: verified"
done

echo "candidates in $ROOT"
