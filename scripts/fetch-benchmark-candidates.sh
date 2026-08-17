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
  # The same 8B as production, one quantization finer. Measured on the
  # integrated lane, not the discrete one: 5.45 GiB of weights leaves the
  # 32,768-token cache about half a gigabyte short of the 8 GiB card, the same
  # margin the 14B misses by. On this card q4_k_m is the only 8B that fits.
  "qwen3-8b-q5-k-m|1.0.0|068bae163faa96ad48032daf4e071a6a28fe67d8dcc95367609c2ff165e52738|https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q5_K_M.gguf"

  # Qwen3.5, the successor generation to what production runs. IQ4_XS is the
  # finest quantization of the 9B that fits the discrete card's weights budget
  # of about 5 GiB once the cache is paid for.
  "qwen3-5-9b-iq4-xs|1.0.0|7e918aeca06c52bcb528ea6b04b4ec957e75ee8c0a73138854c0dfcf371ea429|https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-IQ4_XS.gguf"
  # The same 9B at q4_k_m for the integrated lane, where the ceiling is 45 GiB
  # rather than 8 and the question stops being what fits.
  "qwen3-5-9b-q4-k-m|1.0.0|03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8|https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf"
  # A mixture of experts: 35B of weights, about 3B of them read per token. On a
  # bandwidth-bound lane that is the shape that matters — the integrated card
  # has room for all 20.5 GiB and only pays for the experts each token wakes.
  # If this is both better and faster than the dense 8B, it changes the answer.
  "qwen3-5-35b-a3b-q4-k-m|1.0.0|3b46d1066bc91cc2d613e3bc22ce691dd77e6f0d33c9060690d24ce6de494375|https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/main/Qwen3.5-35B-A3B-Q4_K_M.gguf"
  # The same 9B distilled from DeepSeek-V4-Flash. V4-Flash itself is 291B and
  # would need ~145 GiB at q4 against 90 GiB of RAM, so this distillation is
  # the only way that behaviour reaches this desk. Judged against the plain 9B
  # at the same quantization on the same lane, which is the only fair reading.
  "qwen3-5-9b-deepseek-v4-flash-q4-k-m|1.0.0|9be227448d319e6a7acca8056b71bf7d9a2c6b2811986e6658a9dedc208d0ada|https://huggingface.co/Jackrong/Qwen3.5-9B-DeepSeek-V4-Flash-GGUF/resolve/main/Qwen3.5-9B-DeepSeek-V4-Flash-Q4_K_M.gguf"
)

# Not fetched, and why — so nobody spends an afternoon rediscovering it:
#   Kimi K3        2.78 trillion parameters; the pruned community GGUF is a
#                  512 GB file. Three orders of magnitude out, not a near miss.
#   DeepSeek V4    291B for Flash, more for Pro. ~145 GiB at q4 against 90 GiB
#                  of system RAM; an IQ2 build would leave nothing for the
#                  desktop and answer at about a token per second.
# Both are represented here by the 9B distilled from V4-Flash instead.
#
# These Qwen3.5 builds are multimodal upstream; the vision projector is a
# separate `mmproj` file that is deliberately not fetched. The workloads are
# text, and the text path loads without it.

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
