#!/usr/bin/env bash
# Download a Qwen3.8-Flash-Next checkpoint into the local Hugging Face cache.
# Resumable — safe to re-run if the connection drops, and safe to interrupt: partial
# blobs are kept as .incomplete and resumed where they left off.
#
#   scripts/download-weights.sh                    # the default NVFP4 checkpoint (~126 GiB, 135 GB)
#   MODEL=<org/name> scripts/download-weights.sh   # some other checkpoint
#   MODEL=<org/name> EXCLUDE='glob1 glob2' scripts/download-weights.sh
#   MAX_WORKERS=24 scripts/download-weights.sh     # more parallel connections
#   XET=1 scripts/download-weights.sh              # Xet backend (see the caution below)
#
# EXCLUDE takes space-separated globs and skips those files. Its use is a
# compressed-tensors checkpoint whose BF16 PLE table you intend to replace with a
# donor's FP8 one (see scripts/prepare-ct.sh): that table is one ~102 GB shard, so
# skipping it turns a 183 GB download into 81 GB. Verify the donor first with
# tools/verify_ple_donor.py, which needs no download at all.
#
# XET=1 turns the Xet backend back on. It is OFF by default because it stalled on some
# Spark setups -- that is the reason for HF_HUB_DISABLE_XET below, not a speed judgement.
# On a fast link the difference is large: --max-workers only parallelises across *files*,
# so a checkpoint of a dozen large shards leaves most of a gigabit idle over plain HTTPS,
# while Xet issues concurrent ranged reads *within* each file. Measured here on a DGX
# Spark on gigabit fibre, pulling 81 GB of orcarouter/...-Uncensored-NVFP4:
#
#     plain HTTPS, 8 workers    14.7 MB/s   (117 Mbit/s)
#     XET=1                    101   MB/s   (809 Mbit/s, ~86% of line rate) -- 13.4 min
#
# Caveat, and why it stays off by default: that run still ended in an httpx.ReadTimeout
# after the last file finished. Every blob was complete and verified, but the exit code
# was non-zero, so a wrapper that trusts it will think the download failed. Re-run to
# confirm -- it is resumable and a completed download re-checks in seconds.
# Xet-backed repos are the ones whose API tree entries carry an "xetHash".
#
# MAX_WORKERS is worth raising when the Hub, rather than your link, is the limit.
# Check which it is: measure the NIC while the download runs
# (/sys/class/net/<if>/statistics/rx_bytes), then run a few parallel streams from an
# unrelated CDN. If the total goes well above what the download alone was getting,
# there is headroom and more workers will use it; if it does not, the pipe is full and
# more workers only add overhead.
#
# The default checkpoint needs ~140 GB free on the filesystem holding
# ~/.cache/huggingface; check the model's own file list for others.
#
# Gated repos: run `hf auth login` once. The token lands in the HF cache, which is
# mounted into the container, so it is picked up without exporting HF_TOKEN.
set -euo pipefail

MODEL="${MODEL:-RadixArk/Qwen3.8-Flash-Next-NVFP4}"
IMAGE="${IMAGE:-qwen38-flash-dgx}"          # or the upstream image; only needs `hf`
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
EXCLUDE="${EXCLUDE:-}"
MAX_WORKERS="${MAX_WORKERS:-8}"
XET="${XET:-0}"
mkdir -p "$HF_CACHE"

# One --exclude per glob. Patterns must not contain spaces (filenames don't).
EXCL_FLAGS=""
for pat in $EXCLUDE; do EXCL_FLAGS="$EXCL_FLAGS --exclude '$pat'"; done

# hf authenticates via HF_TOKEN (or the older HUGGING_FACE_HUB_TOKEN name).
# docker -e NAME (no value) copies the host env var into the container.
TOKEN_ARGS=()
if [ -n "${HF_TOKEN:-}" ]; then
  TOKEN_ARGS+=(-e HF_TOKEN)
elif [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  TOKEN_ARGS+=(-e HUGGING_FACE_HUB_TOKEN -e HF_TOKEN="$HUGGING_FACE_HUB_TOKEN")
elif [ -s "$HF_CACHE/token" ]; then
  echo ">> using the token from $HF_CACHE/token (hf auth login)"
else
  echo ">> no HF_TOKEN and no $HF_CACHE/token; Hub will rate-limit, and gated repos will 401"
fi

echo ">> downloading $MODEL into $HF_CACHE (resumable, $MAX_WORKERS workers)${EXCLUDE:+, excluding: $EXCLUDE}"
# HF_HUB_DISABLE_XET=1: the Xet backend stalled on some Spark setups; plain HTTPS is
# reliable, but on a fast link it leaves most of it idle -- see XET=1 above.
XET_ENV=(-e HF_HUB_DISABLE_XET=1)
[ "$XET" = 1 ] && XET_ENV=(-e HF_HUB_DISABLE_XET=0 -e HF_XET_HIGH_PERFORMANCE=1)
docker run --rm --name qwen38-dl \
  -e HF_HOME=/hf "${XET_ENV[@]}" \
  "${TOKEN_ARGS[@]}" \
  -v "$HF_CACHE:/hf" --entrypoint bash "$IMAGE" \
  -c "hf download '$MODEL' --max-workers $MAX_WORKERS$EXCL_FLAGS"

echo ">> done. Verify with:  scripts/serve.sh"
