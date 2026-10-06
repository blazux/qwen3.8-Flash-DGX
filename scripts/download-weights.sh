#!/usr/bin/env bash
# Download a Qwen3.8-Flash-Next checkpoint into the local Hugging Face cache.
# Resumable — safe to re-run if the connection drops, and safe to interrupt: partial
# blobs are kept as .incomplete and resumed where they left off.
#
#   scripts/download-weights.sh                    # the default checkpoint, nvidia/Qwen3.8-Flash-Next-NVFP4 (~124 GiB, Xet)
#   MODEL=RadixArk/Qwen3.8-Flash-Next-NVFP4 scripts/download-weights.sh   # the previous default (~122 GiB, 418 files)
#   MODEL=<org/name> scripts/download-weights.sh   # some other checkpoint
#   MODEL=<org/name> EXCLUDE='glob1 glob2' scripts/download-weights.sh
#   MAX_WORKERS=24 scripts/download-weights.sh     # more parallel connections
#   XET=0 scripts/download-weights.sh              # plain HTTPS instead of Xet (see below)
#
# EXCLUDE takes space-separated globs and skips those files -- for a checkpoint
# where you already have an equivalent copy of one large shard, or do not intend to
# serve it. On some checkpoints the PLE n-gram table alone is a ~102 GB shard, so
# skipping one file can halve the download. Only skip a shard you can account for:
# the index still names it, so whatever consumes the checkpoint has to be told
# where those tensors live instead.
#
# XET=1 (the default since 2026-09-13) uses the Xet backend; XET=0 falls back to plain HTTPS.
# Xet used to be off because it stalled on some Spark setups, but the Hub now refuses to
# serve files over 50 GB through the plain path at all (the NVIDIA checkpoint's PLE table is a
# single 50 GiB shard, and the error it prints -- "install hf_xet" -- is misleading: hf_xet is
# in the image, HF_HUB_DISABLE_XET=1 was what blocked it). Measured here on a DGX Spark:
# ~105 MB/s with Xet against ~15 MB/s without, two clean exits. If Xet stalls for you, XET=0.
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

MODEL="${MODEL:-nvidia/Qwen3.8-Flash-Next-NVFP4}"   # default since 2026-09-14; see README "Checkpoints"
IMAGE="${IMAGE:-qwen38-flash-dgx:v0.31}"          # or the upstream image; only needs `hf`
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
EXCLUDE="${EXCLUDE:-}"
MAX_WORKERS="${MAX_WORKERS:-8}"
XET="${XET:-1}"
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
# Xet by default (files over 50 GB need it); XET=0 = HF_HUB_DISABLE_XET=1, plain HTTPS.
XET_ENV=(-e HF_HUB_DISABLE_XET=0 -e HF_XET_HIGH_PERFORMANCE=1)
[ "$XET" = 0 ] && XET_ENV=(-e HF_HUB_DISABLE_XET=1)
docker run --rm --name qwen38-dl \
  -e HF_HOME=/hf "${XET_ENV[@]}" \
  "${TOKEN_ARGS[@]}" \
  -v "$HF_CACHE:/hf" --entrypoint bash "$IMAGE" \
  -c "hf download '$MODEL' --max-workers $MAX_WORKERS$EXCL_FLAGS"

echo ">> done. Verify with:  scripts/serve.sh"
