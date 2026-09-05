#!/usr/bin/env bash
# Download the NVFP4 checkpoint (~126 GiB, 135 GB) into the local Hugging Face cache.
# Resumable — safe to re-run if the connection drops.
#
#   scripts/download-weights.sh
#
# Needs ~140 GB free on the filesystem holding ~/.cache/huggingface.
set -euo pipefail

MODEL="${MODEL:-RadixArk/Qwen3.8-Flash-Next-NVFP4}"
IMAGE="${IMAGE:-qwen38-flash-dgx}"          # or the upstream image; only needs `hf`
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
mkdir -p "$HF_CACHE"

# hf authenticates via HF_TOKEN (or the older HUGGING_FACE_HUB_TOKEN name).
# docker -e NAME (no value) copies the host env var into the container.
TOKEN_ARGS=()
if [ -n "${HF_TOKEN:-}" ]; then
  TOKEN_ARGS+=(-e HF_TOKEN)
elif [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
  TOKEN_ARGS+=(-e HUGGING_FACE_HUB_TOKEN -e HF_TOKEN="$HUGGING_FACE_HUB_TOKEN")
else
  echo ">> no HF_TOKEN in the environment; Hub will rate-limit unauthenticated downloads"
fi

echo ">> downloading $MODEL into $HF_CACHE (resumable)"
# HF_HUB_DISABLE_XET=1: the Xet backend stalled on some Spark setups; plain HTTPS
# is reliable and saturates the link.
docker run --rm --name qwen38-dl \
  -e HF_HOME=/hf -e HF_HUB_DISABLE_XET=1 \
  "${TOKEN_ARGS[@]}" \
  -v "$HF_CACHE:/hf" --entrypoint bash "$IMAGE" \
  -c "hf download '$MODEL' --max-workers 8"

echo ">> done. Verify with:  scripts/serve.sh"
