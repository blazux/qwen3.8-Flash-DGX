#!/usr/bin/env bash
# Download the DFlash2 draft model into the local Hugging Face cache.
# Resumable — safe to re-run if the connection drops, and safe to interrupt: partial
# blobs are kept as .incomplete and resumed where they left off, so a completed
# download re-checks in seconds.
#
#   scripts/download-draft.sh                                  # the default draft (incoai/Qwen3.8-Flash-Next-DFlash2, ~1.2 GiB)
#   DRAFT_MODEL=<org/name> scripts/download-draft.sh           # some other draft model
#   MAX_WORKERS=16 scripts/download-draft.sh                   # more parallel connections
#
# This is a draft model for speculative decoding, not a standalone model: it needs no
# preparation (no prepare-hybrid.sh/graft) — scripts/serve.sh with DRAFT_MODEL set points
# the vLLM speculative config straight at the snapshot.
#
# Gated repos: run `hf auth login` once. The token lands in the HF cache, which is
# mounted into the container, so it is picked up without exporting HF_TOKEN.
set -euo pipefail

DRAFT_MODEL="${DRAFT_MODEL:-incoai/Qwen3.8-Flash-Next-DFlash2}"
IMAGE="${IMAGE:-qwen38-flash-dgx:v0.32}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
MAX_WORKERS="${MAX_WORKERS:-8}"
mkdir -p "$HF_CACHE"

# The default draft is a single ~1.2 GiB shard — under the Hub's 50 GB Xet threshold,
# so plain HTTPS is fine. (Xet stays available; it is used transparently if a draft
# repo's large files need it.)

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

echo ">> downloading $DRAFT_MODEL into $HF_CACHE (resumable, $MAX_WORKERS workers)"
docker run --rm --name qwen38-draft-dl \
  -e HF_HOME=/hf -e HF_HUB_DISABLE_XET=0 -e HF_XET_HIGH_PERFORMANCE=1 \
  "${TOKEN_ARGS[@]}" \
  -v "$HF_CACHE:/hf" --entrypoint bash "$IMAGE" \
  -c "hf download '$DRAFT_MODEL' --max-workers $MAX_WORKERS"

echo ">> done. Verify with:  ./flash doctor (the DFlash2 line)  or  scripts/serve.sh with DRAFT_MODEL set"
