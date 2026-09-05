#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next on a single DGX Spark / GB10 with the PLE table mmapped
# from disk. OpenAI-compatible API on $PORT.
#
#   scripts/serve.sh                          # NVFP4 checkpoint as published, 262k ctx
#   MODE=hybrid scripts/serve.sh              # NVFP4 experts + fp8 side layers (scripts/prepare-hybrid.sh first)
#   YARN=1 CTX=500000 scripts/serve.sh        # 500k context via YaRN (validated)
#   docker logs -f qwen38-flash               # wait for "Application startup complete"
#
# Tunables (env):
#   MODE=nvfp4        nvfp4 = the checkpoint as published (side layers bf16)
#                     hybrid = side layers in blockwise fp8: +20% decode, +15-20% KV, same
#                     tournament score. Needs the one-time scripts/prepare-hybrid.sh
#   PREFIX_CACHE=1    1 = --enable-prefix-caching (correct with this image's block_size fix;
#                     repeated prefixes — system prompts, multi-turn, tool loops — skip the prefill)
#   DET_TOPK=1        1 = deterministic QSA top-k KERNEL (@jschmied, vllm#55122): identical output at
#                     temperature 0 at no prefill cost. The default.
#   EXACT_TOPK=0      1 = exact torch.topk fallback (also deterministic, but -20-40% long prefill); wins over DET_TOPK
#   PAD_M4=0          1 = pad M%4 in the blockwise-fp8 GEMM (@jschmied). Hybrid mode only; a no-op with
#                     PREFIX_CACHE=1 (chunks are 1600-aligned), about -40% TTFT at 8k with PREFIX_CACHE=0
#   PORT=18300        host port for the API
#   CTX=262144        max context length (native). With YARN=1 up to ~500000 (see README)
#   YARN=0            1 = YaRN rope scaling (factor 4) for CTX > 262144
#   SEQS=8            max concurrent sequences. Do NOT leave this at 1-2 when measuring
#                     throughput: requests queue silently and aggregate tok/s flatlines
#   GPU_MEM=0.80      fraction of the 128 GB pool for weights+KV. 0.85 buys ~2 GiB of KV but the
#                     box drifted into swap after a day at it; 0.875 got OOM-killed on a 300k prefill
#   MTP=2             speculative tokens from the model's MTP head (0 = off)
#   KV_DTYPE=auto     auto (=bf16) recommended; fp8_e4m3 = x1.9 KV / 1M ctx at a speed+quality cost (README)
#   PREWARM=0         1 = stream the 48 GiB table once at boot to warm the page cache
#   WORKERS=32        threads for the mmap gather
#   EXTRA=            extra vllm flags passed verbatim
#   IMAGE=qwen38-flash-dgx   MODEL=RadixArk/Qwen3.8-Flash-Next-NVFP4
set -euo pipefail

NAME="${NAME:-qwen38-flash}"
IMAGE="${IMAGE:-qwen38-flash-dgx}"
MODEL="${MODEL:-RadixArk/Qwen3.8-Flash-Next-NVFP4}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
MODE="${MODE:-nvfp4}"
PREFIX_CACHE="${PREFIX_CACHE:-1}"
DET_TOPK="${DET_TOPK:-1}"
EXACT_TOPK="${EXACT_TOPK:-0}"
PAD_M4="${PAD_M4:-0}"
PORT="${PORT:-18300}"
CTX="${CTX:-262144}"
YARN="${YARN:-0}"
SEQS="${SEQS:-8}"
GPU_MEM="${GPU_MEM:-0.80}"
MTP="${MTP:-2}"
KV_DTYPE="${KV_DTYPE:-auto}"
PREWARM="${PREWARM:-0}"
EXTRA="${EXTRA:-}"

# Resolve the local snapshot directory and map it to the in-container mount.
REPO_DIR="$HF_CACHE/hub/models--${MODEL//\//--}"
# Pick the revision the cache actually points at (refs/main), not whichever snapshot
# sorts first: a cache that holds two revisions would otherwise serve the wrong one.
SNAP_HOST=""
for REF in main master; do
  REV="$(cat "$REPO_DIR/refs/$REF" 2>/dev/null || true)"
  if [ -n "$REV" ] && [ -d "$REPO_DIR/snapshots/$REV" ]; then SNAP_HOST="$REPO_DIR/snapshots/$REV/"; break; fi
done
SNAP_HOST="${SNAP_HOST:-$(ls -dt "$REPO_DIR"/snapshots/*/ 2>/dev/null | grep -v -- '-fp8hybrid' | head -1 || true)}"
if [ -z "$SNAP_HOST" ]; then
  echo "!! checkpoint not found under $REPO_DIR"
  echo "   run scripts/download-weights.sh first."
  exit 1
fi
SNAP_NAME="$(basename "$SNAP_HOST")"
HYBRID_ENV=()
case "$MODE" in
  nvfp4) ;;
  hybrid)
    if [ ! -f "$REPO_DIR/snapshots/${SNAP_NAME}-fp8hybrid/.prepared" ]; then
      echo "!! hybrid checkpoint not prepared: run scripts/prepare-hybrid.sh first (one-time, ~10 min)"
      exit 1
    fi
    SNAP_NAME="${SNAP_NAME}-fp8hybrid"
    HYBRID_ENV=(-e VLLM_FP8_HYBRID=1 -e VLLM_USE_DEEP_GEMM=0)
    ;;
  *) echo "!! MODE must be nvfp4 or hybrid"; exit 1 ;;
esac
SNAP_IN="/hf/hub/models--${MODEL//\//--}/snapshots/$SNAP_NAME"

# The PLE gather is a CPU op + a pageable host->device copy: it MUST run outside
# CUDA graphs. We declare it a splitting op and use PIECEWISE capture (never FULL*).
SPLIT='["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::qwen3_8_flash_next_ple_short_conv","vllm::qwen3_8_flash_next_qsa_with_output","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed","vllm::sparse_attn_indexer","vllm::ple_mmap_lookup"]'
CC="${CC:--cc.cudagraph_mode=PIECEWISE -cc.splitting_ops=$SPLIT}"

# YaRN (Qwen's published recipe) to go past the native 262144.
OVR_ARGS=()
YARN_OVR='{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}'
ALLOW_LONG=0
if [ "$YARN" != 0 ]; then OVR_ARGS=(--hf-overrides "$YARN_OVR"); ALLOW_LONG=1; fi

# MTP + YaRN: dict hf_overrides are not propagated to the draft model, whose
# max_model_len then stays 262144 and vLLM aborts with
# "--mamba-block-size can only be set with --enable-prefix-caching". Forcing the
# draft's max_model_len through the speculative config fixes it.
SPEC=()
if [ "$MTP" != 0 ]; then
  if [ "$YARN" != 0 ]; then
    SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP},\"max_model_len\":${CTX}}")
  else
    SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP}}")
  fi
fi

DETENV=(); [ "$DET_TOPK" = 1 ] && DETENV=(-e VLLM_QSA_DET_TOPK=1 -e VLLM_QSA_DET_LIB=/opt/llm/kernel-det/_C_det.so)
PC_ARG=--no-enable-prefix-caching
[ "$PREFIX_CACHE" = 1 ] && PC_ARG=--enable-prefix-caching

docker rm -f "$NAME" >/dev/null 2>&1 || true
# shellcheck disable=SC2086
docker run -d --name "$NAME" --restart unless-stopped \
  --gpus all --ipc=host --shm-size 16g -p "${PORT}:8000" \
  -v "$HF_CACHE:/hf" -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
  -e VLLM_PLE_MMAP=1 -e VLLM_PLE_MMAP_WORKERS="${WORKERS:-32}" -e VLLM_PLE_MMAP_PREWARM="$PREWARM" \
  -e VLLM_QSA_EXACT_TOPK="$EXACT_TOPK" "${DETENV[@]}" -e VLLM_FP8_PAD_M4="$PAD_M4" \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN="$ALLOW_LONG" \
  "${HYBRID_ENV[@]}" \
  "$IMAGE" \
  "$SNAP_IN" --served-model-name qwen3.8-flash-next \
    --host 0.0.0.0 --port 8000 --load-format safetensors \
    --max-model-len "$CTX" --max-num-seqs "$SEQS" --gpu-memory-utilization "$GPU_MEM" \
    $PC_ARG --enable-chunked-prefill --max-num-batched-tokens 8192 \
    $CC \
    --no-enable-flashinfer-autotune \
    --kv-cache-dtype "$KV_DTYPE" \
    "${OVR_ARGS[@]}" $EXTRA \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    "${SPEC[@]}"

sleep 5
STATE="$(docker inspect -f '{{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}}' "$NAME" 2>/dev/null || echo 'missing')"
if [ "$STATE" != running ]; then
  echo "!! $NAME is not running ($STATE) - last log lines:"
  docker logs --tail 25 "$NAME" 2>&1 | sed 's/^/   /'
  echo "   (port ${PORT} in use by another process, or the image is missing?)"
  exit 1
fi

echo ">> $NAME starting on :$PORT (model 'qwen3.8-flash-next', mode=$MODE, ctx $CTX, yarn=$YARN, mtp=$MTP, seqs=$SEQS, prefix_cache=$PREFIX_CACHE, det_topk=$DET_TOPK, exact_topk=$EXACT_TOPK, pad_m4=$PAD_M4)"
echo ">> first boot loads ~76 GiB of weights (~8-13 min). Follow:  docker logs -f $NAME"
echo ">> ready when the log says 'Application startup complete'. Then: scripts/smoke-test.sh"
