#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next on a single DGX Spark / GB10 with the PLE table mmapped
# from disk. OpenAI-compatible API on $PORT.
#
#   scripts/serve.sh                          # NVFP4 checkpoint as published, 262k ctx
#   MODE=hybrid scripts/serve.sh              # NVFP4 experts + fp8 side layers (scripts/prepare-hybrid.sh first)
#   MODE=hybrid-mtp scripts/serve.sh          # hybrid + NVFP4 MTP draft experts (scripts/prepare-mtp-graft.sh first)
#   YARN=1 CTX=500000 scripts/serve.sh        # 500k context via YaRN (validated)
#   docker logs -f qwen38-flash               # wait for "Application startup complete"
#
# Tunables (env):
#   MODE=nvfp4        nvfp4 = the checkpoint as published (side layers bf16)
#                     hybrid = side layers in blockwise fp8: +20% decode, +15-20% KV, same
#                     tournament score. Needs the one-time scripts/prepare-hybrid.sh
#                     hybrid-mtp = hybrid with the MTP draft experts in NVFP4 (grafted from
#                     Inferact's checkpoint): ~3.4 GB less on the card, ~4x fewer bytes read
#                     per draft step. Needs scripts/prepare-mtp-graft.sh
#   PREFIX_CACHE=1    1 = --enable-prefix-caching (correct with this image's block_size fix;
#                     repeated prefixes — system prompts, multi-turn, tool loops — skip the prefill)
#   DET_TOPK=1        1 = deterministic QSA top-k KERNEL (@jschmied, vllm#55122): identical output at
#                     temperature 0 at no prefill cost. The default.
#   EXACT_TOPK=0      1 = exact torch.topk fallback (also deterministic, but -20-40% long prefill); wins over DET_TOPK
#   PAD_M4=0          1 = pad M%4 in the blockwise-fp8 GEMM (@jschmied). Hybrid mode only; a no-op with
#                     PREFIX_CACHE=1 (chunks are 1600-aligned), about -40% TTFT at 8k with PREFIX_CACHE=0
#   DRAFT_VOCAB=1     1 = the MTP drafter scores only the 65,536 most frequent tokens (+20% decode, same
#                     tournament score); 0 = full vocabulary; a path = your own ids.npy (tools/build_draft_vocab.py)
#   MADVISE=random    madvise on the mmapped PLE table: random (default; no readahead, cleaner page cache) or normal
#   LOG_REQUESTS=0    1 = log every prompt and output (VLLM_LOGGING_LEVEL=DEBUG, --enable-log-requests
#                     --enable-log-outputs) for tools/vllm_watch.py. Debugging only: privacy + unbounded logs
#   PORT=18300        host port for the API
#   CTX=262144        max context length (native). With YARN=1 up to ~500000 (see README)
#   YARN=0            1 = YaRN rope scaling (factor 4) for CTX > 262144
#   SEQS=8            max concurrent sequences. Do NOT leave this at 1-2 when measuring
#                     throughput: requests queue silently and aggregate tok/s flatlines
#   KV_CACHE_MEM=     bytes for the KV cache, as --kv-cache-memory-bytes. GPU_MEM is a fraction of
#                     TOTAL device memory, so it also leaves out whatever was already resident;
#                     vLLM prints the exact value it would accept at startup ("Replace
#                     gpu_memory_utilization config with --kv-cache-memory=..."). On a Spark the
#                     headroom it reports is also what the page cache uses for the PLE table, so
#                     claiming it trades prefill for KV. Watch vllm:ple_mmap_gather_seconds_total
#   GPU_MEM=0.80      fraction of the 128 GB pool for weights+KV. 0.85 buys ~2 GiB of KV but the
#                     box drifted into swap after a day at it; 0.875 got OOM-killed on a 300k prefill
#   MTP=2             speculative tokens from the model's MTP head (0 = off)
#   KV_DTYPE=auto     auto (=bf16) recommended; fp8_e4m3 = x1.9 KV / 1M ctx at a speed+quality cost (README)
#   PREWARM=0         1 = stream the 48 GiB table once at boot to warm the page cache
#   WORKERS=32        threads for the mmap gather
#   EXTRA=            extra vllm flags passed verbatim
#   IMAGE=qwen38-flash-dgx   MODEL=RadixArk/Qwen3.8-Flash-Next-NVFP4
#   BASE=             preview|v0.29 — normally read from the image label (Dockerfile vs Dockerfile.v0.29).
#                     On v0.29: KV_DTYPE must stay auto (fp8 KV not ported), PAD_M4 is a no-op.
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
DRAFT_VOCAB="${DRAFT_VOCAB:-1}"
MADVISE="${MADVISE:-random}"
LOG_REQUESTS="${LOG_REQUESTS:-0}"
PORT="${PORT:-18300}"
CTX="${CTX:-262144}"
YARN="${YARN:-0}"
SEQS="${SEQS:-8}"
GPU_MEM="${GPU_MEM:-0.80}"
MTP="${MTP:-2}"
KV_DTYPE="${KV_DTYPE:-auto}"
KV_CACHE_MEM="${KV_CACHE_MEM:-}"
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
  hybrid|hybrid-mtp)
    SUFFIX="-fp8hybrid"
    [ "$MODE" = hybrid-mtp ] && SUFFIX="-fp8hybrid-mtpnvfp4"
    if [ ! -f "$REPO_DIR/snapshots/${SNAP_NAME}${SUFFIX}/.prepared" ]; then
      [ "$MODE" = hybrid ] && echo "!! hybrid checkpoint not prepared: run scripts/prepare-hybrid.sh first (one-time, ~10 min)" \
        || echo "!! hybrid-mtp checkpoint not prepared: run scripts/prepare-mtp-graft.sh first (needs prepare-hybrid.sh; one-time, ~5 min)"
      exit 1
    fi
    SNAP_NAME="${SNAP_NAME}${SUFFIX}"
    HYBRID_ENV=(-e VLLM_FP8_HYBRID=1 -e VLLM_USE_DEEP_GEMM=0)
    ;;
  *) echo "!! MODE must be nvfp4, hybrid or hybrid-mtp"; exit 1 ;;
esac
SNAP_IN="/hf/hub/models--${MODEL//\//--}/snapshots/$SNAP_NAME"

# The PLE gather is a CPU op + a pageable host->device copy: it MUST run outside
# CUDA graphs. We declare it a splitting op and use PIECEWISE capture (never FULL*).
# The splitting-op names depend on the base image: the preview names the model qwen3_8_flash_next and
# our PLE op is ple_mmap_lookup; vLLM >= 0.29 names it qwen4_exp and the op is ple_mmap_lookup_ids.
# Both Dockerfiles stamp a label so this picks the right list (BASE=preview|v0.29 overrides).
BASE="${BASE:-$(docker image inspect -f '{{index .Config.Labels "qwen38.base"}}' "$IMAGE" 2>/dev/null || true)}"
if [ "$BASE" = "v0.29" ]; then
  SPLIT='["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::qwen4_exp_compute_ple_ngram_ids","vllm::qwen4_exp_ple_short_conv","vllm::qwen4_exp_qsa_with_output","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed","vllm::sparse_attn_indexer","vllm::ple_mmap_lookup_ids"]'
else
  SPLIT='["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::qwen3_8_flash_next_ple_short_conv","vllm::qwen3_8_flash_next_qsa_with_output","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed","vllm::sparse_attn_indexer","vllm::ple_mmap_lookup"]'
fi
# Options the v0.29 image does not carry: patch 7 (fp8 KV on the QSA path) is not ported, and patch 9
# (M%4 padding) is unnecessary there (vllm#52775 is in the release) so the image has no such kernel.
if [ "$BASE" = "v0.29" ]; then
  if [ "$KV_DTYPE" != auto ]; then
    echo "!! KV_DTYPE=$KV_DTYPE: the fp8 KV cache patch is not ported to the v0.29 base yet — use the preview image (Dockerfile) for fp8 KV"; exit 1
  fi
  [ "$PAD_M4" != 0 ] && echo "!! PAD_M4 has no effect on the v0.29 base (vllm#52775 fixed the fp8 GEMM there); ignoring" && PAD_M4=0
fi
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
case "$DRAFT_VOCAB" in
  0|"") ;;
  1) DETENV+=(-e VLLM_MTP_DRAFT_VOCAB=/opt/llm/draft_vocab_65536.npy) ;;
  *) DETENV+=(-e VLLM_MTP_DRAFT_VOCAB="$DRAFT_VOCAB") ;;
esac
DETENV+=(-e VLLM_PLE_MMAP_MADVISE="$MADVISE")
LOGARGS=(); [ "$LOG_REQUESTS" = 1 ] && { DETENV+=(-e VLLM_LOGGING_LEVEL=DEBUG); LOGARGS=(--enable-log-requests --enable-log-outputs); }
PC_ARG=--no-enable-prefix-caching
[ "$PREFIX_CACHE" = 1 ] && PC_ARG=--enable-prefix-caching

docker rm -f "$NAME" >/dev/null 2>&1 || true
# If docker run itself fails (port already bound, ...) do not leave a Created container behind.
trap 'rc=$?; [ $rc -ne 0 ] && docker rm -f "$NAME" >/dev/null 2>&1; exit $rc' EXIT
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
    --kv-cache-dtype "$KV_DTYPE" ${KV_CACHE_MEM:+--kv-cache-memory-bytes "$KV_CACHE_MEM"} \
    "${OVR_ARGS[@]}" "${LOGARGS[@]}" $EXTRA \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    "${SPEC[@]}"

# Fail loudly instead of printing a success line over a dead container: give vLLM a few
# seconds to parse its arguments, then check the state (the status word only — the string
# also carries the exit code and the OOM flag for the message).
sleep 8
STATE="$(docker inspect -f '{{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}}' "$NAME" 2>/dev/null || echo 'missing')"
case "$STATE" in
  running*) ;;
  *)
    echo "!! $NAME is not running ($STATE) - last log lines:"
    docker logs --tail 25 "$NAME" 2>&1 | sed 's/^/   /'
    echo "   (bad flag in EXTRA, port ${PORT} in use, or the image is missing?)"
    exit 1
    ;;
esac

echo ">> $NAME starting on :$PORT (model 'qwen3.8-flash-next', mode=$MODE, ctx $CTX, yarn=$YARN, mtp=$MTP, seqs=$SEQS, prefix_cache=$PREFIX_CACHE, det_topk=$DET_TOPK, exact_topk=$EXACT_TOPK, pad_m4=$PAD_M4, draft_vocab=$DRAFT_VOCAB, madvise=$MADVISE)"
echo ">> first boot loads ~76 GiB of weights (~8-13 min). Follow:  docker logs -f $NAME"
echo ">> ready when the log says 'Application startup complete'. Then: scripts/smoke-test.sh"
