#!/usr/bin/env bash
# One-time preparation of the "hybrid" checkpoint: NVFP4 experts as published, plus the
# dense side layers (GDN in/out projections, QSA q/k/v/o, shared experts — ~15 GiB of
# bf16) rewritten as blockwise fp8-e4m3 (DeepSeek layout: fp8 `weight` + fp32
# `weight_scale_inv`, 128x128 blocks). Those layers are read in full on every decoded
# token, so halving them is what buys the +20% decode and the extra KV.
#
#   scripts/prepare-hybrid.sh        # ~10 min, needs ~13 GB more disk
#   MODE=hybrid scripts/serve.sh
#
# Layout: a sibling of the HF snapshot, <snapshot>-fp8hybrid/, made of the same
# relative symlinks into blobs/ (so it resolves inside the container under /hf) — only
# the 4 converted shards are real files. Nothing in the original snapshot is touched.
# All filesystem work runs inside the container (the HF cache is usually root-owned
# after scripts/download-weights.sh). Conversion tool: tools/fp8_convert.py by
# @Saren-Arterius (Apache-2.0).
set -euo pipefail

MODEL="${MODEL:-RadixArk/Qwen3.8-Flash-Next-NVFP4}"
IMAGE="${IMAGE:-qwen38-flash-dgx}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

REPO_DIR="$HF_CACHE/hub/models--${MODEL//\//--}"
# Same revision resolution as scripts/serve.sh - the two must agree on the snapshot.
SNAP_HOST=""
for REF in main master; do
  REV="$(cat "$REPO_DIR/refs/$REF" 2>/dev/null || true)"
  if [ -n "$REV" ] && [ -d "$REPO_DIR/snapshots/$REV" ]; then SNAP_HOST="$REPO_DIR/snapshots/$REV/"; break; fi
done
SNAP_HOST="${SNAP_HOST:-$(ls -dt "$REPO_DIR"/snapshots/*/ 2>/dev/null | grep -v -- '-fp8hybrid' | head -1 || true)}"
[ -n "$SNAP_HOST" ] || { echo "!! checkpoint not found under $REPO_DIR — run scripts/download-weights.sh first"; exit 1; }
SNAP_NAME="$(basename "$SNAP_HOST")"
DST="$REPO_DIR/snapshots/${SNAP_NAME}-fp8hybrid"
SRC_IN="/hf/hub/models--${MODEL//\//--}/snapshots/${SNAP_NAME}"
DST_IN="${SRC_IN}-fp8hybrid"

if [ -f "$DST/.prepared" ]; then echo ">> already prepared: $DST"; exit 0; fi

echo ">> preparing $DST (relative symlinks + 4 shards converted to blockwise fp8, ~10 min)"
docker run --rm --name qwen38-fp8convert \
  -v "$HF_CACHE:/hf" -v "$PWD/tools:/tools:ro" --entrypoint bash "$IMAGE" -c "
set -euo pipefail
rm -rf '$DST_IN'
# cp -a keeps the relative symlinks (../../blobs/<sha>) as symlinks: instant, no copy.
cp -a '$SRC_IN' '$DST_IN'
# The converter rewrites the index: make it a real file first.
cp --remove-destination \"\$(readlink -f '$DST_IN/model.safetensors.index.json')\" '$DST_IN/model.safetensors.index.json'
python3 /tools/fp8_convert.py '$DST_IN' | tee '$DST_IN/fp8_convert.log'
# The converter moves each rewritten shard's symlink to <shard>.bf16.bak (a symlink,
# costs nothing) and writes the fp8 shard as a real file.
n=\$(python3 -c \"import json;print(sum(1 for k in json.load(open('$DST_IN/model.safetensors.index.json'))['weight_map'] if k.endswith('weight_scale_inv')))\")
[ \"\$n\" -gt 0 ] || { echo '!! conversion produced no fp8 tensors'; exit 1; }
chmod 644 '$DST_IN'/*.safetensors '$DST_IN'/model.safetensors.index.json
touch '$DST_IN/.prepared'
echo \">> done: \$n fp8 side-layer tensors\"
"
echo ">> hybrid checkpoint ready. Serve with:  MODE=hybrid scripts/serve.sh"
