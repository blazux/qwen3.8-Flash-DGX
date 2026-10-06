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

MODEL="${MODEL:-nvidia/Qwen3.8-Flash-Next-NVFP4}"   # default since 2026-09-14; works unchanged on RadixArk/Qwen3.8-Flash-Next-NVFP4
IMAGE="${IMAGE:-qwen38-flash-dgx:v0.31}"
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

# The hybrid directory is a copy of the snapshot AS IT IS NOW. A download that stopped early (this
# checkpoint is 24 files, one of them 50 GiB; Xet sometimes ends on a ReadTimeout) leaves a snapshot
# without tokenizer.json or a shard, and a hybrid layout prepared from it stays incomplete forever:
# vLLM then fails on "Couldn't instantiate the backend tokenizer ... sentencepiece" (issue #17).
# So: refuse an incomplete snapshot, and when the layout already exists, repair it instead of exiting.
ESSENTIAL="config.json generation_config.json tokenizer.json tokenizer_config.json chat_template.jinja preprocessor_config.json model.safetensors.index.json"
missing_in() {  # <dir> [shards] -> names of essential files (and, with 'shards', of index shards) missing or dangling
  python3 - "$1" "${2:-}" "$ESSENTIAL" <<'PY'
import json, os, sys
d, shards, ess = sys.argv[1], sys.argv[2], sys.argv[3].split()
# quantization config: hf_quant_config.json (ModelOpt) OR quantization_config inside config.json
# (compressed-tensors checkpoints, e.g. orcarouter/lychee888 derivatives, ship only the latter — #23)
def quant_cfg(d):
    if os.path.exists(os.path.join(d, "hf_quant_config.json")):
        return "hf_quant_config.json"
    try:
        c = json.load(open(os.path.join(d, "config.json")))
        q = c.get("quantization_config") or (c.get("text_config") or {}).get("quantization_config")
        return "config.json" if q else None
    except Exception:
        return None
miss = [f for f in ess if not os.path.exists(os.path.join(d, f))]
if "config.json" not in miss and not quant_cfg(d):
    miss.append("quantization-config(hf_quant_config.json-or-config.json:quantization_config)")
if shards and "model.safetensors.index.json" not in miss:
    try:
        wm = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        miss += sorted(f for f in set(wm.values()) if not os.path.exists(os.path.join(d, f)))
    except Exception as e:
        miss.append(f"model.safetensors.index.json(unreadable: {e})")
print(" ".join(miss))
PY
}
MISS="$(missing_in "$SNAP_HOST" shards)"
if [ -n "$MISS" ]; then
  echo "!! snapshot INCOMPLETE, missing or dangling: $MISS"
  echo "   re-run scripts/download-weights.sh (resumable, re-checks in seconds), then this script again"
  exit 1
fi

# The conversion targets the bf16 side layers of a ModelOpt NVFP4 checkpoint. A compressed-tensors
# checkpoint (orcarouter / lychee888 derivatives: quant_method "compressed-tensors" in config.json) already
# ships those layers quantized, and its dispatch is vLLM's own, not our ModelOpt shim: nothing to convert.
QM="$(python3 - "$SNAP_HOST" <<'PY'
import json, os, sys
c = json.load(open(os.path.join(sys.argv[1], "config.json")))
q = c.get("quantization_config") or (c.get("text_config") or {}).get("quantization_config") or {}
print(q.get("quant_method") or ("compressed-tensors" if "config_groups" in q else "modelopt-or-unknown"))
PY
)"
if [ "$QM" = "compressed-tensors" ]; then
  echo "!! $MODEL is a compressed-tensors checkpoint (quantization_config in config.json): its dense side layers are"
  echo "   already quantized and vLLM dispatches them natively — there is nothing for the hybrid conversion to do."
  echo "   Serve it as published:  MODE=nvfp4 scripts/serve.sh   (or ./flash serve published)"
  exit 1
fi

if [ -f "$DST/.prepared" ]; then
  STALE="$(missing_in "$DST")"
  if [ -z "$STALE" ]; then echo ">> already prepared: $DST"; exit 0; fi
  echo ">> already prepared but missing $STALE (prepared from an incomplete download) — repairing from the snapshot"
  docker run --rm --name qwen38-fp8repair -v "$HF_CACHE:/hf" --entrypoint bash "$IMAGE" -c "
set -euo pipefail
for f in '$SRC_IN'/*; do
  b=\$(basename \"\$f\")
  case \"\$b\" in *.safetensors|model.safetensors.index.json) continue ;; esac
  [ -e '$DST_IN'/\"\$b\" ] || { cp -a \"\$f\" '$DST_IN'/; echo \"   + \$b\"; }
done"
  STALE="$(missing_in "$DST")"
  [ -z "$STALE" ] || { echo "!! still missing after repair: $STALE"; exit 1; }
  echo ">> repaired: $DST"
  exit 0
fi

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
