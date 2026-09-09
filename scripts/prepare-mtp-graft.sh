#!/usr/bin/env bash
# One-time graft of the NVFP4 MTP draft experts (Inferact/Qwen3.8-Flash-Next-NVFP4,
# nvfp4_experts_mtp.safetensors) on top of the -fp8hybrid checkpoint — the combination
# neither parent ships: fp8 PLE table (48 GiB mmap'd from disk instead of resident)
# plus NVFP4 MTP draft experts. No retraining, no requantization.
#
#   scripts/prepare-mtp-graft.sh         # needs scripts/prepare-hybrid.sh first; ~5 min, ~3.4 GB
#   MODE=hybrid-mtp scripts/serve.sh
#
# What the graft changes relative to the -fp8hybrid snapshot it builds on:
#   - model-bf16-00011.safetensors: rewritten WITHOUT the 2 fused BF16 MTP expert
#     tensors (mtp.layers.0.mlp.experts.{down_proj,gate_up_proj}, ~5 GB). They must
#     not reach the loader alongside the per-expert NVFP4 tensors: the weight
#     iterator walks whole files (the index only selects files, not tensors), so
#     dropping the keys from the index alone is not enough.
#   - +6,144 NVFP4 per-expert MTP tensors, symlinked from the donor repo
#     (same per-expert naming/layout as the main-model experts the engine already
#     loads; RoutedExperts.load_weights accepts both layouts in one mapping).
#   - model.safetensors.index.json: -2 fused keys, +6,144 donor keys.
#   - config.json AND hf_quant_config.json: the blanket "mtp.*"/"model.mtp.*"
#     exclusion globs (which keep the whole draft head in BF16) are replaced by the
#     donor's 29 explicit non-expert MTP module names, so the routed experts are
#     quantized while everything else under mtp. stays full-width. BOTH lists must
#     carry the fix: the draft's quant config and the target's read different files,
#     and fixing only one leaves the MTP unquantized (load dies on missing
#     'w2_input_scale' / fused-tensor shape asserts). The globs cannot simply be
#     deleted: that would quantize the draft's attention, shared expert and fc_*
#     linears, whose tensors are BF16 (mtp.fc_embedding / mtp.fc_hidden are not
#     covered by any remaining pattern).
#
# Layout: a sibling of the hybrid snapshot made of relative symlinks (into the
# hybrid snapshot and through the cache root to the Inferact blob), so it resolves
# inside the container under /hf. Nothing in either parent snapshot is touched.
# Method: https://gist.github.com/thavoc/d7083457f6f2d981f879670c34df34ab and
# https://github.com/Peuqui/mtp-quant-transplant.
set -euo pipefail

MODEL="${MODEL:-RadixArk/Qwen3.8-Flash-Next-NVFP4}"
IMAGE="${IMAGE:-qwen38-flash-dgx}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"

REPO_DIR="$HF_CACHE/hub/models--${MODEL//\//--}"
# Same revision resolution as scripts/serve.sh and scripts/prepare-hybrid.sh.
SNAP_HOST=""
for REF in main master; do
  REV="$(cat "$REPO_DIR/refs/$REF" 2>/dev/null || true)"
  if [ -n "$REV" ] && [ -d "$REPO_DIR/snapshots/$REV" ]; then SNAP_HOST="$REPO_DIR/snapshots/$REV/"; break; fi
done
SNAP_HOST="${SNAP_HOST:-$(ls -dt "$REPO_DIR"/snapshots/*/ 2>/dev/null | grep -v -- '-fp8hybrid' | head -1 || true)}"
[ -n "$SNAP_HOST" ] || { echo "!! checkpoint not found under $REPO_DIR — run scripts/download-weights.sh first"; exit 1; }
SNAP_NAME="$(basename "$SNAP_HOST")"
HYBRID_NAME="${SNAP_NAME}-fp8hybrid"
GRAFT_NAME="${HYBRID_NAME}-mtpnvfp4"
[ -f "$REPO_DIR/snapshots/$HYBRID_NAME/.prepared" ] || {
  echo "!! hybrid checkpoint not prepared: run scripts/prepare-hybrid.sh first (one-time, ~10 min)"; exit 1; }
[ -f "$REPO_DIR/snapshots/$GRAFT_NAME/.prepared" ] && { echo ">> already prepared: $REPO_DIR/snapshots/$GRAFT_NAME"; exit 0; }

echo ">> grafting the NVFP4 MTP experts onto $HYBRID_NAME (downloads a 1.5 GiB donor shard once)"
docker run --rm -i --name qwen38-mtp-graft \
  -v "$HF_CACHE:/hf" --entrypoint python3 "$IMAGE" -u - <<'PYEOF'
import hashlib, json, os, re, struct, sys, urllib.request

REPO = "/hf/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4"
DONOR = "/hf/hub/models--Inferact--Qwen3.8-Flash-Next-NVFP4"
DONOR_REV = "103a7608316173ca6edd49929544244de7ffda70"
DONOR_SHA = "0d44e6d705d2313c713e60114e56874adf358ed5f646dc8704bb5be15f5ddbf7"
hybrids = sorted(d for d in os.listdir(f"{REPO}/snapshots") if d.endswith("-fp8hybrid"))
if len(hybrids) != 1:
    sys.exit(f"!! expected exactly one -fp8hybrid snapshot, found: {hybrids}")
HYBRID = f"{REPO}/snapshots/{hybrids[0]}"
GRAFT = HYBRID + "-mtpnvfp4"
DONOR_BLOB = f"{DONOR}/blobs/{DONOR_SHA}"
DONOR_SHARD = "nvfp4_experts_mtp.safetensors"
FUSED = ("mtp.layers.0.mlp.experts.down_proj", "mtp.layers.0.mlp.experts.gate_up_proj")

def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n

# --- 1. the donor blob: download once, pinned revision + sha256 ------------
os.makedirs(f"{DONOR}/blobs", exist_ok=True)
os.makedirs(f"{DONOR}/snapshots/{DONOR_REV}", exist_ok=True)
open(f"{DONOR}/refs/main", "w").write(DONOR_REV)
if not os.path.exists(DONOR_BLOB):
    url = f"https://huggingface.co/Inferact/Qwen3.8-Flash-Next-NVFP4/resolve/{DONOR_REV}/{DONOR_SHARD}"
    print(">> downloading donor shard (~1.5 GiB)")
    tmp = DONOR_BLOB + ".incomplete"
    h = hashlib.sha256()
    with urllib.request.urlopen(urllib.request.Request(url), timeout=300) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 22)
            if not chunk:
                break
            h.update(chunk)
            f.write(chunk)
    got = h.hexdigest()
    if got != DONOR_SHA:
        os.remove(tmp)
        sys.exit(f"!! donor sha256 mismatch: got {got}, want {DONOR_SHA}")
    os.rename(tmp, DONOR_BLOB)
link = f"{DONOR}/snapshots/{DONOR_REV}/{DONOR_SHARD}"
if not os.path.lexists(link):
    os.symlink(f"../../blobs/{DONOR_SHA}", link)

# --- 2. donor sanity: a clean, quantized draft-head shard ------------------
dhdr, _ = read_header(DONOR_BLOB)
dhdr.pop("__metadata__", None)
stray = [k for k in dhdr if not k.startswith("mtp.")]
if stray:
    sys.exit(f"!! donor shard holds {len(stray)} tensors outside mtp. (e.g. {stray[0]}) — refusing")
experts = sorted(k for k in dhdr if re.match(r"mtp\.layers\.0\.mlp\.experts\.\d+\.", k))
shared = sorted(k for k in dhdr if k not in set(experts))
print(f">> donor: {len(experts)} NVFP4 expert tensors + {len(shared)} shared (full-width)")
if len(experts) != 6144 or len(shared) != 29:
    sys.exit(f"!! unexpected donor layout: {len(experts)} experts / {len(shared)} shared")
if any(k in dhdr for k in FUSED):
    sys.exit("!! donor unexpectedly contains fused expert tensors")
bad = [k for k in experts
       if k.endswith(".weight")
       and (dhdr[k]["dtype"] != "U8" or k.removesuffix(".weight") + ".weight_scale" not in dhdr)]
if bad:
    sys.exit(f"!! donor expert tensors without U8 dtype or a weight_scale sibling: {bad[:3]}")
fullwidth = [k for k in shared if dhdr[k]["dtype"] not in ("BF16", "F16", "F32")]
if fullwidth:
    sys.exit(f"!! shared donor tensors not full-width: {fullwidth[:3]}")
# the 29 non-expert MTP modules that stay full-width once "mtp.*" is gone
additions = sorted({k.removesuffix(".weight") for k in shared})
for need in ("mtp.fc_embedding", "mtp.fc_hidden"):
    if need not in additions:
        sys.exit(f"!! {need} missing from the donor's shared tensors — the exclude fix would be wrong")

# --- 3. lay out the graft snapshot as relative symlinks --------------------
os.makedirs(GRAFT, exist_ok=True)
SKIP = ("model.safetensors.index.json", "config.json", "hf_quant_config.json",
        "model-bf16-00011.safetensors", DONOR_SHARD, ".prepared", "fp8_convert.log")
linked = 0
for name in sorted(os.listdir(HYBRID)):
    if name in SKIP:
        continue
    dst = os.path.join(GRAFT, name)
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(f"../{hybrids[0]}/{name}", dst)
    linked += 1
dst = os.path.join(GRAFT, DONOR_SHARD)
if os.path.lexists(dst):
    os.remove(dst)
os.symlink(f"../../../models--Inferact--Qwen3.8-Flash-Next-NVFP4/blobs/{DONOR_SHA}", dst)
print(f">> linked {linked} files from the hybrid snapshot + the donor shard")

# --- 4. rewrite the contaminated shard (hybrid's real 00011 minus the fused)
import torch
from safetensors.torch import load_file, save_file
tensors = load_file(f"{HYBRID}/model-bf16-00011.safetensors")
dropped = [k for k in FUSED if k in tensors]
if sorted(dropped) != sorted(FUSED):
    sys.exit(f"!! expected the 2 fused MTP expert tensors in the hybrid shard, found {dropped}")
for k in FUSED:
    del tensors[k]
out = f"{GRAFT}/model-bf16-00011.safetensors"
save_file(tensors, out, metadata={"format": "pt"})
size = os.path.getsize(out) / 2**30
del tensors
print(f">> rewrote model-bf16-00011.safetensors minus {len(dropped)} fused tensors ({size:.2f} GiB)")

# --- 5. rewrite the index --------------------------------------------------
index = json.load(open(f"{HYBRID}/model.safetensors.index.json"))
wm = index["weight_map"]
removed = [k for k in FUSED if k in wm]
for k in removed:
    del wm[k]
for k in experts:
    wm[k] = DONOR_SHARD
json.dump(index, open(f"{GRAFT}/model.safetensors.index.json", "w"))
print(f">> index: removed {len(removed)} fused keys, added {len(experts)} NVFP4 expert keys "
      f"({len(wm)} entries)")

# --- 6. fix BOTH exclude lists, identically --------------------------------
def fix(entries, source):
    kept = [p for p in entries if p not in ("mtp.*", "model.mtp.*")]
    dropped = [p for p in entries if p not in kept]
    if len(dropped) != 2:
        sys.exit(f"!! {source}: expected the 2 blanket mtp globs, found {dropped}")
    clash = [p for p in kept if p.startswith("mtp.") or p == "model.mtp"]
    if clash:
        sys.exit(f"!! {source}: explicit mtp entries already present: {clash}")
    return kept + additions, dropped

cfg = json.load(open(f"{HYBRID}/config.json"))       # read via symlink, write real
field = next((f for f in ("ignore", "exclude_modules")
              if isinstance(cfg.get("quantization_config", {}).get(f), list)), None)
if field is None:
    sys.exit("!! no ignore/exclude_modules list in config.json quantization_config")
cfg["quantization_config"][field], dropped_cfg = fix(cfg["quantization_config"][field], "config.json")
json.dump(cfg, open(f"{GRAFT}/config.json", "w"))

hqc = json.load(open(f"{HYBRID}/hf_quant_config.json"))
hqc["quantization"]["exclude_modules"], dropped_hqc = fix(hqc["quantization"]["exclude_modules"], "hf_quant_config.json")
json.dump(hqc, open(f"{GRAFT}/hf_quant_config.json", "w"))
print(f">> {field} and exclude_modules: dropped {dropped_cfg}, added {len(additions)} explicit "
      f"MTP module names (13 -> {len(cfg['quantization_config'][field])} entries)")

# --- 7. static verification (gate the .prepared marker on it) --------------
errors = []
for name in sorted(os.listdir(GRAFT)):
    p = os.path.join(GRAFT, name)
    if os.path.islink(p) and not os.path.exists(p):
        errors.append(f"dangling symlink: {name}")

index = json.load(open(f"{GRAFT}/model.safetensors.index.json"))
wm = index["weight_map"]
by_file = {}
for k, f in wm.items():
    by_file.setdefault(f, []).append(k)
donor_keys = set()
for fname, keys in by_file.items():
    path = os.path.join(GRAFT, fname)
    if not os.path.exists(path):
        errors.append(f"index references missing file: {fname}")
        continue
    hdr, _ = read_header(path)
    hdr.pop("__metadata__", None)
    for k in keys:
        if k not in hdr:
            errors.append(f"index key absent from shard: {k} -> {fname}")
    stray_fused = [k for k in FUSED if k in hdr]
    if stray_fused:
        errors.append(f"fused MTP tensors still in {fname}: {stray_fused}")
    if fname == DONOR_SHARD:
        donor_keys = set(keys)

if len(donor_keys) != 6144:
    errors.append(f"expected 6144 donor keys in the index, got {len(donor_keys)}")
gcfg = json.load(open(f"{GRAFT}/config.json"))
ghqc = json.load(open(f"{GRAFT}/hf_quant_config.json"))
ig = gcfg["quantization_config"][field]
em = ghqc["quantization"]["exclude_modules"]
if ig != em:
    errors.append("config.json and hf_quant_config.json exclude lists disagree")
if any(p in ig for p in ("mtp.*", "model.mtp.*")):
    errors.append("blanket mtp glob still present in the graft configs")
if len(ig) != 40:
    errors.append(f"expected 40 exclude entries after the fix, got {len(ig)}")
for need in ("mtp.fc_embedding", "mtp.fc_hidden"):
    if need not in ig:
        errors.append(f"{need} not in the graft exclude list — the draft fc linears would be quantized")
if gcfg.get("text_config", {}).get("ple_embedding_dtype") != "float8_e4m3fn":
    errors.append("ple_embedding_dtype is no longer float8_e4m3fn")
if errors:
    for e in errors:
        print(f"   !! {e}")
    sys.exit(f"!! static verification failed ({len(errors)} problems) — graft NOT marked prepared")

os.chmod(out, 0o644)
for j in ("model.safetensors.index.json", "config.json", "hf_quant_config.json"):
    os.chmod(f"{GRAFT}/{j}", 0o644)
open(f"{GRAFT}/.prepared", "w").close()
print(f">> static verification passed — graft ready: {GRAFT}")
PYEOF
[ -f "$REPO_DIR/snapshots/$GRAFT_NAME/.prepared" ] || {
  echo "!! graft script finished without a .prepared marker — static verification failed?"; exit 1; }

echo ">> graft checkpoint ready. Serve with:  MODE=hybrid-mtp scripts/serve.sh"
