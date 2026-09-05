"""CPU unit test for vllm_ple_mmap: synthetic FP8 shards -> gather == reference.

Run inside the vLLM image (needs numpy + torch, no GPU):
  docker run --rm -v $PWD:/t -w /t --entrypoint python3 vllm/vllm-openai:qwen38-flash-next test_ple_mmap_cpu.py
"""
import json
import os
import struct
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vllm_ple_mmap as m  # noqa: E402

ROWS, COLS, PARTS = 100_000, 160, 8
shard_size = -(-ROWS // PARTS)
rng = np.random.default_rng(0)
table = rng.integers(0, 256, size=(ROWS, COLS), dtype=np.uint8)

tmp = tempfile.mkdtemp()
# write shards into 2 safetensors files (4 shards each) with a dummy tensor first,
# so data offsets are non-trivial
file_of = {}
for fi in range(2):
    tensors = {"dummy.weight": np.arange(37, dtype=np.float32).tobytes()}
    header = {"dummy.weight": {"dtype": "F32", "shape": [37], "data_offsets": [0, 37 * 4]}}
    off = 37 * 4
    for si in range(fi * 4, fi * 4 + 4):
        rows = table[si * shard_size : (si + 1) * shard_size]
        name = f"model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{si}.weight"
        header[name] = {"dtype": "F8_E4M3", "shape": list(rows.shape), "data_offsets": [off, off + rows.nbytes]}
        tensors[name] = rows.tobytes()
        off += rows.nbytes
        file_of[name] = f"model-plefp8-0000{fi}.safetensors"
    if fi == 1:
        name = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale"
        header[name] = {"dtype": "F32", "shape": [], "data_offsets": [off, off + 4]}
        tensors[name] = struct.pack("<f", 0.03125)
        off += 4
        file_of[name] = f"model-plefp8-0000{fi}.safetensors"
    hb = json.dumps(header).encode()
    with open(os.path.join(tmp, f"model-plefp8-0000{fi}.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        for name in header:
            f.write(tensors[name])
with open(os.path.join(tmp, "model.safetensors.index.json"), "w") as f:
    json.dump({"weight_map": file_of}, f)

shards, dtype_str, scale_entry, cols = m._find_shards(tmp, 1)
assert dtype_str == "F8_E4M3" and cols == COLS, (dtype_str, cols)
assert len(shards) == PARTS, len(shards)
assert abs(float(m._read_scale(scale_entry)) - 0.03125) < 1e-9
for idx, (_p, _o, rows) in shards.items():
    assert rows == max(0, min(shard_size, ROWS - idx * shard_size)), (idx, rows)

t = m.MmapPleTable(shards, shard_size, cols, torch.float8_e4m3fn, workers=8, chunk=512)
assert t.rows_total == ROWS

# the last shard is partial: ids in [rows_total, parts*shard_size) are out of range and
# must raise IndexError, not read past the table (both the inline and the pooled path)
for n, kbad in ((8, 1), (5_000, 3)):
    bad = np.random.default_rng(7).integers(0, ROWS, size=n, dtype=np.int64)
    bad[:kbad] = ROWS + 7          # across the shard boundary: pooled multi-task path
    try:
        t.gather(bad)
        raise AssertionError(f"gather accepted {n} ids with {kbad} out-of-range")
    except IndexError:
        pass

for n in (1, 16, 5000, 131_072):
    ids = rng.integers(0, ROWS, size=n, dtype=np.int64)
    ids[: n // 3] = ids[0]  # lots of duplicates, like real n-grams
    t0 = time.perf_counter()
    got = t.gather(ids)
    dt = time.perf_counter() - t0
    ref = table[ids]
    assert got.shape == (n, COLS) and got.dtype == np.uint8
    assert np.array_equal(got, ref), f"mismatch for n={n}"
    print(f"gather n={n:>7}: OK in {dt*1e3:7.2f} ms")

# torch view path used by the placeholder
emb = m._MmapNgramEmbedding(ROWS, COLS)
emb.table = t
ids_t = torch.from_numpy(rng.integers(0, ROWS, size=(300, 16), dtype=np.int64))
out = emb(ids_t)
assert out.shape == (300, 16, COLS) and out.dtype == torch.float8_e4m3fn
assert np.array_equal(out.view(torch.uint8).numpy().reshape(-1, COLS), table[ids_t.numpy().reshape(-1)])
print("placeholder forward: OK (fp8 view, shape", tuple(out.shape), ")")

# zeros path (no table)
emb2 = m._MmapNgramEmbedding(ROWS, COLS)
z = emb2(ids_t)
assert z.shape == (300, 16, COLS) and float(z.abs().sum()) == 0.0
print("zeros path: OK")

# out-of-range must raise, not corrupt
try:
    t.gather(np.array([ROWS + 5], dtype=np.int64))
    raise SystemExit("expected IndexError")
except IndexError:
    print("out-of-range: raises IndexError OK")

t.prewarm()
print("prewarm: OK")
print("ALL OK")
