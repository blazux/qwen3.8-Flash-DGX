#!/usr/bin/env python3
"""Faster MoE weight loading on GB10 (VLLM_LOAD_CLONE=1, default on; 0 disables).

FusedMoE's _load_w13 / _load_w2 copy every per-expert tensor to the GPU with
expert_data.copy_(loaded_weight), straight from the safetensors mmap view. On GB10 that pageable H2D
copy costs ~1.7 ms per 800 KiB tensor when the source is a file-backed page, ~0.23 ms from anonymous
memory (cached or not, NFS or NVMe), and the NVFP4 checkpoints have ~149k such weight + block-scale
tensors. Cloning the view into ordinary memory first gives byte-identical weights at a transient
cost of one tensor; main-model weight loading 541 -> 150 s on a DGX Spark. Measured with tools/bench_moe_load.py; see docs/HOW-IT-WORKS.md (patch 14)."""
import ast
import sys

RE = f"{sys.argv[1]}/vllm/model_executor/layers/fused_moe/routed_experts.py"
s = open(RE).read()

# v0.31 adds a bounded copy for strided TP slices in _load_w2. Preserve that
# branch and hook only each method's final ordinary copy, as on v0.30.
lines = s.splitlines(keepends=True)
tree = ast.parse(s)
sites = []
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name in ("_load_w13", "_load_w2"):
        final = node.body[-1]
        line = lines[final.lineno - 1]
        assert line.strip() == "expert_data.copy_(loaded_weight)", f"{node.name}: final copy moved"
        sites.append(final.lineno - 1)
assert len(sites) == 2, f"expected _load_w13 and _load_w2, found {len(sites)}"
for site in sites:
    lines[site] = lines[site].replace(
        "expert_data.copy_(loaded_weight)",
        "expert_data.copy_(_qwen38_h2d_src(expert_data, loaded_weight))")
s = "".join(lines)

s += '''

# --- qwen38-flash-dgx: clone mmap-backed weights before the H2D copy (VLLM_LOAD_CLONE=1) ---
import os as _qwen38_os

_QWEN38_LOAD_CLONE = _qwen38_os.environ.get("VLLM_LOAD_CLONE", "1") != "0"


def _qwen38_h2d_src(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    if (
        _QWEN38_LOAD_CLONE
        and dst.device.type != "cpu"
        and src.device.type == "cpu"
        and not src.is_pinned()
    ):
        return src.clone()
    return src
'''

ast.parse(s)
open(RE, "w").write(s)
print("routed_experts.py: MoE load clone applied OK")
