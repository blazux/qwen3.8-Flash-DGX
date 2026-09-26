#!/usr/bin/env python3
"""GPU unit test for patch 7 (fp8_e4m3 KV read path on the QSA path).

Proves the read side of the patched split-K kernel dequantizes an fp8_e4m3
cache with the layer's real scales to exactly the same BF16 tiles the bf16
reference path loads, so the attention output is bit-identical (max abs diff
0.0), and that the fp8 cache spec allocates exactly half the bf16 page size.

    docker run --rm --gpus all -v "$PWD:/t" -w /t \
        --entrypoint python3 qwen38-flash-dgx:v0.30fp8 -m pytest tests/test_fp8_kv_read.py -v

Skips cleanly without a GPU or against an image without patch 7. No model
load: a few small tensors only.
"""
import inspect

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

vllm_qsa = pytest.importorskip(
    "vllm.models.qwen4_exp.nvidia.ops.qsa",
    reason="module qwen4_exp not found (vLLM < 0.29 layout)",
)
if not torch.cuda.is_available():
    pytest.skip("no GPU", allow_module_level=True)

qsa_sparse_paged_attention = vllm_qsa.qsa_sparse_paged_attention

# Patch 7 adds k_scale/v_scale to the public wrapper; without them the image
# carries no fp8 read path (src/patch_qsa_fp8_kv_v030.py not applied).
_params = inspect.signature(qsa_sparse_paged_attention).parameters
if "k_scale" not in _params:
    pytest.skip(
        "image is not patched with patch 7 (qsa wrapper has no k_scale)",
        allow_module_level=True,
    )

HEAD_DIM = 128
PAGE_SIZE = 16
NUM_PAGES = 4
NUM_ROWS = 2
NUM_Q_HEADS = 4   # group size 4 over the single KV head
TOPK = 8          # one tile in both modes -> same accumulation order in bf16/fp8

# The layer's real per-tensor scales (not 1.0): the v0.30 patch reads with the
# scales the write side quantized with, so the test uses two distinct ones.
K_SCALE = 0.37
V_SCALE = 1.25


def _scenario(seed: int):
    """Build one small QSA decode step: q, bf16 K/V, indices, page table."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(
        (NUM_ROWS, NUM_Q_HEADS, HEAD_DIM),
        generator=gen, device="cuda", dtype=torch.float32,
    ).to(torch.bfloat16)
    # float "model" tensors, quantized to fp8 with the scales below
    k_f = torch.randn(
        (NUM_PAGES, PAGE_SIZE, 1, HEAD_DIM),
        generator=gen, device="cuda", dtype=torch.float32,
    )
    v_f = torch.randn(
        (NUM_PAGES, PAGE_SIZE, 1, HEAD_DIM),
        generator=gen, device="cuda", dtype=torch.float32,
    )
    # packed selection buffer: [rows, TOPK + 1], trailing column = valid count
    logical_indices = torch.randint(
        0, NUM_PAGES * PAGE_SIZE, (NUM_ROWS, TOPK),
        generator=gen, device="cuda", dtype=torch.int32,
    )
    counts = torch.full((NUM_ROWS, 1), TOPK, dtype=torch.int32, device="cuda")
    logical_indices = torch.cat([logical_indices, counts], dim=1)
    block_table = torch.arange(
        NUM_PAGES, dtype=torch.int32, device="cuda"
    ).expand(NUM_ROWS, NUM_PAGES).contiguous()
    token_to_req = torch.zeros(NUM_ROWS, dtype=torch.int32, device="cuda")
    gate = torch.zeros(
        (NUM_ROWS, NUM_Q_HEADS, HEAD_DIM), device="cuda", dtype=torch.bfloat16
    )
    return q, k_f, v_f, logical_indices, block_table, token_to_req, gate


def _run(q, k_cache, v_cache, idx, bt, t2r, gate, k_scale=None, v_scale=None):
    return qsa_sparse_paged_attention(
        q, k_cache, v_cache, idx, bt, t2r,
        use_prefill_config=False,
        k_scale=k_scale, v_scale=v_scale,
        output_gate=gate,
    )


def test_fp8_read_matches_bf16_reference_bitwise():
    """fp8 cache + real scales == the bf16 cache its tiles dequantize to."""
    q, k_f, v_f, idx, bt, t2r, gate = _scenario(seed=0)
    k_s = torch.tensor(K_SCALE, dtype=torch.float32, device="cuda")
    v_s = torch.tensor(V_SCALE, dtype=torch.float32, device="cuda")

    # write side, as do_kv_cache_update does it: value/scale -> fp8 storage
    k_fp8 = (k_f / k_s).to(torch.float8_e4m3fn)
    v_fp8 = (v_f / v_s).to(torch.float8_e4m3fn)
    # read side reference: exactly what _cast_kv_tile(mode=1) materializes
    k_ref = (k_fp8.float() * k_s).to(torch.bfloat16)
    v_ref = (v_fp8.float() * v_s).to(torch.bfloat16)

    out_fp8 = _run(q, k_fp8, v_fp8, idx, bt, t2r, gate, k_scale=k_s, v_scale=v_s)
    out_ref = _run(q, k_ref, v_ref, idx, bt, t2r, gate)

    diff = (out_fp8.float() - out_ref.float()).abs().max().item()
    assert diff == 0.0, f"fp8 read path differs from bf16 reference: {diff}"


def test_fp8_read_uses_the_passed_scales():
    """A wrong (1.0) scale must NOT reproduce the reference -> scales are live."""
    q, k_f, v_f, idx, bt, t2r, gate = _scenario(seed=1)
    k_s = torch.tensor(K_SCALE, dtype=torch.float32, device="cuda")
    v_s = torch.tensor(V_SCALE, dtype=torch.float32, device="cuda")
    k_fp8 = (k_f / k_s).to(torch.float8_e4m3fn)
    v_fp8 = (v_f / v_s).to(torch.float8_e4m3fn)

    one = torch.ones((), dtype=torch.float32, device="cuda")
    out_real = _run(q, k_fp8, v_fp8, idx, bt, t2r, gate, k_scale=k_s, v_scale=v_s)
    out_wrong = _run(q, k_fp8, v_fp8, idx, bt, t2r, gate, k_scale=one, v_scale=one)
    assert not torch.equal(out_real, out_wrong), (
        "the kernel ignored k_scale/v_scale (read still fixed at 1.0?)"
    )


def test_fp8_page_size_is_half_of_bf16():
    """The cache spec the patch relies on allocates exactly half the bytes."""
    pytest.importorskip("vllm.models.qwen4_exp.nvidia.model")
    from vllm.v1.kv_cache_interface import FullAttentionSpec

    fp8 = FullAttentionSpec(
        block_size=PAGE_SIZE, num_kv_heads=1, head_size=HEAD_DIM,
        head_size_v=HEAD_DIM, dtype=torch.uint8, kv_quant_mode=1,
    )
    bf16 = FullAttentionSpec(
        block_size=PAGE_SIZE, num_kv_heads=1, head_size=HEAD_DIM,
        head_size_v=HEAD_DIM, dtype=torch.bfloat16, kv_quant_mode=0,
    )
    assert fp8.page_size_bytes * 2 == bf16.page_size_bytes
