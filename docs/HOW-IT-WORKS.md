# How it works

## The memory problem

Qwen3.8-Flash-Next is a sparse MoE with an unusual extra component: a **51B-parameter
n-gram embedding table** (the paper calls it PLE / "Engram"). The `RadixArk` NVFP4
checkpoint breaks down roughly as:

| Component | Format | Size |
|---|---|---|
| Routed experts (48 layers × 512 experts, 10 active) | NVFP4 | ~63 GiB |
| Attention / GDN / QSA / shared experts / gate / lm_head / MTP | bf16 | ~15 GiB |
| **N-gram (PLE) table** — 16 heads × 20M rows × 160 dims | FP8 e4m3 + 1 scale | **~48 GiB** |
| **Total** | | **~126 GiB** |

A DGX Spark has **128 GB unified memory**, of which ~10 GiB is OS/driver/Docker. So
126 GiB of weights leaves essentially nothing for the KV cache — you cannot serve.

vLLM ships an offload path (`VLLM_PLE_CPU_OFFLOAD=1`) that moves the table to pinned
**host** RAM. On a discrete-GPU server that frees VRAM. On a Spark, host and device
are the **same physical pool**, so it frees nothing. That is why, until now, the only
thing that ran Flash-Next on a Spark was a llama.cpp GGUF — which mmaps its weights by
default, but has no sparse-attention kernel and so has poor prefill and no MTP.

## The lever

The table is a **lookup**, not compute. Per token the model reads exactly
**16 rows × 160 bytes = 2.5 KB**, at hashed (random) addresses. Even a 20k-token
prefill is ~320k row reads ≈ 1.3 GB — under a second on NVMe — and natural language
and code hit a very concentrated set of n-grams, so the hot rows stay in the page
cache after the first pass.

So the table does not need to be resident. This repo `mmap`s the checkpoint's
`model-plefp8-*.safetensors` shards and gathers rows on demand. That is exactly what
llama.cpp does with its GGUF — we just bring it to the vLLM path, which keeps the real
QSA/GDN kernels and MTP.

Result: **~76 GiB resident** (78 GiB of non-table weights minus a little), leaving
~20–22 GiB for KV at `GPU_MEM=0.85` — a 720–790k-token pool, i.e. ~3× concurrency at
the native 262k or a single 500k request with YaRN.

## The patch (`src/vllm_ple_mmap.py`)

Enabled by `VLLM_PLE_MMAP=1`; a complete no-op otherwise. It patches exactly one
class, `Qwen3_8FlashNextNGramEmbedding`, in three small ways:

1. **`__init__`** — swap the 44/95 GiB `VocabParallelEmbedding` for a tiny
   placeholder. No large parameter is ever allocated. The placeholder's `forward(ids)`
   gathers rows from `np.memmap` views of the shards (dedup + sort for locality, a
   thread pool so page faults overlap), returns an fp8 tensor on the GPU.

2. **`load_weights`** — drop the 128 shard tensors on the floor (they're served from
   disk) and keep only the global FP8 `weight_scale`, stored as
   `_offload_weight_scale` — which the **unmodified** `Qwen3_8FlashNextPLELayer.
   _dequantize_embeddings` already knows how to consume. Then open the memmaps.

3. **`forward_impl`** — wrap the hashing+lookup in a custom op
   `vllm::ple_mmap_lookup`. This is the crucial bit for GB10 (below).

Everything else — the n-gram hashing, the short-conv, the dequant, the sparse
attention — is stock vLLM.

## Three GB10 bugs this works around

Bringing the official image up on a real Spark with real weights surfaced three
issues. All are handled by the patch + the flags in `scripts/serve.sh`:

1. **`Cannot copy between CPU and CUDA tensors during CUDA graph capture`.**
   The gather is CPU work plus a pageable host→device copy; that cannot live inside a
   captured CUDA graph. Fix: the lookup is a **custom op declared as a splitting op**,
   so vLLM runs it *between* graph segments. Use `-cc.cudagraph_mode=PIECEWISE` (never
   `FULL*`). `--enforce-eager` also avoids it but is slower — and note it does **not**
   fully suppress capture here (the mamba/short-conv path still captures), so PIECEWISE
   + the splitting op is the right answer.

2. **`KeyError` on the layer registry during capture.** The custom op looks the layer
   up by name; registering it inside `forward_impl` fails because torch.compile does
   not re-run that Python line on graph replay. Fix: register in `__init__`.

3. **Two stock-model issues on sm_121, unrelated to this patch but required to run:**
   - prefix caching crashed (`CUBLAS_STATUS_INTERNAL_ERROR` in a GDN `in_proj` GEMM, later
     `illegal memory access` in the Mamba state copy) on the cached-block path. The root
     cause turned out to be a vLLM block-size bug, fixed in this image — see
     [Prefix caching](#prefix-caching-the-root-cause-and-the-fix) below. The old advice
     (`--no-enable-prefix-caching`) is no longer needed.
   - full `torch.compile` off — an Inductor int64-indexing assert
     (`index out of bounds`) fires in the embedding gather codegen on sm_121. PIECEWISE
     capture with compile disabled on the splitting op sidesteps it.

## Long context: what works and what does not

Measured on the GX10 with the mmap patch, `GPU_MEM=0.85`, MTP=2 unless noted.

| Config | Result |
|---|---|
| 262144 native, MTP | KV pool ~720–790k tokens, ~3× concurrency at full length. Baseline prod. |
| **YaRN, CTX 500000, MTP** | **Works.** Pool ~724k tokens. Needle-in-a-haystack found at 276k and 414k tokens; decode 25–28 tok/s typical (36 on predictable text, ~94% draft acceptance there); no OOM. **This is the validated ceiling.** |
| YaRN, CTX 800000, MTP, `GPU_MEM=0.875` | Boots (pool 928k) and answers, but a 300k-token prefill got **SIGTERM from earlyoom** at 1.96% free memory: the prefill's activation peak plus the draft do not fit in ~5 GiB of headroom. |
| `--kv-cache-dtype fp8_e4m3` | Refused by the stock model (`QSA requires a BF16 main KV cache`); enabled by @Nanetnounou's patch in this image — see [fp8 KV cache](#fp8-kv-cache-on-the-qsa-path-opt-in). In bf16 a single 1M request needs 26.3 GiB of KV (28 KB/token, of which the attention K/V is the only part fp8 halves). |

Two YaRN-specific traps, both handled by `scripts/serve.sh`:

- YaRN is applied with Qwen's published `--hf-overrides` (rope `yarn`, factor 4,
  `original_max_position_embeddings` 262144) and needs `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`.
- **YaRN + MTP fails to boot** with `--mamba-block-size can only be set with
  --enable-prefix-caching`. Cause: dict `hf_overrides` are not propagated to the draft
  model (`SpeculativeConfig.compose_draft_hf_overrides` only forwards callables), so the
  draft keeps `max_model_len=262144` while sharing the `cache_config`, whose
  `mamba_block_size` was auto-set to the target's `max_model_len`. Fix: put
  `"max_model_len": <CTX>` inside `--speculative-config`, which overrides the draft's
  length (`_maybe_override_draft_max_model_len`).

## Correctness

`src/test_ple_mmap_cpu.py` builds synthetic FP8 shards (with the real safetensors
layout and non-trivial data offsets) and checks the mmap gather bit-for-bit against a
reference `table[ids]`, including dedup, multi-shard spans, the fp8 view path used by
the placeholder, and out-of-range → `IndexError`. It needs only numpy+torch (no GPU):

```bash
docker run --rm -v "$PWD/src:/t" -w /t --entrypoint python3 qwen38-flash-dgx test_ple_mmap_cpu.py
```

End-to-end, the served model is coherent ("The capital of France is Paris."), which is
the real test that the FP8 rows are being gathered and dequantized correctly — a wrong
gather turns the n-gram contribution to noise and the model degrades immediately.

## Performance notes

- **Prefill** ~2,400–2,660 tok/s (ctx 32k, single request). This is the axis that
  matters most versus llama.cpp (~540 tok/s), because Flash-Next's QSA prefill kernels
  only exist in vLLM/SGLang.
- **Decode** ~17 tok/s without speculation; with `MTP=2` **25–28 tok/s** on free-form
  prose (~63% draft acceptance) and up to ~36 tok/s on predictable text (~94%). The gather does one host↔device sync per decode step, which is pure
  latency at batch 1; MTP amortizes it. Removing that sync (staging ids through a
  pinned buffer, or a small resident hot-row cache) is the obvious next optimization.
- **First request into a cold region** of the table pays some NVMe I/O; it smooths out
  as the page cache warms. This makes single-shot prefill measurements cache-state
  dependent: the same prompt can run 2–3× slower on the first pass than once the rows it
  touches are resident, and the lower `GPU_MEM` is, the more RAM the page cache keeps for
  the 48 GiB table. Benchmark prefill on a second pass (or after `PREWARM=1`, which
  streams the whole table once at boot, ~10 s) and say which one you are quoting.

## Contributed GB10 fixes and the faster gather

Three changes from [@Saren-Arterius](https://github.com/Saren-Arterius)'s fork
([qwen3.8-Flash-DGX-AutoRound](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound)),
merged in and measured A/B on the GX10 (same flags, YaRN 500k, MTP=2, greedy, real
prompts — no `ignore_eos`, which pushes this model into a degenerate post-EOS regime
and makes decode numbers meaningless):

| | before | after |
|---|---|---|
| Decode, 400-token answers (median of 6) | 21.8 tok/s | **26.2 tok/s** (+20%; 25–29 on a warm server) |
| Prefill 8k (warm) | 2,289 tok/s | **~2,570 tok/s** (+12%) |
| Prefill 32k | 2,316 tok/s | 2,418 tok/s (+4%) |
| Needle at 92k tokens | found, 47.1 s | found, 44.7 s |

1. **FLA shared-memory gate.** sm_121 reports 99 KiB of shared memory per block; the
   flash-linear-attention gate (`ops/utils.py`, `DEFAULT = 102400`) asks for 100 KiB, so
   all 36 GDN layers silently ran the small-tile kernels. Lowering the constant to
   101376 lets the GB10 take the big-tile path. This is the same fix the
   Qwen3.5-122B Spark recipe carried as `patch_fla_shmem.py`.
2. **`chunk_delta_h` `num_warps=2` pin** — [fla#953](https://github.com/fla-org/flash-linear-attention/issues/953),
   a `tl.dot` race on Blackwell with `num_warps=4`. A correctness fix; no speed effect expected.
3. **PLE gather hot path** in `src/vllm_ple_mmap.py`: dedup row ids on CPU (`np.unique`),
   gather only unique rows, stage them through a persistent pinned buffer with an async
   H2D copy, expand on the GPU via the inverse index; decode-sized gathers (≤
   `VLLM_PLE_MMAP_FAST_ROWS`=512 unique rows) skip the thread pool. Also bf16/f16 tables,
   `VLLM_PLE_MMAP_DIR`, and a periodic `PLE mmap stats` line — which shows where the
   remaining decode cost is: ~6.5 ms of the ~9.5 ms per lookup is the disk gather
   itself (the page cache holds only part of the 48 GiB table at `GPU_MEM=0.80`).

## Independent reproduction and the native offload path

[@jschmied](https://github.com/jschmied) reproduced this recipe on a DGX Spark
([issue #1](https://github.com/blazux/qwen3.8-Flash-DGX/issues/1)). They also
ran vLLM's native `VLLM_PLE_CPU_OFFLOAD=1` path and documented what it needs on the
NVFP4 checkpoint (the `Fp8Config` gate in `_get_ple_embedding_quant_method`, and
`CAP_SYS_PTRACE` because `yama.ptrace_scope=1` blocks the sibling-process
`pidfd_getfd` used for the CUDA-IPC handoff), plus concurrency traces showing
aggregate throughput of ~267 tok/s at 48 streams with page-fault cost per token
*falling* with batch size. Full notes:
<https://github.com/jschmied/qwen38-flash-next-gb10>.

## Upstream references

- vLLM recipe: <https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next>
- vLLM PR (Flash-Next support): <https://github.com/vllm-project/vllm/pull/53896>
- NVFP4 checkpoint: <https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4>
- SGLang day-0 write-up (PLE offload mechanics): <https://www.lmsys.org/blog/2026-08-26-qwen-flash-next>

## Prefix caching: the root cause and the fix

With `--enable-prefix-caching` vLLM puts this hybrid model's Mamba-style layers (the 36
GDN layers and the PLE short-conv) in cache mode `align`: their recurrent state is
captured at every `mamba_block_size` boundary (1600 tokens here — vLLM also raises the
attention block to 1600 so both page sizes agree) and restored when a later request
hits a cached prefix.

What we saw on GB10, in order:

1. Stock image: `CUDA illegal memory access` on the first batch of cached requests.
2. With [vllm#50729](https://github.com/vllm-project/vllm/pull/50729) (a genuine fix
   for an overlapping-copy race) — same crash.
3. With @Saren-Arterius's bounds guard on top: no crash, but 40–80 "out-of-range block
   id" skips per ~100 requests, and greedy outputs that **changed on cache hits** (an
   answer became an empty completion, or the reverse).
4. Ruling things out one at a time (all with the deterministic top-k, so any difference
   is real): `--mamba-ssm-cache-dtype float32`, `MTP=0`, `--no-async-scheduling`,
   `--mamba-cache-mode all` — every variant produced the *same* wrong outputs. So it was
   structural, not numerical.
5. Instrumenting the two state-restore sites (checksums of the GDN layer-0 state and the
   PLE conv state, with the block index and `has_initial_state`) gave the answer in one
   probe. Cold request, 5150 tokens: state after 5136 tokens = `4.443284220e3`. Cache hit
   at 3200 tokens: `has_initial_state=True`, restored state checksum **`0.000`**, from a
   block slot that had never been written.

Steps 2–3 describe the state *before* the fix below. The bounds guard stayed in the image as a
safety net (a skipped copy plus a counter instead of a dead CUDA context); since the block_size
fix its counter has been 0 on every run, tournaments included, and a non-zero count would now
mean a new bug, not this one.

The bug: `EngineCore._initialize_kv_caches` (`vllm/v1/engine/core.py`) sets
`cache_config.block_size = min(group.block_size for every KV group)`. For this model
one of the groups is the QSA raw-key ring (`CircularBufferSpec`, `qsa_cache.py`), whose
block is its ring capacity `compress_ratio × cdiv(compress_ratio + num_spec, compress_ratio)`
= 8 tokens with MTP=2, 4 without. (This group type does not exist on upstream vLLM `main`,
where the same `min()` is harmless because every group is 1600.) `cache_config.mamba_block_size` stays 1600, but two consumers used
`cache_config.block_size` *as* the Mamba block size:

- `v1/worker/gpu/model_states/mamba_hybrid.py`, `add_request`: the running state slot
  is seeded from `(num_computed_tokens - 1) // block_size` → `3199 // 8 = 399` instead
  of `1`. Column 399 is past the end of the request's Mamba block-table row, the
  persistent table is zero-filled there, block id 0 is the null block — in range, so the
  guard never fires — and its all-zero page is copied in as the "restored" state.
- `v1/core/sched/scheduler.py`, `_mamba_block_aligned_split`: prefill chunks were
  aligned to 8-token boundaries instead of 1600, so states were almost never captured
  at a real boundary and cold requests rarely cached anything. This is directly visible
  in the traces: a 5,150-token cold prompt stopped its chunk at 5,136 with MTP=2
  (`5150 - 5150 % 8 - 8`, the Eagle back-off) and at 5,148 with MTP=0 (`5150 - 5150 % 4`)
  — exactly the ring block sizes, never 4,800.

The fix (`src/patch_mamba_block_size.py`) is two lines: use
`cache_config.mamba_block_size` in the first and the scheduler's own `self.block_size`
(the LCM of all group block sizes, 1600) in the second. Validation on the same probe:
hit restores `4.149867426e3` = exactly the state the cold run wrote at 3200; first-token
top-5 logprobs identical to the 4th decimal between cold and two hits on 4k/8k/15k/31k
prompts; 32/32 greedy completions identical cold vs hit; guard counter 0 through 100
concurrent multi-turn requests; tournament 45/51 with caching vs 44–45 without.

Note that in single-process mode (`UniProcExecutor`, the default on one GPU) the
scheduler and the worker share the same `vllm_config` object, so both halves of the bug
apply; with the multiprocess executor only the scheduler half would. We will upstream
this.

## Exact top-k for the sparse attention

QSA scores every query against compressed key blocks and keeps the top `k` (512–2048
tokens' worth). vLLM does that with `torch.ops._C.persistent_topk`, a histogram-based
approximate select. On GB10 the faster `cooperative_topk` is disabled
(`not is_device_capability_family(120)`), so `persistent_topk` runs for prefill *and*
decode.

@k3dani ([issue #3](https://github.com/blazux/qwen3.8-Flash-DGX/issues/3),
[vllm#51782](https://github.com/vllm-project/vllm/issues/51782)) showed that the kernel
drops legitimate candidates when more than 16384 logits share a coarse histogram bin —
which trained indexer logits do — and that its output differs between launches. We
confirmed both: with the stock kernel, 3 identical greedy runs of the same prompt gave
3 different outputs on 2 of 4 prompts (2.6k–128k tokens), and first-token top-5 sets
that didn't even overlap.

`src/patch_qsa_exact_topk.py` adds `VLLM_QSA_EXACT_TOPK=1`: mask the columns the
scoring kernel never wrote (≥ `visible_blocks[row]`, they are `torch.empty`) to `-inf`
in place, then `torch.topk` over the row. Results: 4/4 prompts stable, first-token
logprobs identical to the 4th decimal across runs, tournament score unchanged (44/51 →
44/51 on NVFP4; 45/51 with prefix caching). Cost on the GX10: decode unchanged, prefill
−8% at 8k, −20–40% at 32k+ (the top-k runs over the full visible width per chunk).
Masking the uninitialized columns and keeping the stock kernel (`VLLM_QSA_EXACT_TOPK=fill`)
does **not** restore determinism, so the kernel itself is the problem, not the garbage.

### The kernel-side fix (patch 8, `VLLM_QSA_DET_TOPK=1`)

vLLM's `persistent_topk` hands out output slots with `atomicAdd` (thread-arrival order) and
takes exact-key ties at the last radix round first-come; when more elements share the
threshold key than fit its candidate buffers, the selected *set* changes too. Since the
sparse attention sums the selected keys in output order, either forks the hidden state.
[@jschmied](https://github.com/jschmied)'s rewrite ([vllm#55122](https://github.com/vllm-project/vllm/pull/55122))
makes every single-CTA row go through a radix select that rescans the row per key byte (no
candidate buffers, exact pivot) followed by an index-ordered block scan, and gives the
multi-CTA path a deterministic emission (per-CTA counts + prefix over CTAs, ties ranked by
index). Micro-benchmark cost is 1.3–4× per call, which at model level is noise.

We build it as a standalone extension (`_C_det.so`) with the image's `nvcc` at `docker build`
time, from his repo at a pinned commit, and route `qsa_select_paged_tokens` to
`torch.ops._C_det.persistent_topk` when `VLLM_QSA_DET_TOPK=1`. Measured on the GX10 (hybrid,
MTP=2, prefix caching, same box, same bench script and prompts; the exact and stock columns are the earlier runs from the sections above):

| | stock kernel | exact `torch.topk` | **deterministic kernel** |
|---|---|---|---|
| Deterministic (4 prompts × 3, first-token logprobs) | no | yes (0.000) | **yes (0.000)** |
| Decode | 32.4 tok/s | 30.8 | **32.5** |
| Prefill 8k | ~1,650 | 1,476 | **2,436** |
| Prefill 32k | 2,105 | 1,794 | **2,904** |
| Needle 92k | 45 s | 69 s | **48 s** |

His standalone `test_det.py` (177 cases: bit-identical across calls, equal to an exact
reference, adversarial tie populations around every buffer size the original kernels used)
passes 177/177 on the GX10, and the stock op fails to reproduce itself on the same inputs.

## Hybrid mode: NVFP4 experts + blockwise-fp8 side layers

The RadixArk checkpoint quantizes only the routed experts (ModelOpt NVFP4) and leaves
the dense side layers — GDN `in_proj`/`out_proj`, QSA `q/k/v/o_proj`, shared experts,
~15 GiB — in bf16. Every decoded token reads all of them, so they set the decode
bandwidth floor. `scripts/prepare-hybrid.sh` rewrites those 300 tensors as blockwise
fp8-e4m3 with a per-128×128 fp32 `weight_scale_inv` (the DeepSeek-V3 layout that vLLM's
`Fp8LinearMethod` already loads), in a sibling snapshot directory made of relative
symlinks — only the 4 rewritten shards are real files.

Serving them needs a small dispatch shim (`src/vllm_fp8_hybrid_modelopt.py`,
`VLLM_FP8_HYBRID=1`), a port of @Saren-Arterius's int4+fp8 dispatch to the ModelOpt
config: it scans the safetensors metadata for `F8_E4M3` weights that have a
`weight_scale_inv` sibling and, for those (fused) modules, returns vLLM's blockwise-fp8
linear method instead of the bf16 path, leaving the NVFP4 MoE path alone. One
model-specific wrinkle: `qsa.py` builds the QSA `qkv_proj` with
`without_modelopt_fp4(quant_config)` — i.e. with **no** quant config at all — so the
shim is never consulted there and loading dies on `'QKVParallelLinear' object has no
attribute 'data'`. The Dockerfile redirects that call to a proxy config that dispatches
to fp8 when the checkpoint has fp8 q/k/v for that layer and to bf16 otherwise.

Measured on the GX10 (YaRN 500k, MTP=2, exact top-k, prefix caching, greedy, real
prompts):

| | NVFP4 | hybrid | Intel int4 + fp8 (fork) |
|---|---|---|---|
| Tournament (17 agentic scenarios × 3, ok/51) | 45 | 45 | 44 |
| Deterministic at T=0 | yes | yes | **no** (also with Marlin atomic adds off) |
| Decode, 400-token answers (median of 6) | 25.7 tok/s | **30.8 tok/s** | 34.3 tok/s |
| Prefill 32k | ~1,900 tok/s | ~1,800 tok/s | ~1,900 tok/s |
| Needle at 92k | 64 s | 69 s | 69 s |
| TTFT, 2nd+ turn, 8 concurrent 20k-token conversations | 5.9 s | **4.1 s** | 8.7 s |
| KV cache (`GPU_MEM=0.80`) | 582k tokens | **633k** | 762k |
| Resident weights | ~84 GiB | ~77 GiB | ~70 GiB |

The six failed tournament passes are the same two scenarios (`b6_reconcile`,
`c5_inventory_reconcile`) in every configuration we have ever run, including the
unquantized-side-layer NVFP4 without any of our changes — they are the model, not the
quantization. Before we raised the harness's turn budget the hybrid lost a few extra
passes by spending one more tool call (it checks balances before acting), which is a
behavioural nuance rather than a precision loss; every tool call it made was correct.

The Intel AutoRound variant is fastest at raw decode but could not be made
deterministic and has the worst cached-TTFT (its prefill is the slowest), so we did not
adopt it; the numbers are here for completeness.


### The blockwise-fp8 GEMM's `M % 4` slow path (patch 9, opt-in)

Found by [@jschmied](https://github.com/jschmied) (issue #3): the preview image's sm_12x
blockwise-fp8 cutlass dispatch takes `swap_ab = (M <= 64) || (M % 4 != 0)`, and that path is
slow. Kernel micro-bench on the GX10 (K=4096, N=8192, our image):

| rows M | aligned | M % 4 ≠ 0 | padded to 4 |
|---|---|---|---|
| 501 / 1,201 / 1,601 | 0.22 / 0.47 / 0.63 ms | 0.37 / 0.79 / 1.09 ms (×1.7) | = aligned |
| 2,049 / 2,401 / 3,001 | 0.73 / 0.87 / 1.07 ms | 8.2 / 9.6 / 12.0 ms (**×10–11**) | = aligned |
| 8,001 / 32,001 | 9.8 / 40 ms | 37 / 147 ms (×3.7) | = aligned |

Upstream removed the clause in C++ (vllm#52775, 2026-08-19); his `fp8_m4pad_patch.py` pads M
to a multiple of 4 (zero rows, unit scale rows, output sliced) inside an opaque custom op so
`torch.compile` cannot freeze the branch at the profiling shape. Why it does not show on our
default config: with `--enable-prefix-caching` the scheduler's Mamba align mode
(`_mamba_block_aligned_split`) clips every prefill chunk to the 1,600-token block boundary, so
the large chunks always reach the GEMM with M % 4 == 0 and only the last chunk of a prompt has an
arbitrary row count. Same-session A/B on the hybrid (MTP=2, prefix caching): 8k 3.33 → 3.24 s,
32k 11.63 → 10.97 s, needle 48.0 → 47.2 s, salted prefills within noise; prompts built to leave a
misaligned last chunk above 2,048 rows (8,801 / 8,803 tokens) cost the same as aligned ones
(3.60–3.69 s). Hence `PAD_M4=0` by default. With prefix caching off the chunks are not aligned and
his −40% TTFT at 8k applies — that is the case the option is for. NVFP4 mode never calls this GEMM.

## fp8 KV cache on the QSA path (opt-in)

vLLM already had the plumbing (`kv_quant_mode`, `_k_scale`/`_v_scale`, allocation and
writes) — what was missing for this model was the read side: the QSA Triton kernels
loaded the cache as bf16 and five guards rejected anything else.
[@Nanetnounou](https://github.com/Nanetnounou)'s `src/patch_qsa_fp8_kv.py`
([issue #6](https://github.com/blazux/qwen3.8-Flash-DGX/issues/6)) wires vLLM's own
`_cast_kv_tile` into the decode and MQA (block-selector) kernels, reinterprets the
`uint8` allocation as `float8_e4m3fn` — for the main KV *and* the indexer's raw-key ring,
which otherwise picks arbitrary blocks — halves `block_n` under quantization to stay
under the GB10's 101,376-byte shared memory, and neutralises the dtype guard inherited
from `FlashAttentionImpl` (whose kernels QSA never calls). Inert with `--kv-cache-dtype auto`.

Measured (hybrid, MTP=2, exact top-k, prefix caching, `GPU_MEM=0.80`, `CTX=1000000` YaRN):
KV pool 1,219,879 tokens (bf16 at 500k: 634k) and a 1M single request boots with 1.22×
concurrency; decode 27.9 vs 30.8 tok/s, prefill 32k 1,254 vs 1,794 tok/s, needle 92k
89 s vs 69 s; tournament 45/51 with the usual b6/c5 failures, but `b3_itinerary` falls
from 6/6 to 2/6 passes (and its one success took 193 s instead of ~50 s; the failures ran
to the 412 s cap). vLLM raises the attention block to 3,184 tokens in this mode to keep
attention and Mamba pages equal. Note for anyone sizing this: the fp8 saving applies to
the attention K/V only — the GDN/PLE recurrent states, the QSA compressed keys and the
raw-key ring stay as they are — which is why bf16 at 1M asked for 26.3 GiB and fp8 gets
1M into 17 GiB.

We keep bf16 in production: the model's speed is our scarcest resource and the b3
regression is the kind of long-reasoning case we care about. The option is there for
workloads that need the context.
