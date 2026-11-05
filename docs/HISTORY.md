# History

The [archived full README](https://github.com/blazux/qwen3.8-Flash-DGX/blob/383e1f975b4ce9eed3f2e5d6d923d37d54c5c0e1/README.md) records earlier release notes and benchmark tables.

What the README said about each release of this recipe, kept as it was written. Until 2026-09-28 the
repo supported three base images: the Qwen3.8-Flash-Next preview image (`Dockerfile`), vLLM v0.29.0
(`Dockerfile.v0.29`) and vLLM v0.30.0 (`Dockerfile.v0.30`). Since then vLLM v0.30.0 is the only base.
The files these entries mention (`Dockerfile.v0.29`, the preview `Dockerfile`, the `v0.29` and `v0.30`
profiles, `PAD_M4`, patches 3, 9 and 11) are at the git tag
[`multi-base-final`](https://github.com/blazux/qwen3.8-Flash-DGX/tree/multi-base-final).

## Update 2026-09-25 — vLLM v0.30.0 base, 3-minute boots

- **`Dockerfile.v0.30`**: the recipe on the vLLM v0.30.0 release (profile `v0.30`). Same weights, same
  defaults. On our GX10 against the v0.29 setup that ran production: tournament 87.8% (3 runs) vs 88.8%
  (5 runs), no detectable difference; cold prefill 1.5–2× faster, the 413k-token needle in 121 s instead
  of 148 s, same decode. It is our production image now. Details in
  [vLLM v0.30.0 as the base image](#vllm-v0300-as-the-base-image-dockerfilev030); thanks to
  [@ChengYen-Tang](https://github.com/ChengYen-Tang) for the nudge in [#14](https://github.com/blazux/qwen3.8-Flash-DGX/issues/14).
- **Patch 14, faster weight loading** ([#33](https://github.com/blazux/qwen3.8-Flash-DGX/pull/33) by
  [@Willian-Zhang](https://github.com/Willian-Zhang)): on all three bases. Main-model loading went from
  452 s to 242 s on our box (v0.29 image) and 231 s on v0.30, with byte-identical first-token
  log-probs.
- **Patches 15–18, the rest of weight loading** ([#34](https://github.com/blazux/qwen3.8-Flash-DGX/pull/34), also by
  @Willian-Zhang): `pread` for small tensors, an index for expert names, chunked embedding copies,
  and an MTP drafter that skips non-MTP tensors before reading them. On our box, v0.30 image: main
  load 245 → 114 s, drafter 40 → 1.3 s, **boot 6 min 30 → 3 min 35**; first-token log-probs 5/5
  identical to the image without them, same decode speed. Each has its own switch
  (`VLLM_LOAD_PREAD`, `VLLM_MOE_NAME_INDEX`, `VLLM_LOAD_EMBED_CHUNK`, `VLLM_MTP_NAME_PREFILTER`).
- **The KV pool reads smaller, and that is the real number.** On GB10 vLLM sizes the pool from
  free *host* memory. A slow load pushed some of vLLM's own memory to swap before the profile, and
  that memory was counted as free: our 714k-token boot had 1.6 GiB of vLLM swapped out. With the
  fast load nothing is swapped and the same recipe gets ~520k (v0.30) to ~630k (preview) tokens.
  For a pool that does not move between boots, set `KV_CACHE_MEM`.
- **fp8 KV cache on v0.30** (patch 7, the preview's by [@Nanetnounou](https://github.com/Nanetnounou), re-targeted):
  `./flash serve v0.30 KV_DTYPE=fp8_e4m3 CTX=1000000`. 1,039k-token pool at `GPU_MEM=0.80`; needles found
  at 196k, 413k, 635k and **931k tokens**; tournament 87.1% (one run, bf16 87.3–88.1%); decode −4%,
  prefill −3 to −17%. It now reads the cache with the layer's real scales (the preview read with 1.0,
  correct only because the writes used 1.0 too). Prefix caching works at 3,184-token blocks instead of
  1,600: the cache aligns attention blocks to the Mamba state page, and fp8 halves the bytes per token.
  This was the last patch missing on v0.30.
- **Determinism is sequential** ([#32](https://github.com/blazux/qwen3.8-Flash-DGX/issues/32)): the
  same request repeated one at a time is byte-identical; concurrent requests are not batch-invariant,
  a vLLM limit for GDN models. Scope added to [Deterministic top-k](https://github.com/blazux/qwen3.8-Flash-DGX/blob/383e1f975b4ce9eed3f2e5d6d923d37d54c5c0e1/README.md#deterministic-top-k-det_topk1-default).

## Update 2026-09-14 — NVIDIA's checkpoint is the default

- **`MODEL` now defaults to `nvidia/Qwen3.8-Flash-Next-NVFP4`** in `flash`, `serve.sh`, `download-weights.sh`
  and `prepare-hybrid.sh`. RadixArk's checkpoint, the default until now, stays fully supported:
  `MODEL=RadixArk/Qwen3.8-Flash-Next-NVFP4` on any of them (or `./flash setup default MODEL=…`), nothing
  else changes. Already running RadixArk? Nothing breaks: the variable was always honoured, and the
  recipe, patches and profiles are identical for both.
- **Why**: a head-to-head on 2026-09-13/14 — same image (`main`), same recipe, only the checkpoint
  changed, 5 full passes of a 55-scenario agentic tournament per side plus speed, memory, long-context
  and determinism probes on the same boot. NVIDIA 88.8% ± 1.0 vs RadixArk 86.1% ± 1.9; behind on no
  scenario beyond one-run noise; reasoning-runaway rate identical; needle 6/6 to 413k and deterministic
  on both; KV pool +15% on that boot (679k vs 589k tokens, +22% on another); single-stream decode −7 to
  −8% (34.5 vs 37.1 tok/s), equal under load. The default rule of this repo is quality first: at parity
  or better, the checkpoint with more KV room and the vendor's own export wins. Numbers, protocol and
  the honest caveats: [Checkpoints](https://github.com/blazux/qwen3.8-Flash-DGX/blob/383e1f975b4ce9eed3f2e5d6d923d37d54c5c0e1/README.md#checkpoints-nvidias-nvfp4-default-and-radixarks).
- **What it costs you**: ~3 tok/s of single-stream decode against RadixArk, and a 124 GiB download in
  24 files, one of them 50 GiB, that only comes through Xet (the default of `download-weights.sh`
  since PR #19). On the v0.29 image the fp8 MTP drafter loads through the vllm#55513 backport; on the
  preview image through the stopgap shim — both validated.
- **`MODE=hybrid-mtp` (the NVFP4 draft-experts graft, profile `context`) is RadixArk-only** and now
  says so: NVIDIA's drafter is already fp8, which is exactly where its KV advantage comes from. The
  `context` profile pins `MODEL=RadixArk/…` for that reason.
- **`reasoning_effort: high` no longer returns 400** (`EFFORT_ALIAS=1`, default). The chat template all
  these checkpoints share — NVIDIA's, RadixArk's and the abliterated copies — accepts only `xhigh` (its
  default), `medium` and `low`, and raises on anything else. vLLM passes the request's effort straight
  through (on `/v1/messages`, `output_config.effort`), and Claude Code sends `high` by default, so every
  such request failed with `Unexpected reasoning effort high`. `serve.sh` now serves a copy of the
  checkpoint's own template with its one effort-resolving line rewritten: `high` and `max` → `xhigh`,
  `minimal` → `low`. Every other value renders byte-identically, and a template without that check or
  that line is left alone. The copy is bind-mounted into the container from the first writable of the HF
  cache, `~/.cache/qwen38-flash-dgx/` and the checkout: an HF cache first created by a manual `docker run`
  belongs to root, and the alias must not silently degrade to the old 400 there (`./flash doctor` now
  warns about such a cache).
- **`FAST_ROWS=0`: every PLE gather goes to the thread pool** (new default). The mmap patch gathered
  decode-sized batches (≤ 512 unique rows) inline on one thread, so every row the page cache had dropped
  was its own serial page fault — and on a Spark the 48 GiB table never fits in cache. Measured within
  one boot (drowzeys' NVIDIA-based checkpoint, v0.29, hybrid, MTP=3, `vm.swappiness=10`), the path
  switched at runtime in ABBA-BAAB phases with fresh prompts each phase: **1 stream 35.1 → 37.9 tok/s
  (+8%)**, gather 11.4 → 5.0 ms; **4 streams 68.0 → 79.4 tok/s aggregate (+17%)**, gather 36.4 → 11.5 ms,
  every pool phase ahead of every inline phase. The rows gathered are the same either way, so outputs do
  not change. `FAST_ROWS=512` restores the old path, which is ~0.8 ms faster per gather only when every
  row is already cached.

## Update 2026-09-13 — what changed

Newest first. If you cloned this before, this is the short version; details in the linked sections.

**2026-09-13** — NVIDIA's own NVFP4 checkpoint runs on the recipe, and three contributed options:

- **`nvidia/Qwen3.8-Flash-Next-NVFP4` is supported** (issue #17, [@PathosEthosLogos](https://github.com/PathosEthosLogos)).
  Same recipe, `MODEL=nvidia/Qwen3.8-Flash-Next-NVFP4`; the hybrid layout works on it unchanged. It needed
  patch 11 (its MTP drafter's experts are blockwise fp8 under a ModelOpt *mixed-precision* config that
  vLLM 0.29 does not know how to load) and the hybrid shim extended to that config class. Patch 11 started
  as a stopgap shim; on the v0.29 image it is now a backport of vLLM's own fix (vllm#55513, by
  @techfury90), and only the preview image keeps the shim. Measured against
  RadixArk at equal recipe: **quality at parity, needle 6/6 on both up to 413k, decode 34.0 vs 36.4 tok/s,
  KV pool +22–28% (721k tokens)**. The default stayed RadixArk that day; the 5-pass head-to-head of
  the next night made NVIDIA the default (see the 2026-09-14 update above).
  → [Checkpoints](https://github.com/blazux/qwen3.8-Flash-DGX/blob/383e1f975b4ce9eed3f2e5d6d923d37d54c5c0e1/README.md#checkpoints-nvidias-nvfp4-default-and-radixarks)
- **Download knobs, PLE counters, `KV_CACHE_MEM`** — [@techfury90](https://github.com/techfury90)'s PR #19:
  `XET`/`EXCLUDE`/`MAX_WORKERS` for `download-weights.sh`, five `vllm:ple_mmap_*` Prometheus counters
  (opt-in export with `PROM_MULTIPROC=1`), and `KV_CACHE_MEM` for an explicit KV budget. All verified live.
  We flipped **Xet on by default** right after: the Hub no longer serves files over 50 GB through the plain
  path, and NVIDIA's PLE table is one 50 GiB shard. → [Watching the mmapped table](https://github.com/blazux/qwen3.8-Flash-DGX/blob/383e1f975b4ce9eed3f2e5d6d923d37d54c5c0e1/README.md#watching-the-mmapped-table-vllmple_mmap_)
- **`COMPILE_CACHE`** — [@AronRubin](https://github.com/AronRubin)'s PR #21 keeps vLLM's compiled graphs in
  docker volumes across boots: **init engine 122 s → 41 s** on our box (compilation 34 s → 0.5 s), outputs
  identical. Opt-in; worth it whenever something recreates the container for you.
  → [Persistent compile cache](https://github.com/blazux/qwen3.8-Flash-DGX/blob/383e1f975b4ce9eed3f2e5d6d923d37d54c5c0e1/README.md#optional-persistent-compile-cache-compile_cache)
- `./flash doctor` now reports a checkpoint as **incomplete** when a shard named by the index is missing
  (an interrupted download leaves dangling symlinks and everything looks present).

**2026-09-12** — one command for newcomers, nothing removed for everyone else:

- **`./flash`** — `doctor`, `setup`, `serve <profile>`, `wait`, `test`, `status`, `logs`, `stop`, `start`,
  `rm`. It is a thin front-end over the existing scripts: a profile is a plain env file in `profiles/`
  holding the `serve.sh` variables for one recipe (`default`, `speed`, `context`, `context-1m`, `shared`,
  `published`, `native`, `v0.29`), `setup` runs the build / download / prepare steps only when they are
  not done yet, `doctor` checks the box before you spend an hour downloading. **The scripts and every
  `MODE=… scripts/serve.sh` command in this README keep working exactly as before**; if you already have
  a working setup there is nothing to change. → [The `flash` command](../README.md#the-flash-command)

**2026-09-11** — the recipe runs on the vLLM **v0.29.0** release too:

- **`Dockerfile.v0.29`** builds the same recipe on `vllm/vllm-openai:v0.29.0`, the first official
  release that ships the model natively (as `qwen4_exp`), instead of the Qwen preview image.
  Prompted by [@ChengYen-Tang](https://github.com/ChengYen-Tang) (issue #14). Two of our patches
  are in that release and are dropped there (vllm#50729, the fp8 GEMM `M%4` fix vllm#52775); the
  prefix-caching block-size fix, the GB10 FLA gate, the deterministic top-k kernel, the hybrid
  dispatch and the reduced draft vocabulary are still needed and were re-targeted; the PLE
  mmap patch was rewritten for the new layer. **Measured at parity** on the tournament
  (45/51 at 38.7 tok/s vs 45/51 at 38.5 on the preview image, same two scenarios failed; a
  first run gave 42.5/51 with two reasoning runaways, within the usual variance), decode
  36.4 tok/s, prefill 2,529–3,026 tok/s, KV pool in the same range (~577k; the same recipe on the
  preview image boots anywhere between 565k and 630k depending on the page-cache state at profiling).
  `scripts/serve.sh` reads the base from an image label and adjusts the splitting ops.
  Not ported yet: the fp8 KV cache (patch 7). The preview `Dockerfile` stays the default
  until the v0.29 base has more field time. → [vLLM v0.29.0 as the base image](#vllm-v0290-as-the-base-image-dockerfilev029)

**2026-09-08** — the defaults are now decided by an agentic/coding benchmark (called tournament), and two new ones came out of it:

- **Quality is the gate for defaults now.** Every default in `scripts/serve.sh` is the setting that
  scored best on the 17-scenario agentic tournament (3 repeats); anything that only buys tok/s
  or TTFT is an option. MTP=3 (+7% decode, −1 point), fp8 KV, the M%4 padding and the exact
  top-k fallback are documented options, not defaults.
- **The MTP drafter now scores a 65,536-token vocabulary instead of 248,320** (`DRAFT_VOCAB=1`,
  default): the target verifies every drafted token, so outputs are unchanged; the draft step
  reads 320 MiB of head instead of 1.27 GiB. Measured with the tournament, one variable at a
  time on the same day: **45/51, the best score of any configuration we ran, at 38.5 tok/s
  (+23%)**. Idea taken from [MiaAI-Lab's recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark),
  reimplemented here (their code is AGPL). Same source for the `MADV_RANDOM` advice on the
  mmapped table, now on by default (no readahead: cold prefill −4–8%, cleaner page cache, a
  slightly larger KV pool). → [Reduced draft vocabulary](https://github.com/blazux/qwen3.8-Flash-DGX/blob/383e1f975b4ce9eed3f2e5d6d923d37d54c5c0e1/README.md#reduced-draft-vocabulary-draft_vocab1-default)
- **`MODE=hybrid-mtp`** — [@pfy](https://github.com/pfy)'s graft of Inferact's NVFP4 MTP draft
  experts onto the hybrid checkpoint (PR #11): −3.9 GiB of weights, **+22% KV pool**. On our box
  decode is unchanged (the cheaper draft is accepted less often) and the tournament is neutral
  (44/51), so it ships as an option for people who need context or concurrency more than the
  last percent of quality. → [NVFP4 MTP draft experts](https://github.com/blazux/qwen3.8-Flash-DGX/blob/383e1f975b4ce9eed3f2e5d6d923d37d54c5c0e1/README.md#nvfp4-mtp-draft-experts-modehybrid-mtp-radixark-only)
- Full same-day comparison behind those choices (all hybrid, MTP=2, prefix caching, YaRN 500k):
  exact `torch.topk` 43/51 @31.4 tok/s → deterministic kernel 44/51 @31.9 → `MADV_RANDOM` 44.5/51 @33.5 →
  **reduced draft vocabulary 45/51 @38.5 (default)** → same with MTP=3 44/51 @41.2 (option) → `hybrid-mtp` 44/51 @33.0, KV +22% (option).

**2026-09-07** — kernel pin bump and multi-client guidance:

- **Greedy decoding is deterministic now — at no prefill cost.** The GB10 sparse-attention
  top-k kernel was non-deterministic and dropped candidates, diagnosed and reported upstream by
  [@k3dani](https://github.com/k3dani) (issue #3, vllm#51782). First fixed with an exact
  `torch.topk` (deterministic but −20–40% on long prefill); now replaced by
  [@jschmied](https://github.com/jschmied)'s **deterministic kernel** (vllm#55122), compiled
  into the image: identical outputs at temperature 0 **and** full prefill speed back
  (32k: 1,794 → 2,996 tok/s). `DET_TOPK=1` is the default; `EXACT_TOPK=1` stays as a fallback.
  Kernel pin bumped 2026-09-07 (PR #10): signed-zero fix, low-shared-memory path, a launcher bug
  that would have crashed some long-context widths, and a faster kernel.
  → [Deterministic top-k](https://github.com/blazux/qwen3.8-Flash-DGX/blob/383e1f975b4ce9eed3f2e5d6d923d37d54c5c0e1/README.md#deterministic-top-k-det_topk1-default)

**2026-08-29 → 2026-09-07**:

- **Prefix caching works now** — `--enable-prefix-caching` was crashing, then silently
  returning wrong answers on cache hits. Root cause was a vLLM block-size bug that made
  every prefix hit restore an *all-zero* Mamba state; two-line fix in the image. Getting
  there took [@Saren-Arterius](https://github.com/Saren-Arterius)'s pointer to
  vllm#50729 and their state-copy guard, and [@0xBakeer](https://github.com/0xBakeer)'s
  attempt to reproduce it, which sharpened the write-up.
  `PREFIX_CACHE=1` is the new default. Repeated prefixes (system prompts, multi-turn,
  tool loops) skip the prefill: ~14 s → ~1.4 s TTFT on a 20k-token prefix.
  → [Prefix caching now works](https://github.com/blazux/qwen3.8-Flash-DGX/blob/383e1f975b4ce9eed3f2e5d6d923d37d54c5c0e1/README.md#prefix-caching-now-works-and-why-it-didnt)
- **Optional M%4 padding for the fp8 GEMM** (`PAD_M4=1`, hybrid mode) — the image's blockwise-fp8
  kernel is up to 10× slower on chunks whose row count is not a multiple of 4; the padding is
  [@jschmied](https://github.com/jschmied)'s. With prefix caching on (default) chunks are already
  aligned and it changes nothing, so it is off by default; with `PREFIX_CACHE=0` it is worth about
  −40% TTFT at 8k. → [M%4 padding](#optional-m4-padding-for-the-fp8-gemm-pad_m41)
- **Why decoding stalls when other clients prefill, and the slider for it** — reported by
  [@kutovoy](https://github.com/kutovoy) (issue #9), reproduced and measured: it is vLLM's
  chunked prefill (one decode token per 3–7 s step while a new prompt is being prefilled), not a
  bug. `EXTRA='--long-prefill-token-threshold 1024'` trades single-stream TTFT for
  responsiveness; numbers and guidance in [the concurrency section](https://github.com/blazux/qwen3.8-Flash-DGX/blob/383e1f975b4ce9eed3f2e5d6d923d37d54c5c0e1/README.md#decoding-clients-stall-while-other-clients-prefill-the-long-prefill-token-threshold-slider).
- **Audit fixes** — [@sternnick](https://github.com/sternnick) audited the repo line by line
  against their own Spark (issue #8). Taken so far: the files fetched from @jschmied's repo are
  pinned by sha256 as well as by commit (and that repo is Apache-2.0 now), the fp8-KV guard only
  admits `e4m3` (the kernel launch never handled `e5m2`), `GPU_MEM` defaults to `0.80` as the
  docs already recommended, and the undocumented `VLLM_PLE_MMAP_CHUNK` and the no-auth binding
  are in the docs. Their three fixes are merged with their authorship: the measured sizes
  (the checkpoint is 126 GiB, the table 48 GiB, plan 140 GB of disk), the PLE range guard
  (the last shard is partial), and the scripts (snapshot resolved from `refs/main`, a start
  check after `docker run`, the prefix-cache hit proven with `vllm:prefix_cache_hits_total`
  instead of a stopwatch).
- **Two checkpoint modes** — `MODE=nvfp4` (as published) or `MODE=hybrid` (NVFP4 experts
  + fp8 side layers, one-time `scripts/prepare-hybrid.sh`): **+20% decode, +8% KV,
  same quality**. Our box runs the hybrid. The fp8 side-layer conversion and the
  original int4+fp8 dispatch it is ported from are
  [@Saren-Arterius](https://github.com/Saren-Arterius)'s. → [Two checkpoint modes](https://github.com/blazux/qwen3.8-Flash-DGX/blob/383e1f975b4ce9eed3f2e5d6d923d37d54c5c0e1/README.md#two-checkpoint-modes-nvfp4-or-hybrid)
- **Also in the image**: vllm#50729 (Mamba state-copy race, by
  [@AndreasKaratzas](https://github.com/AndreasKaratzas)) + a bounds guard, the GB10 FLA
  fixes and the faster PLE gather from [@Saren-Arterius](https://github.com/Saren-Arterius)'s fork.
- **We benchmarked the int4 (Intel AutoRound) variant too** with the same patches:
  fastest raw decode, but not deterministic and slowest cached-TTFT, so we did not adopt
  it. Numbers in [docs/HOW-IT-WORKS.md](HOW-IT-WORKS.md#hybrid-mode-nvfp4-experts--blockwise-fp8-side-layers).
- **fp8 KV cache is available** (`KV_DTYPE=fp8_e4m3`), contributed by
  [@Nanetnounou](https://github.com/Nanetnounou): ×1.9 KV, 1M context on one box — at a
  speed and quality cost, so it is opt-in. → [fp8 KV cache](HOW-IT-WORKS.md#fp8-kv-cache-on-the-qsa-path-opt-in)
- `scripts/smoke-test.sh` now also checks the prefix-cache hit and determinism, and
  measures decode on a real answer instead of `ignore_eos` (which produces meaningless
  numbers with this model). `scripts/download-weights.sh` now forwards `HF_TOKEN`
  ([@wawimundo](https://github.com/wawimundo), PR #4).

Everything was measured on one ASUS GX10 with a 17-scenario agentic tournament (3 repeats
each), single-request speed benches on real prompts, and state checksums for the
prefix-caching work; nothing here is extrapolated.

| | llama.cpp IQ4_XS | **NVFP4 (this repo)** | **hybrid (this repo)** |
|---|---|---|---|
| Prefill | ~540 tok/s | **~2,400–2,900 tok/s** (deterministic kernel; warm page cache — a first pass over a cold region of the table reads from NVMe and can be 2–3× slower, see `PREWARM`) | same |
| Decode, single stream | ~22 tok/s (no MTP) | **~26 tok/s** with MTP=2 | **~37 tok/s** (reduced draft vocabulary; ~31 without) |
| Prefix-cache hit, TTFT on a 20k-token prefix | n/a | **~1.4 s** (vs ~14 s cold) | same |
| Context | 262k | **262k native, 500k with YaRN** | same |
| KV cache @0.80 (500k YaRN, MTP) | — | ~580k tokens | ~630k tokens |
| Deterministic at temperature 0 (sequential requests) | yes | **yes** (`DET_TOPK=1`) | **yes** |

*Measured on an ASUS GX10 (GB10, 128 GB), single request, real prompts, greedy. Quality
(a 17-scenario agentic tournament, 3 repeats) is identical across NVFP4 and hybrid:
45/51 both, same two scenarios failed by every quantization we tried. Details and the
full comparison tables are in [docs/HOW-IT-WORKS.md](HOW-IT-WORKS.md).*

---

## vLLM v0.30.0 as the base image (`Dockerfile.v0.30`)

vLLM **v0.30.0** brings Qwen3.8-Flash-Next work of its own: separate prefill and decode QSA
indexer kernels (vllm#54513), fused PLE kernels (vllm#54517) and an FP8 indexer cache (vllm#54890).
It also moved enough code that half of our patches had to be re-targeted:

```bash
./flash setup v0.30                     # builds Dockerfile.v0.30 (and checks weights, hybrid layout)
./flash serve v0.30
# or by hand
docker build -f Dockerfile.v0.30 -t qwen38-flash-dgx:v0.30 .
IMAGE=qwen38-flash-dgx:v0.30 MODE=hybrid YARN=1 CTX=500000 scripts/serve.sh
```

| patch | v0.29 base | v0.30 base |
|---|---|---|
| 1 PLE mmap | swaps `PLEVocabParallelEmbedding` | re-targeted: the n-gram module moved to `ngram_embedding.py`, the hashing became a Triton kernel and the lookup goes through `Qwen4ExpPLEDeviceEmbedding`; `apply()` detects the layout |
| 1, CUDA graphs | our gather is a splitting op | v0.30 captures this model with **breakable CUDA graphs** (torch.compile off, no FX splitting), so the gather breaks the capture itself, writes into a static per-layer buffer, and does not read the not-yet-computed ids at capture time |
| 5 exact top-k, 8 deterministic kernel | `ops/qsa.py` | re-targeted to the shared `_topk` of `ops/qsa_indexer.py`; the kernel wiring is `src/patch_qsadet.py` (covers both layouts) |
| 11 block-FP8 MTP experts | vllm#55513 backport | **in the release, dropped** |
| 12, 13 Qwen tool-marker fixes | `src/patches/*.patch` | same fixes rebased on the new parser engine (`src/patches/*-v030.patch`), same 30-case test module |
| 2, 4, 6, 10, 14 | as is | unchanged |
| 7 fp8 KV cache | not ported | **ported** (`src/patch_qsa_fp8_kv_v030.py`): only the main attention's read path and guards; the indexer caches are v0.30's own. Inert in bf16 (log-probs 5/5 identical) |

Measured on our GX10, NVIDIA checkpoint, hybrid, YaRN 500k, MTP=2, head-to-head with the v0.29 image
that ran production:

| | v0.29 base (production until 2026-09-25) | v0.30 base |
|---|---|---|
| Tournament, 55 scenarios, temperature 0.2 | 88.8% ± 1.0 (5 runs) | **87.8% ± 0.5 (3 runs)** — difference not detectable (paired bootstrap CI [−1.1, +3.5] points, p = 0.18) |
| Losses that are not reasoning runaways | `v3` partial (1 run), `g31` 25/26 cells | same two |
| Runaways on the coding tasks | 66% | 63% |
| Decode, single stream (median of 6, cold / warm) | 33.2 / 34.5 tok/s | 34.0 / 34.3 tok/s |
| Prefill, cold table region, 8k / 32k | 1,597 / 2,823 tok/s | **2,689 / 4,002 tok/s** |
| Needles at 185k / 323k / 413k tokens | found in 100 / 131 / 148 s | **found in 68 / 120 / 121 s** |
| MTP acceptance after probes | 66.2% | 74.0% |
| KV pool @0.80 | 679k tokens | 641k–714k tokens with patch 14 only; **518k** with patches 15–18, no swap at profiling (see below) |
| Deterministic at temperature 0 (sequential) | 2/2 | 2/2 |

Greedy outputs are not token-identical across the two bases (new kernels, new NVFP4 W4A4 default on
sm_121): on our five reference prompts three came out identical and two diverged at low-confidence
tokens, both answers correct. That is why the comparison is a tournament, not a diff.

## vLLM v0.29.0 as the base image (`Dockerfile.v0.29`)

The default `Dockerfile` patches Qwen's preview image (`qwenllm/qwen3.8-flash-next-vllm`).
vLLM **v0.29.0** is the first official release that ships the model natively — the package
moved to `vllm/models/qwen4_exp`, there is an arm64 image, and two of the fixes this repo
carried are in the release. `Dockerfile.v0.29` builds the same recipe on top of it:

```bash
docker build -f Dockerfile.v0.29 -t qwen38-flash-dgx:v0.29 .
IMAGE=qwen38-flash-dgx:v0.29 MODE=hybrid YARN=1 CTX=500000 scripts/serve.sh
```

Same weights, same snapshot, same env knobs and defaults (`MODE`, `DET_TOPK`, `DRAFT_VOCAB`,
`MADVISE`, `PREFIX_CACHE`, `MTP`, …). Both images carry a `qwen38.base` label and
`scripts/serve.sh` reads it to pick the right splitting-op names (`BASE=preview|v0.29` overrides).

What changes in the patch set:

| patch | preview image | v0.29 base |
|---|---|---|
| 1 PLE mmap | hooks `forward_impl` | rewritten: the release computes the n-gram ids in a GPU op and looks them up in a `PLEVocabParallelEmbedding`; we swap that layer for the mmapped table and keep the stock id computation (same file, `apply()` detects the layout) |
| 2 GB10 FLA fixes | needed | needed (same file) |
| 3 vllm#50729 Mamba state-copy race | needed | **in the release, dropped** |
| 4 prefix-caching block_size | needed | needed (`core.py` still takes the smallest group block size) |
| 5 exact top-k, 8 deterministic kernel | needed | needed, re-targeted (vllm#55122 is still open) |
| 6 hybrid dispatch, 10 reduced draft vocabulary | needed | needed, re-targeted |
| 7 fp8 KV cache | opt-in | **not ported yet** — `KV_DTYPE` must stay `auto`, serve.sh refuses otherwise |
| 9 `M%4` padding | opt-in | **in the release (vllm#52775), dropped**; `PAD_M4` is a no-op there |
| 11 block-FP8 MTP experts in mixed ModelOpt checkpoints | FP8_BLOCK_SCALES shim (`src/vllm_modelopt_block_moe.py`) | **vllm#55513 backported** in place of the shim: NVIDIA-base checkpoints can use MTP; a no-op for RadixArk's |

Two things got simpler on the release: the Inductor int64-indexing assert that forced
`torch.compile` off on the preview image is gone (compile is on, graphs stay PIECEWISE
because the gather still has to run between graph segments), and the FLA/short-conv kernels
need no `--enforce-eager` workarounds.

Measured on our GX10, hybrid, the default recipe, YaRN 500k, same day as the preview numbers:

| | preview image (default) | v0.29 base |
|---|---|---|
| Tournament (17 × 3, temperature 0.2) | 45/51 @ 38.5 tok/s | **45/51 @ 38.7 tok/s** (run 2); 42.5/51 @ 39.2 (run 1, two reasoning runaways) |
| Decode, single stream (median of 6) | ~37 tok/s | 36.4 tok/s |
| Prefill, warm page cache | ~2,500–3,000 tok/s | 2,529 (8k) / 3,026 (32k) tok/s |
| Prefill, cold table region | | 2,513 (8k) / 2,447 (32k) tok/s |
| Needle at 92k | ~45 s | 45.4 s |
| MTP acceptance (reduced vocabulary) | ~68% | 64.8% |
| KV pool @0.80 | 565k–630k tokens (varies with the page-cache state at profiling) | ~577k tokens (two boots) |
| Deterministic at temperature 0 / prefix-cache hit bit-exact | yes / yes | yes (4/4) / yes (log-prob delta 0.0000) |

The two tournament runs fail exactly the same scenarios as the preview image (the two that
every quantization we tried fails); the 42.5 of the first run is two `length` finishes, the
reasoning runaways we see in about one run out of three on any configuration. So: parity,
with a slightly lower draft acceptance that we have not chased yet (the KV pool is within the
boot-to-boot range of the preview image). **The preview `Dockerfile` remains the default** until this base has more
field time on our own box; if you want to be on the release line, it is ready and tested.
Full port notes in [docs/HOW-IT-WORKS.md](#the-vllm-v0290-port-dockerfilev029).

## Optional: M%4 padding for the fp8 GEMM (`PAD_M4=1`)

Hybrid mode runs the GDN/QSA side layers and shared experts through vLLM's blockwise-fp8
cutlass GEMM. On this image (sm_12x) that kernel routes any call whose row count M is not a
multiple of 4 (or ≤ 64) to a `swap_ab` path that is much slower — upstream fixed it in C++
([vllm#52775](https://github.com/vllm-project/vllm/pull/52775)) after the image was cut.
[@jschmied](https://github.com/jschmied) found it and wrote a drop-in that pads M to a
multiple of 4 inside an opaque custom op (`fp8_m4pad_patch.py`, patch 9 in the Dockerfile,
fetched at a pinned commit; issue #3).

Measured on the GX10 at the kernel level (K=4096, N=8192): ×1.7 below 2,048 rows
(0.63 → 1.09 ms at M=1,601), **×10–11 above** (0.87 → 9.6 ms at M=2,401); padding restores the
aligned time in every case. At the server level it depends on how the scheduler cuts prefill
chunks:

- **`PREFIX_CACHE=1` (default): no-op.** The Mamba align mode clips every prefill chunk to the
  1,600-token block boundary, so M % 4 == 0 on all large chunks. Same-session A/B on the hybrid
  (MTP=2): 8k 3.33 → 3.24 s, 32k 11.63 → 10.97 s, salted repeats within noise, and prompts built
  to leave a misaligned last chunk (8,801 / 8,803 tokens) showed no penalty either. Off by default.
- **`PREFIX_CACHE=0`: use it.** Chunks are then whatever the batch size leaves (an 8,001-token
  prompt is one 8,001-row chunk); @jschmied measured −40% TTFT at 8k and −10–15% at 30k on the
  stock image, and the unpatched kernel is bimodal (2.9–6.6 s at 8k depending on the cut).

`PAD_M4=1` also sets `VLLM_FP8_PAD_M4=1`; `scripts/serve.sh` always passes the variable because
the patch itself defaults to on when it is unset. NVFP4 mode does not use this GEMM.

---

*Moved from docs/HOW-IT-WORKS.md on 2026-09-28.*

## The blockwise-fp8 GEMM's `M % 4` slow path (patch 9, opt-in)

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

## fp8 KV cache on the QSA path, preview image (opt-in)

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

## The vLLM v0.29.0 port (`Dockerfile.v0.29`)

Everything above was built on Qwen's preview image (`qwenllm/qwen3.8-flash-next-vllm`,
a vLLM `0.20.x` dev build carrying the model as `vllm/models/qwen3_8_flash_next`). vLLM
**v0.29.0** is the first official release with the model in-tree, renamed
`vllm/models/qwen4_exp` (classes `Qwen4Exp*`, ops `vllm::qwen4_exp_*`), with an arm64
image. [@ChengYen-Tang](https://github.com/ChengYen-Tang) asked whether the recipe would
move ([issue #14](https://github.com/blazux/qwen3.8-Flash-DGX/issues/14)); this is what the
port took and what it measures.

**Patch by patch.** Every path in the Dockerfile moves from
`vllm/models/qwen3_8_flash_next/nvidia/` to `vllm/models/qwen4_exp/nvidia/`; beyond that:

- **3 (vllm#50729, Mamba state-copy race) and 9 (fp8 GEMM `M%4`, vllm#52775) are in the
  release** and are not applied. `PAD_M4` is therefore a no-op on this base and
  `scripts/serve.sh` says so.
- **4 (prefix-caching block size) is still needed.** `v1/engine/core.py` still overwrites
  `cache_config.block_size` with the *smallest* group block size; the worker's align-mode
  state-slot seed and the scheduler's block-aligned prefill split now read
  `mamba_block_size` / the scheduler's own `block_size` (the LCM of the groups), so the
  same two-line fix applies. Prefix-cache hits are bit-exact (log-prob delta 0.0000 on
  cached vs uncached prefills, four prompts).
- **1 (PLE mmap) is rewritten.** The preview layer did hashing and lookup in one Python
  `forward_impl`, which we replaced wholesale. The release splits it: a compiled op
  `qwen4_exp_compute_ple_ngram_ids` computes the n-gram ids on the GPU, then a
  `PLEVocabParallelEmbedding` (in `common/ple.py`) looks them up and hands the layer a
  `weight_scale` for the FP8 dequant. We keep the stock id op and swap only the embedding:
  `apply()` detects the layout (no `forward_impl` → v0.29), wraps `__init__` so the
  embedding is our `_MmapNgramEmbedding` (the `ple_layer` module's reference to
  `PLEVocabParallelEmbedding` is replaced for the duration of the constructor), overrides
  `load_weights` to drop the table shards and keep only `weight_scale`, and overrides
  `forward` to call the stock id op followed by a new splitting op,
  `vllm::ple_mmap_lookup_ids(ngram_ids, output, layer_name)`, which gathers the rows from
  the mmap into a pinned buffer and copies them into `output`. Same `MmapPleTable`, same
  workers/chunk/prewarm/madvise knobs, same 48 GiB saved. One structural difference: the
  ids now live on the GPU, so each lookup starts with a device→host copy of 16 × N int64s
  (the preview computed them on the CPU). In the decode logs it is invisible (1.9–2.1 ms
  per op vs 1.1–1.3 ms of pure gather); on prefill it is inside the same 2,500–3,000 tok/s
  band as the preview image.
- **2, 5, 6, 8, 10 apply as-is** once re-targeted. The draft-vocabulary hook now finds the
  MTP class by pattern (`class \w+MTP\(`) instead of by name, so one file serves both bases.
- **7 (fp8 KV on the QSA path) is not ported yet.** The QSA Triton kernels moved and were
  edited upstream; the patch needs a re-derivation, not a path change. `scripts/serve.sh`
  refuses `KV_DTYPE≠auto` on this base rather than silently running bf16.
- **11 differs by base; on this one it is a backport of vllm#55513.** NVIDIA's own checkpoint
  needs it for MTP: the drafter's experts are blockwise fp8 under a mixed-precision config, and
  v0.29.0 misses them in two ways (see
  [NVIDIA's NVFP4 checkpoint](#nvidias-nvfp4-checkpoint-block-fp8-mtp-experts-under-a-mixed-precision-config-patch-11-temporary)).
  `src/patch_block_fp8_mtp.py` applies the PR's two runtime changes. The draft config now moves
  `quantized_layers` to the drafter's runtime index, as it already did `exclude_modules`, and the
  mixed config sends `FP8_PB_WO` / `FP8_BLOCK_SCALES` experts to vLLM's own `Fp8MoEMethod` with a
  block `Fp8Config`. It leaves out the PR's `has_blocked_weights` hunk: v0.29.0's mixed config has
  no such method, and that gate only picks the CUDA `QuantFP8` op for speed, while the MoE path
  quantizes its input with `per_token_group_quant_fp8` directly. RadixArk's checkpoint (quant_algo
  `NVFP4`, no `quantized_layers`) is unaffected. The preview image keeps the FP8_BLOCK_SCALES shim.
  Measured on a DGX Spark with an NVIDIA-base checkpoint (`MODE=nvfp4`, MTP=2):
  - the draft experts load on vLLM's DeepGEMM FP8 MoE backend;
  - draft acceptance is 70.9% over four greedy prompts (539 of 760 drafted tokens, identical on two builds);
  - the KV pool is 517k–542k tokens at `GPU_MEM=0.80` across three boots;
  - the smoke test's determinism and prefix-cache checks pass.

  The hybrid layout works on it too. With `prepare-hybrid.sh`'s fp8 side layers (`MODE=hybrid`, which
  sets `VLLM_USE_DEEP_GEMM=0`), the draft experts load on the Triton FP8 MoE backend and the KV pool
  grows to 658,980 tokens. Draft acceptance on the same four prompts is 65.9% (382 of 580 drafted
  tokens; the fp8 side layers change the greedy paths), and the smoke test passes.

**Serving.** The splitting-op list changes names (`vllm::qwen4_exp_ple_short_conv`,
`vllm::qwen4_exp_qsa_with_output`, and `vllm::qwen4_exp_compute_ple_ngram_ids` must be in
it too, or the id op gets captured with the lookup after it), and the new lookup op is
`vllm::ple_mmap_lookup_ids`. Both Dockerfiles stamp `LABEL qwen38.base=preview|v0.29` and
`serve.sh` reads it, so the same command line works on both. The Inductor int64 assert
that forced `torch.compile` off on the preview image does not fire on the release, so
compile is on (≈28 s at boot); graphs stay PIECEWISE for the reason in
[Three GB10 bugs](HOW-IT-WORKS.md#three-gb10-bugs-this-works-around).

**Measured** (GX10, hybrid, default recipe: deterministic top-k, reduced draft vocabulary,
`MADV_RANDOM`, prefix caching, MTP=2, YaRN 500k, `GPU_MEM=0.80`):

| | preview image | v0.29 base |
|---|---|---|
| KV pool | 565k–630k tokens (the same recipe, boot to boot: the profiler's headroom depends on the page-cache state) | 575,757–578,787 tokens (two boots) |
| Determinism (4 prompts × repeats, temperature 0) | 4/4 | 4/4 |
| Decode, single stream, median of 6 | ~37 tok/s | 36.4 tok/s (33.8–41.5) |
| Prefill warm, 8k / 32k | ~2,500–3,000 tok/s | 2,529 / 3,026 tok/s |
| Prefill cold table region, 8k / 32k (×3, salted prompts) | | 2,504–2,513 / 2,444–2,448 tok/s |
| Needle at 92,157 tokens | ~45 s | found, 45.4 s |
| MTP draft acceptance (65,536-id vocabulary) | ~68% | 64.8% (8 samples) |
| Tournament, 17 scenarios × 3, temperature 0.2 | 45/51 @ 38.5 tok/s | run 1: 42.5/51 @ 39.2 (two `length` finishes, one partial); run 2: **45/51 @ 38.7**, 0 errors, no runaways |

Run 2 fails exactly the scenarios the preview image fails (`b6_reconcile`,
`c5_inventory_reconcile`, which every quantization we tried fails). Run 1's deficit is two
reasoning runaways that hit the token cap, which we see in roughly one run in three on any
configuration; it is the day-to-day variance of this benchmark, not a property of the base.
Verdict: parity in quality, speed and KV pool, and −3 points of draft acceptance that we have
not investigated. The preview `Dockerfile` stays the default and our production image for
now; `Dockerfile.v0.29` is the tested path onto the release line, and will become the
default once fp8 KV is ported and it has run in production for a while.

## NVIDIA's NVFP4 checkpoint: block-fp8 MTP experts under a mixed-precision config (patch 11, temporary)

`nvidia/Qwen3.8-Flash-Next-NVFP4` declares `quant_algo: MIXED_PRECISION` with a per-layer map:
the routed experts are NVFP4 (as in RadixArk), the PLE table is FP8, and the MTP drafter's
experts are **`FP8_BLOCK_SCALES`, group 128** — fp8 `weight` + fp32 `weight_scale_inv` per
128×128 block, the DeepSeek-V3 layout — where RadixArk keeps them in bf16.

vLLM 0.29's `ModelOptMixedPrecisionConfig` resolves per-layer algorithms but only maps FP8,
FP8_PB_WO, NVFP4, W4A16_NVFP4 and MXFP8 to methods. Two things go wrong for the drafter: the
map's key is `mtp.layers.0.mlp.experts` while vLLM builds the layer as
`mtp.layers.<num_hidden_layers>.mlp.experts` (the model remaps `exclude_modules` for that
offset but not `quantized_layers`), and even with the name matched there is no method for the
algorithm. The layer is created unquantized (bf16 parameters) and weight loading dies with
`Layer mtp.layers.48.mlp.experts has no parameter 'w2_weight_scale_inv'`.

`src/vllm_modelopt_block_moe.py`, the shim the preview image still uses, fixes both at the layer:
it hooks `RoutedExperts._get_quant_method`, reads `quantized_layers` from the served checkpoint,
matches the drafter's entry by its tail (`.mlp.experts`) under either spelling of the index,
and returns vLLM's own `Fp8MoEMethod` with `Fp8Config(weight_block_size=[128, 128])` — the
same method DeepSeek-V3 checkpoints use. (A first version hooked the config class only; in
practice the expert layer never reached it, hence the layer-level hook.) On the v0.29 base,
before the backport below replaced it there, vLLM picked the DeepGEMM fp8 MoE backend for it on
GB10 and it works: drafter acceptance 83–89%, decode 27.7 tok/s on the published layout and
34.0 on the hybrid, deterministic, needle 6/6 to 413k. The shim is inert for checkpoints without
`FP8_BLOCK_SCALES` layers, and `VLLM_MODELOPT_BLOCK_MOE=0` disables it.

**On the preview image this shim is still a stopgap, not the fix.** The proper fix landed upstream
as vllm#55513 (merged 2026-09-08, after the 0.29 release, and not yet in a release): a
`quantized_layers` remap in the MTP and a block-fp8 MoE branch in the mixed config, at the source of
both gaps. The v0.29 image carries a backport of it instead of the shim (patch 11 on that base,
`src/patch_block_fp8_mtp.py`, by @techfury90, with a CPU test; see
[the v0.29 port](#the-vllm-v0290-port-dockerfilev029)), and the shim is removed there. With the index
remapped where it goes wrong, the expert layer reaches the config's method, so that base needs no
layer-level hook.

The hybrid layout needed one more change: `vllm_fp8_hybrid_modelopt.py` used to patch only
`ModelOptNvFp4Config`; on the mixed config the fp8-converted side layers were caught by the
checkpoint's exclude list and sent to the bf16 path. It now patches both classes.

## Persistent compile cache on the preview base (PR #21)

The README's measurements for `COMPILE_CACHE` as it was introduced, with two volumes
(`<name>-vllm`, `<name>-flashinfer`), on the preview image. On v0.30 the picture differs (no
torch.compile cache, Triton and the driver JIT dominate); see the README section.

Measured on a GX10, hybrid + YaRN 500k + MTP=2, same recipe each time. **Boot totals are not usable
for this**: weight loading varied between 464 s and 554 s on page-cache state alone, and CUDA-graph
capture between 4 s and 13 s, both larger than the effect being measured. The signal is in init
engine with capture excluded, one row per boot:

| boot | init engine | capture | init engine − capture | `torch.compile` |
|---|---|---|---|---|
| 1 — unset (default) | 129.2 s | 13 s | 116.2 s | 37.9 s |
| 2 — set, populating | 125.5 s | 10 s | 115.5 s | 37.5 s |
| 3 — set, reused | 41.1 s | 4 s | 37.1 s | 4.2 s |
| 4 — set, reused, Triton volume emptied | 50.2 s | 13 s | 37.2 s | 0.7 s |
| 5 — set, reused, final two-mount config | 37.3 s | 4 s | 33.3 s | 0.7 s |

Reused (boots 3–5) is **35.9 s ± 2 s**, against **116.2 s** with the cache off: **−80 s ± 2 s
(−69%)** per boot. Populating it costs nothing (boot 2 at 115.5 s against boot 1 at 116.2 s).
Reused boots log `Directly load AOT compilation from path …`. Disk: 168 MB for the vLLM cache,
0.5 MB for FlashInfer. Startup only — no effect on outputs, so nothing for the tournament to say.

**Triton's `/root/.triton` is deliberately not persisted.** It looks like it should be the
interesting one: `jit_monitor` warns that five kernels (`_qsa_mqa_paged_kernel`,
`_qsa_sparse_paged_gqa_splitk`, `_compute_local_logits_stats_`, `_rejection_kernel`,
`_resample_kernel`) JIT-compile *during the first request*. Boot 4 above tested it directly — vLLM
cache warm, only the Triton volume emptied — and came out at 37.2 s against boot 3's 37.1 s: a
0.2 s difference, an order of magnitude below the capture noise. The five warnings appear in every
boot either way, warm or cold. Mounting it would have been cargo cult.
