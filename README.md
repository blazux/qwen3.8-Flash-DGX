# Qwen3.8-Flash-Next on a single DGX Spark (GB10)

Run **Qwen3.8-Flash-Next** — a ~176B-parameter model (125B main + 51B n-gram, 6B
active) — on **one NVIDIA DGX Spark / ASUS GX10** with **vLLM**, at full prefill
speed, with MTP speculative decoding, **working prefix caching**, **deterministic
greedy decoding**, and up to **500k tokens of context**.

The catch this repo solves: the NVFP4 checkpoint is **126 GiB**, which does not fit
next to a usable KV cache in the Spark's **128 GB unified pool**. 48 GiB of that is
the n-gram embedding ("PLE") table — a pure lookup that a token only touches 16 rows
of. This repo patches the official vLLM image to **serve that table from NVMe via
`mmap`** instead of keeping it resident. Weights drop to **~76 GiB**, the rest of the
pool goes to KV, and everything runs on stock GB10 kernels.

Along the way it also fixes two things that were broken for this model on GB10 —
**prefix caching** (a vLLM block-size bug silently restored an all-zero Mamba state
on every cache hit) and **non-deterministic top-k in the sparse attention** (a GB10
kernel that drops candidates) — and offers an optional **hybrid** checkpoint layout
(NVFP4 experts + fp8 side layers) that decodes ~20% faster at the same quality.

## TL;DR — run it on a DGX Spark

```bash
git clone https://github.com/blazux/qwen3.8-Flash-DGX.git && cd qwen3.8-Flash-DGX
./flash doctor      # docker, GPU, memory, disk, port, image, weights: tells you what is missing
./flash setup       # builds the image, downloads the checkpoint (126 GiB, resumable), prepares the hybrid layout
./flash serve       # the recommended recipe (profile "default"): hybrid, 500k context, deterministic
./flash wait        # first boot loads ~76 GiB of weights, 8-13 min; prints the KV pool when the API is up
./flash test        # health, coherence, prefix-cache hit, determinism, tok/s
```

Other recipes are one word away: `./flash profiles` lists them (`speed`, `context`, `context-1m`,
`shared`, `published`, `native`, `v0.29`), `./flash serve speed` runs one, and any variable can still
be overridden on the command line (`./flash serve default MTP=3 PORT=18301`). `./flash status`,
`logs`, `stop`, `start`, `rm` do what they say. Details in [The `flash` command](#the-flash-command).

The same thing by hand, unchanged and still supported (everything `flash` does is these scripts):

```bash
docker build -t qwen38-flash-dgx .            # ~1 min: official vLLM image + the 10 patches below
scripts/download-weights.sh                   # RadixArk NVFP4 checkpoint, ~126 GiB, resumable (one-time)
scripts/prepare-hybrid.sh                     # recommended: fp8 side layers, +20% decode, same quality (~10 min, one-time)
MODE=hybrid YARN=1 CTX=500000 scripts/serve.sh   # the recipe our own box runs; 500k context, ~13 min to load
docker logs -f qwen38-flash                   # ready at "Application startup complete"
scripts/smoke-test.sh                         # health, coherence, prefix-cache hit, determinism, tok/s
```

OpenAI-compatible API on `http://localhost:18300/v1`, model name `qwen3.8-flash-next`,
tool calling and reasoning parsers on. Every default is the setting that scored best on our
agentic tournament (see [How the defaults are chosen](#how-the-defaults-are-chosen-quality-first-speed-as-an-option));
what you get on a GX10: ~37 tok/s single-stream decode, ~2,500–3,000 tok/s prefill, prefix
caching, deterministic greedy output, 500k tokens of context. Want the checkpoint exactly as
published? Drop `prepare-hybrid.sh` and `MODE=hybrid`. Want speed over the last percent of
quality? `MTP=3`, and `MODE=hybrid-mtp` for more KV — both explained in the [options table](#how-the-defaults-are-chosen-quality-first-speed-as-an-option).
Prefer the current vLLM release to the preview image? `docker build -f Dockerfile.v0.29 -t qwen38-flash-dgx:v0.29 .`
and `IMAGE=qwen38-flash-dgx:v0.29` — same recipe, same defaults, measured at parity
(see [vLLM v0.29.0 as the base image](#vllm-v0290-as-the-base-image-dockerfilev029)).
Everything below is the long version: what was broken on GB10, what was fixed, and the numbers.

> **Independently reproduced** on a DGX Spark by
> [@jschmied](https://github.com/jschmied) — see
> [issue #1](https://github.com/blazux/qwen3.8-Flash-DGX/issues/1) and their
> [write-up](https://github.com/jschmied/qwen38-flash-next-gb10).

## Update 2026-09-11 — what changed

Newest first. If you cloned this before, this is the short version; details in the linked sections.

**2026-09-12** — one command for newcomers, nothing removed for everyone else:

- **`./flash`** — `doctor`, `setup`, `serve <profile>`, `wait`, `test`, `status`, `logs`, `stop`, `start`,
  `rm`. It is a thin front-end over the existing scripts: a profile is a plain env file in `profiles/`
  holding the `serve.sh` variables for one recipe (`default`, `speed`, `context`, `context-1m`, `shared`,
  `published`, `native`, `v0.29`), `setup` runs the build / download / prepare steps only when they are
  not done yet, `doctor` checks the box before you spend an hour downloading. **The scripts and every
  `MODE=… scripts/serve.sh` command in this README keep working exactly as before**; if you already have
  a working setup there is nothing to change. → [The `flash` command](#the-flash-command)

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
  slightly larger KV pool). → [Reduced draft vocabulary](#reduced-draft-vocabulary-draft_vocab1-default)
- **`MODE=hybrid-mtp`** — [@pfy](https://github.com/pfy)'s graft of Inferact's NVFP4 MTP draft
  experts onto the hybrid checkpoint (PR #11): −3.9 GiB of weights, **+22% KV pool**. On our box
  decode is unchanged (the cheaper draft is accepted less often) and the tournament is neutral
  (44/51), so it ships as an option for people who need context or concurrency more than the
  last percent of quality. → [NVFP4 MTP draft experts](#nvfp4-mtp-draft-experts-modehybrid-mtp)
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
  → [Deterministic top-k](#deterministic-top-k-det_topk1-default)

**2026-08-29 → 2026-09-07**:

- **Prefix caching works now** — `--enable-prefix-caching` was crashing, then silently
  returning wrong answers on cache hits. Root cause was a vLLM block-size bug that made
  every prefix hit restore an *all-zero* Mamba state; two-line fix in the image. Getting
  there took [@Saren-Arterius](https://github.com/Saren-Arterius)'s pointer to
  vllm#50729 and their state-copy guard, and [@0xBakeer](https://github.com/0xBakeer)'s
  attempt to reproduce it, which sharpened the write-up.
  `PREFIX_CACHE=1` is the new default. Repeated prefixes (system prompts, multi-turn,
  tool loops) skip the prefill: ~14 s → ~1.4 s TTFT on a 20k-token prefix.
  → [Prefix caching now works](#prefix-caching-now-works-and-why-it-didnt)
- **Optional M%4 padding for the fp8 GEMM** (`PAD_M4=1`, hybrid mode) — the image's blockwise-fp8
  kernel is up to 10× slower on chunks whose row count is not a multiple of 4; the padding is
  [@jschmied](https://github.com/jschmied)'s. With prefix caching on (default) chunks are already
  aligned and it changes nothing, so it is off by default; with `PREFIX_CACHE=0` it is worth about
  −40% TTFT at 8k. → [M%4 padding](#optional-m4-padding-for-the-fp8-gemm-pad_m41)
- **Why decoding stalls when other clients prefill, and the slider for it** — reported by
  [@kutovoy](https://github.com/kutovoy) (issue #9), reproduced and measured: it is vLLM's
  chunked prefill (one decode token per 3–7 s step while a new prompt is being prefilled), not a
  bug. `EXTRA='--long-prefill-token-threshold 1024'` trades single-stream TTFT for
  responsiveness; numbers and guidance in [the concurrency section](#decoding-clients-stall-while-other-clients-prefill-the-long-prefill-token-threshold-slider).
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
  [@Saren-Arterius](https://github.com/Saren-Arterius)'s. → [Two checkpoint modes](#two-checkpoint-modes-nvfp4-or-hybrid)
- **Also in the image**: vllm#50729 (Mamba state-copy race, by
  [@AndreasKaratzas](https://github.com/AndreasKaratzas)) + a bounds guard, the GB10 FLA
  fixes and the faster PLE gather from [@Saren-Arterius](https://github.com/Saren-Arterius)'s fork.
- **We benchmarked the int4 (Intel AutoRound) variant too** with the same patches:
  fastest raw decode, but not deterministic and slowest cached-TTFT, so we did not adopt
  it. Numbers in [docs/HOW-IT-WORKS.md](docs/HOW-IT-WORKS.md#hybrid-mode-nvfp4-experts--blockwise-fp8-side-layers).
- **fp8 KV cache is available** (`KV_DTYPE=fp8_e4m3`), contributed by
  [@Nanetnounou](https://github.com/Nanetnounou): ×1.9 KV, 1M context on one box — at a
  speed and quality cost, so it is opt-in. → [fp8 KV cache](docs/HOW-IT-WORKS.md#fp8-kv-cache-on-the-qsa-path-opt-in)
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
| Deterministic at temperature 0 | yes | **yes** (`DET_TOPK=1`) | **yes** |

*Measured on an ASUS GX10 (GB10, 128 GB), single request, real prompts, greedy. Quality
(a 17-scenario agentic tournament, 3 repeats) is identical across NVFP4 and hybrid:
45/51 both, same two scenarios failed by every quantization we tried. Details and the
full comparison tables are in [docs/HOW-IT-WORKS.md](docs/HOW-IT-WORKS.md).*

---

## How the defaults are chosen: quality first, speed as an option

Every default in `scripts/serve.sh` is the setting that scored best on our **17-scenario
agentic tournament** (tool loops, long-context extraction, multi-step reasoning; 3 repeats,
temperature 0.2), run on the GX10 with one variable changed at a time, on the same day. A
change that only buys tok/s or TTFT and costs even a point there ships as an **option**, off
by default, with its measured cost next to it. Two runs of the same configuration differ by
up to 2 points day to day, so anything inside that band is treated as equal and the faster
one wins; anything below it stays an option.

| what | default | measured effect (GX10, hybrid, MTP=2, prefix caching, YaRN 500k) |
|---|---|---|
| Hybrid checkpoint (`MODE=hybrid`) | recommended, `nvfp4` as published is the default | +20% decode, +8% KV, same tournament score |
| Deterministic top-k kernel (`DET_TOPK=1`) | **on** | identical greedy outputs, full prefill speed, tournament neutral (44/51) |
| Reduced draft vocabulary (`DRAFT_VOCAB=1`) | **on** | +20% decode, tournament 45/51 (the best run), outputs unchanged by construction |
| `MADV_RANDOM` on the table (`MADVISE=random`) | **on** | cold prefill −4–8%, cleaner page cache, tournament neutral |
| Prefix caching (`PREFIX_CACHE=1`) | **on** | ~14 s → ~1.4 s TTFT on a repeated 20k prefix |
| `MTP=3` | option (`MTP=2` default) | +7% decode, −1 point at the tournament (44 vs 45/51) |
| NVFP4 MTP draft experts (`MODE=hybrid-mtp`) | option | +22% KV pool, −3.9 GiB weights, decode unchanged here, tournament neutral (44/51) |
| fp8 KV cache (`KV_DTYPE=fp8_e4m3`) | option | ×1.9 KV pool, 1M context; −10% decode, −30% prefill, one scenario lost |
| M%4 GEMM padding (`PAD_M4=1`) | option | no-op with prefix caching on; −40% TTFT at 8k with it off |
| Exact `torch.topk` (`EXACT_TOPK=1`) | fallback | deterministic like the kernel, −20–40% long prefill |
| Persistent compile cache (`COMPILE_CACHE`) | option | −80 s ± 2 s of init engine per boot after the first; startup only, outputs and tournament unaffected |
| `--long-prefill-token-threshold` (via `EXTRA`) | option | keeps decoding clients responsive under concurrent prefills, at a TTFT cost |

If your priority is raw throughput rather than the agent's reliability, the fast profile is
`MODE=hybrid MTP=3` (41 tok/s in the tournament against 38.5 for the default), and
`MODE=hybrid-mtp` on top if you need the KV pool more than the last few percent of quality.

## The `flash` command

`./flash` is the short path. It does not replace the scripts; it calls them with a named set of
variables, checks what is already done, and tells you what is missing.

| command | what it does |
|---|---|
| `./flash doctor [profile]` | checks arm64/GB10, the 128 GB pool and how much of it is free right now (vLLM needs `GPU_MEM` × total *free* to boot), docker + nvidia runtime, other running containers, the port, the image and its base label, the checkpoint, the prepared layouts, disk space for what is still to download, and the profile itself (YaRN vs context, fp8 KV on the right base) |
| `./flash setup [profile]` | build the image the profile expects (`Dockerfile` or `Dockerfile.v0.29`), download the weights, prepare the hybrid layout and the MTP graft — each step skipped when already done, so re-running it is free |
| `./flash serve [profile] [KEY=VALUE…]` | loads the profile, applies your overrides, refuses early if something is missing, then `exec`s `scripts/serve.sh` |
| `./flash wait` | polls the container and the API, shows the loading stage, prints the KV pool when up |
| `./flash test` | `scripts/smoke-test.sh` against the running server |
| `./flash status` / `logs` / `stop` / `start` / `rm` | the container's state, KV pool, active patches, running requests and prefix-cache hit rate; follow the log; stop (kept, `start` reloads in 8-13 min); remove |
| `./flash profiles` | the list below |

Profiles (`profiles/*.env`, each a handful of `serve.sh` variables; copy one to make your own):

| profile | recipe | when |
|---|---|---|
| `default` | hybrid, YaRN 500k, deterministic top-k, reduced draft vocabulary, prefix caching, MTP=2 | the recommended one: best tournament score (45/51) |
| `speed` | default + `MTP=3` | +7% decode for about one tournament point |
| `context` | `MODE=hybrid-mtp` (NVFP4 MTP draft experts) | +22% KV pool for concurrency or long contexts, decode unchanged |
| `context-1m` | hybrid + `KV_DTYPE=fp8_e4m3`, 1M context | when you need 1M tokens in one request (speed and some quality cost; preview base only) |
| `shared` | default + `--long-prefill-token-threshold 1024` | several clients at once: decoding stays responsive while others prefill, single-stream TTFT −17–36% |
| `published` | `MODE=nvfp4`, YaRN 500k | the checkpoint exactly as published, nothing to prepare; ~26 tok/s |
| `native` | hybrid, 262k, no YaRN | if you never go past the native context |
| `v0.29` | default recipe on the vLLM v0.29.0 release image | to be on the release line (`Dockerfile.v0.29`, measured at parity) |

Precedence: a variable already in your environment beats the profile (`PORT=18301 ./flash serve`
works like the plain scripts), and `KEY=VALUE` arguments beat both. Container name and port default
to `qwen38-flash` and `18300`, like `serve.sh`.

## Requirements

- An **NVIDIA DGX Spark or compatible GB10 (sm_121)** box, 128 GB unified memory,
  aarch64, recent NVIDIA driver, Docker with the NVIDIA container runtime.
- **~140 GB free disk** for the checkpoint (+13 GB for the hybrid variant, +5 GB more
  for the NVFP4-MTP graft), on
  reasonably fast storage (the table is read from it at runtime — NVMe strongly
  recommended; the Spark's onboard NVMe is ideal).
- The base image is multi-arch, so `docker build` also works on x86 Blackwell
  (sm_120, e.g. RTX PRO 6000) for testing, though this is tuned for the Spark.

## Quickstart

The commands are in the [TL;DR](#tldr--run-it-on-a-dgx-spark) at the top. Once the log says
`Application startup complete`, hit the OpenAI-compatible API:

```bash
curl http://localhost:18300/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-flash-next",
  "messages": [{"role":"user","content":"Write a haiku about a desktop supercomputer."}],
  "max_tokens": 512
}'
```

`MODE=nvfp4 scripts/serve.sh` (the default) serves the checkpoint as published at the native
262k context; `YARN=1 CTX=500000` goes to 500k (validated with a needle-in-a-haystack at 414k
tokens); `GPU_MEM=0.80` is the long-running-service setting, see [Tuning](#tuning-env-vars-for-scriptsservesh).

## Two checkpoint modes: NVFP4 or hybrid

`scripts/serve.sh` serves one of two layouts of the same RadixArk NVFP4 checkpoint;
pick with `MODE=`.

| | `MODE=nvfp4` (default) | `MODE=hybrid` |
|---|---|---|
| Routed experts (the bulk, ~63 GiB) | NVFP4 | NVFP4 (unchanged) |
| GDN in/out projections, QSA q/k/v/o, shared experts (~15 GiB) | bf16, as published | **blockwise fp8-e4m3** (128×128 blocks, DeepSeek layout) |
| Extra preparation | none | `scripts/prepare-hybrid.sh` once (~10 min, +13 GB disk) |
| Decode (MTP=2, greedy, real answers) | ~26 tok/s | **~31 tok/s (+20%)** |
| Prefill | same | same (±5%) |
| KV cache | ~580k tokens | **~630k tokens (+8%)**, weights ~7 GiB smaller |
| Tournament quality (17 agentic scenarios × 3) | 45/51 | 45/51 |
| Deterministic at T=0 | yes | yes |
| Behavioural difference we noticed | — | slightly "more careful" in tool loops: it sometimes checks state first (one extra tool call), which is the only place it scored differently before we raised the turn budget |

Why it works: those side layers are dense and read in full on every decoded token, so
they dominate decode bandwidth; the experts are sparse (10 of 512 active) and already
4-bit. Halving the dense part is where the tokens/s come from. The MoE path — where
the quality lives — is untouched. Conversion uses
[@Saren-Arterius](https://github.com/Saren-Arterius)'s `fp8_convert.py` (worst
per-tensor max relative error 3.5%), and a small dispatch shim
(`src/vllm_fp8_hybrid_modelopt.py`) that routes those layers to vLLM's blockwise-fp8
GEMM while the ModelOpt NVFP4 config keeps handling the experts.

```bash
scripts/prepare-hybrid.sh                 # builds <snapshot>-fp8hybrid/ next to the HF snapshot
MODE=hybrid YARN=1 CTX=500000 GPU_MEM=0.80 scripts/serve.sh
```

Our own box runs the hybrid. If you want the checkpoint exactly as published, stay on
`MODE=nvfp4` — you lose ~5 tok/s and nothing else.

### NVFP4 MTP draft experts (`MODE=hybrid-mtp`)

A graft on top of the hybrid: the MTP draft head's routed experts (BF16 fused, ~4.7 GiB)
are replaced by the **NVFP4** draft experts from
[Inferact/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/Inferact/Qwen3.8-Flash-Next-NVFP4)
(1.4 GiB) — the combination neither parent ships: fp8 PLE table *and* a cheap draft.
`scripts/prepare-mtp-graft.sh` builds it in ~5 min (~3.3 GB of real bytes: one shard
rewritten without the 2 fused BF16 MTP tensors, a symlink to the donor shard, and the
index + both quantization exclusion lists fixed — the blanket `mtp.*` globs are replaced
by the donor's 29 explicit non-expert MTP module names in **both** `config.json` and
`hf_quant_config.json`, or the draft loads unquantized and dies). Both parents are
modelopt quantizations of the same base — verified by hashing `embed_tokens.weight` and
all 29 shared draft tensors byte-for-byte before grafting. The graft reads through both
parents (it is a directory of symlinks); don't delete either one.

Measured on the GX10 (hybrid, MTP=2, greedy, real answers):

| | `MODE=hybrid` | `MODE=hybrid-mtp` |
|---|---|---|
| Weights on card | 77.83 GiB | **74.75 GiB** (−3.1) |
| KV cache @0.80 | 625,669 tok | **734,292 tok (+17%)** |
| Max concurrency @262k | 2.39x | **2.80x** |
| Decode, 5-prompt greedy probe | 26.6–33.8 tok/s | **34.7–42.5 tok/s (+20–27%)** |
| Mean acceptance length | ~2.73 | ~2.58 (same band) |
| Tournament quality | 45/51 | same target model, byte-identical greedy outputs (see below) |

Those are the author's numbers. Ours, measured the way defaults are decided here (same
box, same day as the table in [Reduced draft vocabulary](#reduced-draft-vocabulary-draft_vocab1-default),
full-vocabulary draft on both arms):

| | `MODE=hybrid` | `MODE=hybrid-mtp` |
|---|---|---|
| Tournament (17 scenarios × 3) | 44.5/51 | **44/51** (same failures, within the day's noise) |
| tok/s in the tournament | 33.5 | 33.0 |
| Decode, single stream (bench) | 33.1 tok/s | 32.6 tok/s |
| MTP acceptance | 75% (mean length 2.50) | 63.5% (2.27) |
| KV pool @0.80, YaRN 500k | 626k tok | **764k tok (+22%)** |
| Weights on card | 77.8 GiB | **73.9 GiB** |
| Deterministic 4/4, smoke-test (cache hit + logprobs) | yes | yes |

So on this box the graft is a **memory** win, not a speed win: the NVFP4 drafter reads ~4×
fewer bytes per draft step but its proposals are accepted less often, and the two cancel.
It is quality-neutral, which is why it ships as an option rather than the default; take it
when the KV pool or concurrency matters more than the last percent. Not yet measured in
combination with the reduced draft vocabulary.

The draft swap is gated on **greedy output equivalence**: every emitted token is the
target model's argmax (the draft only decides how many of its proposals get accepted per
step), so `scripts/greedy-probe.sh <label>` run against both arms must produce
byte-identical text. It does — 5/5 prompts, 400 tokens each, first-token logprobs
identical to 4 decimals. Donor revision is pinned (`103a7608…`, sha256 `0d44e6d7…`); a
different donor revision needs re-gating.

```bash
scripts/prepare-mtp-graft.sh              # one-time, after prepare-hybrid.sh
MODE=hybrid-mtp scripts/serve.sh
```

## Prefix caching now works (and why it didn't)

`--enable-prefix-caching` used to crash this model on GB10 (`CUDA illegal memory
access`) and, with a bounds guard in place, to **silently return different answers on
cache hits**. We traced it (details in [docs/HOW-IT-WORKS.md](docs/HOW-IT-WORKS.md)):
vLLM's engine core overwrites `cache_config.block_size` with the *smallest* KV-group
block size — 8 tokens here with MTP=2 (4 without), the QSA raw-key ring — while the Mamba
state block is 1600 tokens. Two places used the former as the latter, so on a prefix hit
the worker computed the state slot as `(3200-1)//8 = 399` instead of `1`, read past the
block table row, and restored an **all-zero Mamba state**. The image carries a two-line fix;
with it, cold and cache-hit outputs are bit-identical (state checksums and first-token
logprobs match exactly) and the tournament score is unchanged.

What you get: multi-turn chats, agent/tool loops and shared system prompts skip the
prefill of everything already seen. On a 20k-token prefix, TTFT goes from ~14 s to
~1.4 s; with 8 concurrent conversations, from ~80 s to ~4–6 s. `PREFIX_CACHE=1` is the
default. Mamba states are cached at 1600-token boundaries, so the tail of a prefix is
recomputed — expect the benefit to start around a couple of thousand tokens.

## Deterministic top-k (`DET_TOPK=1`, default)

The sparse attention (QSA) picks the top-k key blocks per query with a `persistent_topk`
kernel. On GB10 that kernel is **non-deterministic** — identical greedy requests produce
different outputs 2 times out of 4 — and can drop legitimate candidates
([vllm#51782](https://github.com/vllm-project/vllm/issues/51782)); reported against
this repo by [@k3dani](https://github.com/k3dani) in
[issue #3](https://github.com/blazux/qwen3.8-Flash-DGX/issues/3). The GB10 is more
exposed than other GPUs: the cooperative kernel used elsewhere for decode is disabled on
sm_12x, so this kernel runs for both prefill and decode.

Two fixes are in the image; the second is the default:

- **`DET_TOPK=1` (default) — deterministic kernel.** [@jschmied](https://github.com/jschmied)
  rewrote `persistent_topk` so that output slots are index-ordered and exact ties are resolved
  without candidate buffers (no truncation, exact pivot) — upstream as
  [vllm#55122](https://github.com/vllm-project/vllm/pull/55122). The Dockerfile compiles it
  with the image's `nvcc` as a standalone extension (`_C_det.so`, ~15 s on a GX10, no vLLM
  rebuild) from his repo at a pinned commit, and an env-gated switch routes the QSA block
  selection to it. Measured on the GX10 (hybrid, MTP=2, prefix caching) with the 2026-09-07 pin
  (PR #10 by @jschmied: signed-zero canonicalisation, a deterministic low-shared-memory path, a
  launcher shared-memory bug fixed, and a faster kernel — 1.0–2.4× the stock kernel's time per call
  instead of 1.8–3.8×): **4/4 prompts stable, first-token logprobs identical to the 4th decimal**,
  decode 31.8 tok/s, prefill 2,488 tok/s at 8k / 2,996 at 32k, needle 92k in 46 s — i.e. the same
  as the stock non-deterministic kernel (the previous pin measured 32.5 / 2,436 / 2,904 / 48 s).
- **`EXACT_TOPK=1` — exact `torch.topk` fallback.** Our first fix: also deterministic and same
  tournament score, but −8% prefill at 8k and −20–40% at 32k+ (decode unchanged). Kept as a
  fallback (it wins over `DET_TOPK` when set), e.g. on a GPU where the kernel is not built.
- `DET_TOPK=0 EXACT_TOPK=0` gives the stock kernel back.

Once vllm#55122 is merged into the release branch this image is built from, patch 8 becomes
redundant. (Masking the never-written logits columns before the stock kernel does **not**
restore determinism, so it is the kernel itself.)

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

## Optional: persistent compile cache (`COMPILE_CACHE`)

`scripts/serve.sh` recreates the container on every run (`docker rm -f`, then `docker run`), so
vLLM's compiled graphs — written to `/root/.cache/vllm` inside the container — are discarded and
rebuilt on every boot. If you keep one container and cycle it with `./flash stop` / `./flash start`
this costs nothing, which is why it went unnoticed. It costs you when something else recreates the
container for you: a proxy that loads and evicts models on demand (llama-swap and friends), CI, or
a tournament run that calls `serve.sh` between configurations.

`COMPILE_CACHE=<name>` mounts two docker volumes (`<name>-vllm`, `<name>-flashinfer`) over the two
cache directories. `COMPILE_CACHE=/some/path` binds `/some/path/vllm` and `/some/path/flashinfer`
instead, to put them on a chosen disk. Unset — the default — is exactly the behaviour above.

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

## Reduced draft vocabulary (`DRAFT_VOCAB=1`, default)

vLLM shares the target model's `lm_head` with the MTP draft, so every draft step scores all
248,320 vocabulary rows: a 1.27 GiB bf16 read per drafted token, on a decode step that is
memory-bandwidth bound. With `DRAFT_VOCAB=1` the drafter scores a private 65,536-row slice of
the head (+320 MiB of memory) and every other token gets −∞, so the proposer's argmax/sampling
code is untouched. The target still verifies every drafted token: **outputs are identical to
full-vocabulary drafting**; only the acceptance rate can move, down, when the target wants a
token outside the set (about 6% of the tokens of a French text, 75% → 68% acceptance here).

The 65,536 ids are the most frequent tokens of a small local corpus, then BPE merge order as a
frequency proxy, plus every special and added token (chat template, tool-call and thinking
markers, byte fallbacks). `tools/build_draft_vocab.py` rebuilds the set for another language
mix; `DRAFT_VOCAB=/path/ids.npy` uses your own, `DRAFT_VOCAB=0` disables.

Measured on the GX10 the way defaults are decided here — the 17-scenario agentic tournament,
3 repeats, temperature 0.2, one variable per run, all on the same day (hybrid, MTP=2, prefix
caching, YaRN 500k):

| configuration | tournament | tok/s in the tournament |
|---|---|---|
| hybrid, exact `torch.topk` (the previous default) | 43/51 (84.3%) | 31.4 |
| + deterministic kernel (PR #10) | 44/51 (86.3%) | 31.9 |
| + `MADV_RANDOM` on the table | 44.5/51 (87.3%) | 33.5 |
| **+ reduced draft vocabulary, MTP=2 (the new default)** | **45/51 (88.2%)** | **38.5** |
| same, MTP=3 | 44/51 (86.3%) | 41.2 |

Two runs of the same configuration on different days differ by up to 2 points, so the four
first rows are equivalent in quality; the last one shows why MTP=3 stays an option. Single-stream
`bench` numbers for the default: decode 36.6 tok/s, prefill 2,473 tok/s at 8k / 3,004 at 32k,
needle 92k in 45.5 s, 4/4 deterministic. The KV pool loses the 320 MiB slice plus the
full-width logits buffer the patch rebuilds per draft step (about 60k tokens at `GPU_MEM=0.80`).

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
Full port notes in [docs/HOW-IT-WORKS.md](docs/HOW-IT-WORKS.md#the-vllm-v0290-port-dockerfilev029).

## Tuning (env vars for `scripts/serve.sh`)

| Var | Default | Notes |
|---|---|---|
| `MODE` | `nvfp4` | `hybrid` = fp8 side layers (see above; needs `scripts/prepare-hybrid.sh`). |
| `PREFIX_CACHE` | `1` | `--enable-prefix-caching`. Correct with this image (block_size fix). |
| `DET_TOPK` | `1` | Deterministic QSA top-k **kernel** (vllm#55122): identical outputs at T=0 at full kernel speed. `0` = stock kernel (non-deterministic, may drop attention candidates, issue #3). |
| `EXACT_TOPK` | `0` | `1` = exact `torch.topk` fallback (deterministic; −8% prefill at 8k, −20–40% at 32k+). Wins over `DET_TOPK` when set. |
| `DRAFT_VOCAB` | `1` | MTP drafter scores only the 65,536 most frequent tokens (+20% decode, same tournament score, outputs unchanged). `0` = full vocabulary; a path = your own ids (`tools/build_draft_vocab.py`). |
| `MADVISE` | `random` | `madvise` on the mmapped PLE table: `random` (no readahead: cold prefill −4–8%, cleaner page cache) or `normal`. |
| `PAD_M4` | `0` | `1` = pad M%4 in the blockwise-fp8 GEMM (hybrid mode). No-op with `PREFIX_CACHE=1`; about −40% TTFT at 8k with `PREFIX_CACHE=0`. |
| `PORT` | `18300` | API port |
| `CTX` | `262144` | Max context. Native is 262144; with `YARN=1` up to `500000` is validated. |
| `YARN` | `0` | `1` = YaRN rope scaling (factor 4, Qwen's recipe) for `CTX` > 262144. |
| `SEQS` | `8` | Max concurrent sequences. **Do not benchmark with 1–2**: excess requests queue silently and aggregate tok/s flatlines (see below). |
| `GPU_MEM` | `0.80` | Fraction of the 128 GB pool for weights+KV. `0.85` buys ~2 GiB more KV, but after a day at `0.85` the box drifted into swap, and `0.875` got OOM-killed on a 300k-token prefill with MTP. The lower you set it, the more RAM the page cache has for the 48 GiB table — which is what your prefill speed depends on (below). Right after stopping another big container the first boot can fail with "13.5 GiB KV cache is needed, larger than available" — memory not yet released; the `unless-stopped` retry succeeds. |
| `MTP` | `2` | Speculative tokens from the model's MTP head (`0` = off). `3` is +7% decode but cost a point at the tournament (44 vs 45/51), so it stays an option. |
| `KV_DTYPE` | `auto` | `auto` = bf16 (recommended). `fp8_e4m3` = ~1.9× KV pool, 1M context on one box, at −10% decode / −30% prefill and a measurable quality cost — see [fp8 KV cache](docs/HOW-IT-WORKS.md#fp8-kv-cache-on-the-qsa-path-opt-in) before using it. |
| `PREWARM` | `0` | `1` streams the 48 GiB table once at boot to warm the page cache — steadier first-request latency, ~10 s extra startup. |
| `WORKERS` | `32` | Threads used for the mmap gather (only used above `VLLM_PLE_MMAP_FAST_ROWS`=512 unique rows; decode-sized gathers run inline). |
| `COMPILE_CACHE` | | Keep vLLM's compiled graphs across boots — this script recreates the container every run, so by default they are rebuilt each time. `<name>` = two docker volumes, `/abs/path` = two bind mounts. **−80 s ± 2 s of init engine** per boot after the first, 169 MB of disk; only worth setting if something recreates the container for you (a model-swapping proxy, CI, tournament runs). See [above](#optional-persistent-compile-cache-compile_cache). |
| `LOG_REQUESTS` | `0` | `1` logs every prompt and output (`VLLM_LOGGING_LEVEL=DEBUG --enable-log-requests --enable-log-outputs`) so `tools/vllm_watch.py` can show sessions live. Debugging only: it puts user content in the Docker log, unbounded. |
| `EXTRA` | | Extra vLLM flags, passed verbatim — e.g. `--long-prefill-token-threshold 1024` for multi-client responsiveness (see [the concurrency section](#decoding-clients-stall-while-other-clients-prefill-the-long-prefill-token-threshold-slider)), `--api-key <secret>`. |

## Throughput and concurrency

Single-stream numbers understate this model on a GB10. @jschmied traced one box
under load (RadixArk NVFP4, 8k ctx, **no** speculative decoding, using vLLM's native
PLE CPU offload rather than this repo's mmap — the table-serving cost behaves the
same way) and found aggregate throughput scales far past single-stream:

| concurrent streams | aggregate tok/s | per stream | major faults / token | TTFT |
|---:|---:|---:|---:|---:|
| 1 | 17.1 | 17.1 | 16.0 | 0.22 s |
| 8 | 87.5 | 10.9 | 7.0 | 0.53 s |
| 16 | 131.6 | 8.2 | 9.6 | 0.83 s |
| 32 | 212.0 | 6.6 | 4.3 | 1.19 s |
| 48 | **266.8** | 5.6 | 3.6 | 1.60 s |

Two things worth knowing (their words, lightly condensed):

- **The paged table is an argument *for* concurrency, not against it.** Page-fault cost
  per token *falls* 4.4× from c=1 to c=48: batched tokens share n-gram rows and the
  page cache keeps the hot set, so the marginal token is far cheaper than the first.
  The table gather itself never exceeded ~25% of one CPU core.
- **A low `--max-num-seqs` is indistinguishable from saturation if you only look at
  tok/s.** With `--max-num-seqs 2` their sweep flatlined at ~33 tok/s while
  `vllm:request_queue_time_seconds_sum` climbed to 142 s. Check `max-num-seqs` before
  quoting an aggregate number — this repo's default is now `8` for that reason.

Method and harness: [load-and-waits.md](https://github.com/jschmied/qwen38-flash-next-gb10/blob/main/notes/load-and-waits.md).

### Decoding clients stall while other clients prefill (the `long-prefill-token-threshold` slider)

Reported by [@kutovoy](https://github.com/kutovoy) in
[issue #9](https://github.com/blazux/qwen3.8-Flash-DGX/issues/9): with two or more agents on
the box, a client that is decoding drops from 30 tok/s to 0.1–0.5 tok/s for a minute or two,
then recovers. Reproduced here in minutes and it is not a bug in the recipe: vLLM runs one
step at a time, and with chunked prefill every step that carries a prefill chunk also carries
exactly one token for each decoding request. On this model a chunk of 8,192 tokens
(`--max-num-batched-tokens 8192`, the default) takes ~3.5 s of compute at ~2,400 tok/s, plus
the n-gram lookups of a prompt the page cache has never seen (0.3–0.4 s per chunk here, more
on a box that is short on cache or swapping). So while any new prompt is being prefilled, a
decoding client gets one token per step, i.e. one every 4–7 s. Prefix caching does not help:
the prompts are new.

The knob is `--long-prefill-token-threshold N` (per-request chunk cap; the total step budget
stays at 8,192 for the decoders), passed through `EXTRA=`. Measured on the GX10 (hybrid,
MTP=2, prefix caching): one client decoding, then two other clients sending cold ~72k-token
prompts 10 s later.

| `--long-prefill-token-threshold` | decoding client during the two prefills | p95 / max gap between its tokens | TTFT of each 72k prompt | single-stream prefill 8k / 32k | needle 92k |
|---|---|---|---|---|---|
| none (8,192 chunks, default) | **0.2 tok/s** | 5.5 s / 7.1 s | 66 s / 110 s | 2,493 / 2,994 tok/s | 46.6 s |
| 2048 | 0.4–0.6 tok/s | 1.9 s / 3.0 s | 89 s / 89 s | not measured | — |
| 1024 | 1.0 tok/s | 1.3 s / 2.0 s | 98 s / 98 s | 1,597 / 2,497 tok/s (−36% / −17%) | 64.8 s |
| 512 | 1.6–1.8 tok/s | 0.85 s / 1.5 s | 112 s / 112 s | 1,268 / 2,044 tok/s (−49% / −32%) | 84.8 s |

Read it as a slider, not a fix: on one GPU, keeping a decoding client at X tok/s while
others prefill means at most ~2,400 / X prefill tokens per step, and every step below 8,192
tokens costs single-stream prefill (a 1,024-token cap already takes 36% off an 8k TTFT). A step
still holds one chunk of *each* running prefill, which is why 512 does not reach 5 tok/s. The
smaller chunks do have one unambiguous benefit: peak swap-out during the prefills fell from
90–120 MB/s to 7–60 MB/s, because the activation peak per step shrinks.

- Single main user, occasional second client (our case): keep the default. Long prompts land
  fast; the rare overlap costs the other client a slow minute.
- Several interactive agents that must stay responsive: `EXTRA='--long-prefill-token-threshold 1024'`
  (or `512` if TTFT matters less than never stalling). Confirmed in the field by
  [@techfury90](https://github.com/techfury90) with 2–6 parallel agents. Keep swap small
  (`vm.swappiness=10` — the Spark default is 60; a 134 GB swap file lets the kernel page vLLM
  itself out instead of dropping cache, and once it is swapped every step page-faults) and use
  `PREWARM=1`. Lower `GPU_MEM` (0.75) only if the box is actually swapping: with several
  long-context agents the KV pool matters more than page cache for the table. `SEQS=4` limits
  how many prefills can interleave.
- The structural way out is a second Spark: the four ConnectX-7 ports exist for that, and
  vLLM's prefill/decode disaggregation puts the prefills on the other box.

## How it fits — the one idea

A token's n-gram lookup reads **16 rows × 160 bytes ≈ 2.5 KB**. Over a 20k-token
prefill that's ~1.3 GB of small reads — under a second on NVMe, and the hot n-grams
stay in the page cache. So the 48 GiB table never needs to be in the unified pool:
we `mmap` the checkpoint's `model-plefp8-*.safetensors` shards and gather rows on
demand. Nothing else about the model changes — the hashing, dequant, and the sparse
attention all run stock.

Full details, including the GB10-specific bugs this works around and the long-context
findings, are in [docs/HOW-IT-WORKS.md](docs/HOW-IT-WORKS.md).

### GB10 kernel fixes and the faster gather (contributed)

From [@Saren-Arterius](https://github.com/Saren-Arterius)'s fork, merged here with thanks:

- **FLA shared-memory gate** — sm_121 reports 99 KiB of shared memory per block but the
  flash-linear-attention gate asked for 100 KiB, so all 36 GDN layers silently ran on
  small tiles. One `sed` in the Dockerfile lowers the gate to 99 KiB.
- **`chunk_delta_h` `num_warps` pin** — works around a `tl.dot` race on Blackwell
  ([fla#953](https://github.com/fla-org/flash-linear-attention/issues/953)). Correctness, not speed.
- **PLE gather hot path** — CPU dedup of row ids, a persistent pinned staging buffer with an
  async H2D copy, GPU-side expansion through the inverse index, and an inline fast path
  for decode-sized batches (`VLLM_PLE_MMAP_FAST_ROWS`, default 512; larger gathers are split
  into `VLLM_PLE_MMAP_CHUNK`=2048-row tasks across `WORKERS` threads). Also: bf16/f16 tables,
  `VLLM_PLE_MMAP_DIR` to serve the table from another directory, and a periodic
  `PLE mmap stats` log line (`VLLM_PLE_MMAP_STATS_SEC`, default 30).
- **Mamba state-copy guard** — with [vllm#50729](https://github.com/vllm-project/vllm/pull/50729)
  (the overlapping-copy race fix by @AndreasKaratzas), a bounds check that turns an
  out-of-range block id into a skipped copy plus a log counter instead of a dead CUDA
  context. With the block_size fix above the counter stays at 0; if you ever see
  `mamba state-copy guard: N out-of-range`, something upstream regressed — please report it.
- **`fp8_convert.py`** — the side-layer conversion behind `MODE=hybrid`.

Their fork goes further with an **int4 (Intel AutoRound) + fp8 hybrid** checkpoint:
[qwen3.8-Flash-DGX-AutoRound](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound).
We benchmarked it with the same patches: ~34 tok/s decode, 44/51 on the tournament,
but it is not deterministic even with the exact top-k and Marlin's atomic adds off, and
it has the slowest prefill of the three — we kept the NVFP4-based layouts.

### Alternative: vLLM's native PLE CPU offload

vLLM ships its own path (`VLLM_PLE_CPU_OFFLOAD=1`) that keeps the table in pinned host
RAM in a separate worker process. On a Spark that RAM is the same pool as the GPU, so
it saves less than the mmap — but @jschmied got it running and documented two things
you will need if you go that way (neither applies to the mmap patch, which is a single
process):

1. `_get_ple_embedding_quant_method()` in `ple_layer.py` only accepts `Fp8Config`;
   with the NVFP4 checkpoint the quant config is `modelopt_fp4`, so the FP8 PLE shards
   are rejected and loading dies on `ngram_embedding.weight_scale`. Accepting
   `modelopt`/`modelopt_fp4` there fixes it.
2. The worker hands CUDA tensors to the GPU process over IPC via `pidfd_getfd`, which
   `kernel.yama.ptrace_scope=1` (the Ubuntu/DGX OS default) forbids between sibling
   processes. In Docker: `--cap-add=SYS_PTRACE`. Under systemd:
   `AmbientCapabilities=CAP_SYS_PTRACE`. It fails ~10 minutes in, after all shards
   have loaded, with an unhelpful `Engine core initialization failed`.

Details: [results-radixark-vllm.md](https://github.com/jschmied/qwen38-flash-next-gb10/blob/main/notes/results-radixark-vllm.md).

## What's in here

```
flash                             one-command front-end: doctor / setup / serve <profile> / wait / test / status …
profiles/*.env                    named recipes for it (default, speed, context, context-1m, shared, published, native, v0.29)
Dockerfile                        official vLLM Flash-Next preview image + the patches below (default)
Dockerfile.v0.29                  same recipe on the vLLM v0.29.0 release (patches 3 and 9 dropped, 7 not ported)
src/vllm_ple_mmap.py              1. mmap PLE table (opaque splitting op)            VLLM_PLE_MMAP=1
                                     handles both layouts (preview forward_impl hook / v0.29 embedding swap)
src/mamba_utils_guarded.py        3. vllm#50729 + bounds guard (drop-in mamba_utils.py)
src/patch_mamba_block_size.py     4. prefix-caching block_size fix
src/patch_qsa_exact_topk.py       5. exact, deterministic QSA top-k                  VLLM_QSA_EXACT_TOPK=1
(Dockerfile patch 8)              8. deterministic persistent_topk kernel, built at docker build  VLLM_QSA_DET_TOPK=1
                                     from @jschmied's repo (pinned commit) — vllm#55122
(Dockerfile patch 9)              9. M%4 padding for the blockwise-fp8 GEMM (@jschmied,      VLLM_FP8_PAD_M4=1
                                     pinned commit) — hybrid mode with prefix caching off
src/patch_mtp_draft_vocab.py     10. reduced draft vocabulary for the MTP drafter          VLLM_MTP_DRAFT_VOCAB=<ids.npy>
src/draft_vocab_65536.npy            the default 65,536-id set (tools/build_draft_vocab.py rebuilds it)
src/vllm_fp8_hybrid_modelopt.py   6. NVFP4 experts + fp8 side layers dispatch        VLLM_FP8_HYBRID=1
src/patch_qsa_fp8_kv.py           7. fp8_e4m3 KV cache on the QSA path (by @Nanetnounou) --kv-cache-dtype fp8_e4m3
src/test_ple_mmap_cpu.py          CPU unit test for the gather (no GPU needed)
src/test_qsa_exact_topk_cpu.py    CPU unit test for the exact top-k (no GPU needed)
tools/fp8_convert.py              side-layer bf16 -> blockwise fp8 (by @Saren-Arterius)
scripts/download-weights.sh
scripts/prepare-hybrid.sh         one-time: build the -fp8hybrid snapshot
scripts/prepare-mtp-graft.sh      one-time: graft the NVFP4 MTP draft experts onto it (MODE=hybrid-mtp)
tools/vllm_watch.py               live per-session view of prompts / reasoning / outputs / stats (needs LOG_REQUESTS=1; @0x3dlux)
scripts/serve.sh                  MODE=nvfp4|hybrid|hybrid-mtp, PREFIX_CACHE, DET_TOPK, DRAFT_VOCAB, MADVISE, EXACT_TOPK, PAD_M4, KV_DTYPE, YARN, ...
scripts/smoke-test.sh             health, coherence, prefix-cache hit, determinism, tok/s
scripts/greedy-probe.sh           greedy probe set; diff two arms to gate a draft/checkpoint swap
docs/HOW-IT-WORKS.md
```

Run the unit tests (no GPU):

```bash
docker run --rm -v "$PWD/src:/t" -w /t --entrypoint python3 qwen38-flash-dgx test_ple_mmap_cpu.py
docker run --rm -v "$PWD/src:/t" -w /t --entrypoint python3 qwen38-flash-dgx test_qsa_exact_topk_cpu.py
```

## Limitations & notes

- **One big model at a time.** At `GPU_MEM=0.80` this uses most of the 128 GB pool;
  don't co-locate another large model (an 8B embedding model next to it already
  starves the KV cache — we moved ours to another machine).
- **Full `torch.compile` is off on the preview image** (an Inductor int64-indexing assert
  on sm_121; gone on the v0.29 base); on both, the serve script uses PIECEWISE CUDA graphs
  with the PLE lookup as a splitting op.
- **1M context** needs the fp8 KV cache (`KV_DTYPE=fp8_e4m3`, see above; preview image
  only, not ported to the v0.29 base yet), which costs speed and some quality; in bf16 a single 1M request needs ~26 GiB of KV and 500k with
  YaRN is the validated ceiling (800k booted but got OOM-killed on a long prefill).
- **Exact top-k costs prefill** on long prompts (see above). The implementation is a
  plain `torch.topk` over the full visible width per chunk; a fused kernel would recover
  most of it — PRs welcome.
- **No authentication, binds all interfaces.** `scripts/serve.sh` publishes the API on
  `0.0.0.0:$PORT` with no key, like the upstream image. On a shared network put it behind
  your gateway or pass `EXTRA='--api-key <secret>'` (vLLM then requires it as a Bearer token).
- **Weights are not included** and the checkpoint carries Qwen's license (with a
  MAU/revenue clause) — review it before production use.

## Credits

- `tools/vllm_watch.py`, the live session viewer: **[@0x3dlux](https://github.com/0x3dlux)** (issue #12).
- The nudge to port the recipe to the vLLM v0.29.0 release: **[@ChengYen-Tang](https://github.com/ChengYen-Tang)** (issue #14).

- The two ideas behind the reduced draft vocabulary and the `MADV_RANDOM` table advice come
  from **[MiaAI-Lab](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark)**'s
  recipe for their own checkpoint; both are reimplemented here from scratch (their code is
  AGPL-3.0) and measured on this checkpoint.

- Model: **Qwen team, Alibaba** — Qwen3.8-Flash-Next.
- NVFP4 checkpoint: **[RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)**.
- NVFP4 MTP draft experts (the `hybrid-mtp` graft donor): **[Inferact/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/Inferact/Qwen3.8-Flash-Next-NVFP4)**; the graft recipe follows
  [thavoc's graft write-up](https://gist.github.com/thavoc/d7083457f6f2d981f879670c34df34ab)
  and [Peuqui/mtp-quant-transplant](https://github.com/Peuqui/mtp-quant-transplant).
- Serving engine and base image: **vLLM** (`vllm/vllm-openai:qwen38-flash-next`,
  the `release/qwen38next` recipe / PR #53896); the Mamba state-copy race fix is
  [vllm#50729](https://github.com/vllm-project/vllm/pull/50729) by **@AndreasKaratzas**.
- GB10 FLA fixes, the faster PLE gather, the state-copy guard and the fp8 side-layer
  conversion: **[@Saren-Arterius](https://github.com/Saren-Arterius)**
  ([qwen3.8-Flash-DGX-AutoRound](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound)).
- The fp8_e4m3 KV cache patch for the QSA path: **[@Nanetnounou](https://github.com/Nanetnounou)**
  ([issue #6](https://github.com/blazux/qwen3.8-Flash-DGX/issues/6), [vllm#54426](https://github.com/vllm-project/vllm/issues/54426)).
- The non-deterministic `persistent_topk` diagnosis and upstream report:
  **[@k3dani](https://github.com/k3dani)** ([issue #3](https://github.com/blazux/qwen3.8-Flash-DGX/issues/3),
  [vllm#51782](https://github.com/vllm-project/vllm/issues/51782)).
- The deterministic `persistent_topk` kernel (vllm#55122, patch 8), the fp8 GEMM `M % 4`
  finding and its padding drop-in (patch 9), the independent
  reproduction on a DGX Spark, the native-offload fixes and the concurrency measurements:
  **[@jschmied](https://github.com/jschmied)**
  ([issue #1](https://github.com/blazux/qwen3.8-Flash-DGX/issues/1),
  [qwen38-flash-next-gb10](https://github.com/jschmied/qwen38-flash-next-gb10)).
- The mmap-PLE patch, the hybrid dispatch for ModelOpt-NVFP4, the prefix-caching root
  cause and fix, the exact top-k path and the GB10 serving recipe in this repo: see
  [LICENSE](LICENSE) (Apache-2.0).
