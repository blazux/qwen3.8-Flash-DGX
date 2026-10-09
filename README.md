# Qwen3.8-Flash-Next on a single DGX Spark (GB10)

Run **Qwen3.8-Flash-Next** — a ~176B-parameter model (125B main + 51B n-gram, 6B
active) — on **one NVIDIA DGX Spark / ASUS GX10** with **vLLM**, at full prefill
speed, with MTP speculative decoding, **working prefix caching**, **deterministic
greedy decoding**, and up to **500k tokens of context**.

The catch this repo solves: the NVFP4 checkpoint is **~125 GiB**, which does not fit
next to a usable KV cache in the Spark's **128 GB unified pool**. 48 GiB of that is
the n-gram embedding ("PLE") table — a pure lookup that a token only touches 16 rows
of. This repo patches the official vLLM v0.31.0 image to **serve that table from NVMe via
`mmap`** instead of keeping it resident. Weights drop to **~75 GiB**, the rest of the
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
./flash setup       # builds the image, downloads the checkpoint (NVIDIA's NVFP4, 124 GiB via Xet, resumable), prepares the hybrid layout
./flash serve       # the recommended recipe (profile "default"): hybrid, 500k context, deterministic
./flash wait        # first boot loads ~75 GiB of weights, ~3-4 min (patches 14-18); prints the KV pool when the API is up
./flash test        # health, coherence, prefix-cache hit, determinism, tok/s
```

Other recipes are one word away: `./flash profiles` lists them (`speed`, `context`, `context-1m`,
`shared`, `published`, `native`), `./flash serve speed` runs one, and any variable can still
be overridden on the command line (`./flash serve default MTP=3 PORT=18301`). `./flash status`,
`logs`, `stop`, `start`, `rm` do what they say. Details in [The `flash` command](#the-flash-command).

The same thing by hand, unchanged and still supported (everything `flash` does is these scripts):

```bash
docker build -t qwen38-flash-dgx:v0.31 .      # official vLLM v0.31.0 image + the patches below
scripts/download-weights.sh                   # nvidia/Qwen3.8-Flash-Next-NVFP4, ~124 GiB via Xet, resumable (one-time)
scripts/prepare-hybrid.sh                     # recommended: fp8 side layers, +20% decode, same quality (~10 min, one-time)
MODE=hybrid YARN=1 CTX=500000 scripts/serve.sh   # the recommended recipe; 500k context, ~3-4 min to load
docker logs -f qwen38-flash                   # ready at "Application startup complete"
scripts/smoke-test.sh                         # health, coherence, prefix-cache hit, determinism, tok/s
```

OpenAI-compatible API on `http://localhost:18300/v1`, default model name `qwen3.8-flash-next`
(configurable with `SERVED_MODEL_NAME`), tool calling and reasoning parsers on. Every default is the setting that scored best on our
agentic tournament (see [How the defaults are chosen](#how-the-defaults-are-chosen-quality-first-speed-as-an-option));
what you get on a GX10: ~34 tok/s single-stream decode, ~2,700–4,000 tok/s prefill (8k–32k prompts), a ~505–520k-token
KV pool (larger numbers seen before 2026-09-25 were partly swap, see [#34](https://github.com/blazux/qwen3.8-Flash-DGX/pull/34)), prefix caching, deterministic greedy output, 500k tokens of context. The checkpoint is
**NVIDIA's own NVFP4 quantization** since 2026-09-14 (it replaced RadixArk's after a 5-pass head-to-head:
same or better quality, +15–22% KV, −8% single-stream decode — the whole story is in
[Checkpoints](#checkpoints-nvidias-nvfp4-default-and-radixarks)); RadixArk's is one variable away,
`MODEL=RadixArk/Qwen3.8-Flash-Next-NVFP4`, same recipe, same image. Want the checkpoint exactly as
published? Drop `prepare-hybrid.sh` and `MODE=hybrid`. Want speed over the last percent of
quality? `MTP=3`, and `MODE=hybrid-mtp` for more KV — both explained in the [options table](#how-the-defaults-are-chosen-quality-first-speed-as-an-option).
Everything below is the long version: what was broken on GB10, what was fixed, and the numbers.

> **Independently reproduced** on a DGX Spark by
> [@jschmied](https://github.com/jschmied) — see
> [issue #1](https://github.com/blazux/qwen3.8-Flash-DGX/issues/1) and their
> [write-up](https://github.com/jschmied/qwen38-flash-next-gb10).

## Quoted tool markers

The image includes a parser fix for literal or malformed `<tool_call>`
markers in Qwen reasoning and ordinary text. Previously, quoting that marker could
switch the parser into a tool preamble and discard subsequent text, including a
final answer after `</think>`. The parser now buffers the marker and preserves it as text when ordinary prose
follows, while recognizing a function-header prefix (`<function=`) as a tool call.
Valid calls and existing empty-wrapper/end-of-stream handling are retained.
The fix is enabled for the `qwen3` parser configuration, which is what
`--tool-call-parser qwen3_coder` (the flag `scripts/serve.sh` uses) and
`qwen3_xml` both run; derived parser configurations retain their existing behavior.

That fix covers a marker followed by ordinary prose. It does not cover a marker
followed by a well-formed function header — which is exactly what the model writes
when it *documents* the format, inside a ```` ```xml ```` block or while reasoning
about tool syntax. There the parser confirms the call, and everything after it is
consumed as tool-call syntax: with tools in the request you get a tool call the
model never meant to make, and without them the serving layer drops the call and
returns `content: null`, so the whole answer disappears. The visible output stops at
the fence opener, which is why this reads as "output dies on a backtick".

Patch 13 adds a second guard, on the same `qwen3` configuration (so it also covers
`qwen3_coder` and `qwen3_xml`):

- **Inside a fenced code block**, `<tool_call>` and `<function=` stay text and open
  no call. Fences follow CommonMark: a run of three or more backticks or tildes at
  the start of a line (at most three spaces of indent) opens one, and only a run of
  the same character at least as long, with nothing but whitespace after it, closes
  it — so a ```` ```xml ```` block nested inside a ````` ````md ````` block does not
  close the outer one. Fence state is per channel and is re-synced on use, so a
  fence left open in reasoning never carries into the answer, even when the call
  follows `</think>` with nothing in between.
- **A wrapper-less `<function=` header** opens a call only at the start of a line.
  Mid-line it is prose naming the marker. This is the `(CONTENT, FUNC_PREFIX)`
  fallback, which patch 12 does not guard at all.

Not covered, both deliberate:

- Illustrative `<tool_call>` XML written in reasoning *outside* a fence still opens
  a call. Suppressing that would mean dropping the reasoning → tool-call transition,
  which this model does use.
- A fence the model opens and never closes keeps the guard active for the rest of
  the turn, so a real tool call after it is returned as text instead of being
  parsed. That is the cost of deciding from the text alone;
  `test_unclosed_fence_suppresses_later_calls` asserts it so a change is deliberate.

The regression patches extend vLLM's Qwen parser tests. To run them from a vLLM v0.31.0
source checkout with its test dependencies installed (using absolute paths to
this repository's patch files):

```bash
patch --batch --forward --fuzz=0 -p1 < /path/to/qwen3.8-Flash-DGX/src/patches/qwen-tool-preamble.patch
patch --batch --forward --fuzz=0 -p1 < /path/to/qwen3.8-Flash-DGX/src/patches/qwen-tool-preamble-tests.patch
patch --batch --forward --fuzz=0 -p1 < /path/to/qwen3.8-Flash-DGX/src/patches/qwen-tool-marker-guard.patch
patch --batch --forward --fuzz=0 -p1 < /path/to/qwen3.8-Flash-DGX/src/patches/qwen-tool-marker-guard-tests.patch
.venv/bin/python -m pytest tests/parser/engine -q
```

Patch 13's own module, `tests/parser/engine/test_qwen3_literal_markers.py`, is 30
cases over three chunk sizes: 21 fail and 9 pass on patch 12 alone, all 30 pass with
patch 13. The 9 that pass either way are the no-regression guards — a real call
still parses, a real call after a closed fence still parses, and patch 12's own
inline-quoted-marker case is unchanged.

## Update 2026-10-08 — vLLM v0.31 is the current base

The recipe uses the official v0.31.0 image. Existing prepared checkpoints can
be reused without conversion. Build and run:

```bash
./flash setup
./flash serve
./flash wait
./flash test
```

Tested on GX10 with the NVIDIA hybrid checkpoint, MTP=2 and FP8 KV caching:
the agentic tournament, prefix caching, sequential greedy determinism and
long-context retrieval at 499,000 input tokens. See the
[v0.31 validation results](docs/VLLM-031.md#gx10-validation-2026-10-07-to-2026-10-08)
for quality and performance measurements.

## Update 2026-09-28 — vLLM v0.30 became the stable base

- **One base image, one `Dockerfile`.** It builds the recipe on the vLLM v0.30.0 release
  (`qwen38-flash-dgx:v0.30`) and every profile uses it. v0.30 has run our own box since 2026-09-25:
  same tournament score as the v0.29 image before it, cold prefill 1.5–2× faster, ~3½-minute boots,
  and the fp8 KV cache for 1M-token context. The preview and v0.29 images, `Dockerfile.v0.29`, the
  `v0.29`/`v0.30` profiles and `PAD_M4` are gone. They are kept at the git tag
  [`multi-base-final`](https://github.com/blazux/qwen3.8-Flash-DGX/tree/multi-base-final) if you need them.
- **Upgrading:** `git pull && ./flash setup && ./flash serve`. `flash` sees that your image is an older
  build and rebuilds it; the weights and the hybrid layout are reused as they are. If you ran
  `./flash serve v0.30`, it is now plain `./flash serve` (same recipe). By hand:
  `docker build -t qwen38-flash-dgx:v0.30 .` and `scripts/serve.sh` as before.
- **Earlier updates**, including the v0.30 vs v0.29 head-to-head, are in [docs/HISTORY.md](docs/HISTORY.md).

## How the defaults are chosen: quality first, speed as an option

Every default in `scripts/serve.sh` is the setting that scored best on our **agentic
tournament** (tool loops, long-context extraction, multi-step reasoning, hidden-test coding;
17 scenarios when the recipe was built, 55 for the checkpoint decision; 3–5 repeats, temperature
0.2), run on the GX10 with one variable changed at a time, on the same day. A
change that only buys tok/s or TTFT and costs even a point there ships as an **option**, off
by default, with its measured cost next to it. Two runs of the same configuration differ by
up to 2 points day to day, so anything inside that band is treated as equal and the faster
one wins; anything below it stays an option.

| what | default | measured effect (GX10, hybrid, MTP=2, prefix caching, YaRN 500k) |
|---|---|---|
| Checkpoint (`MODEL`) | **`nvidia/Qwen3.8-Flash-Next-NVFP4`** (since 2026-09-14) | vs RadixArk, 5 × 55 scenarios: 88.8% ± 1.0 vs 86.1% ± 1.9, runaway rate equal, needle and determinism equal; +15–22% KV pool, −8% single-stream decode |
| Hybrid checkpoint (`MODE=hybrid`) | recommended, `nvfp4` as published is the default | +20% decode, +8% KV, same tournament score |
| Deterministic top-k kernel (`DET_TOPK=1`) | **on** | identical greedy outputs, full prefill speed, tournament neutral (44/51) |
| Reduced draft vocabulary (`DRAFT_VOCAB=1`) | **on** | +20% decode, tournament 45/51 (the best run), outputs unchanged by construction |
| `MADV_RANDOM` on the table (`MADVISE=random`) | **on** | cold prefill −4–8%, cleaner page cache, tournament neutral |
| Prefix caching (`PREFIX_CACHE=1`) | **on** | ~14 s → ~1.4 s TTFT on a repeated 20k prefix |
| `MTP=3` | option (`MTP=2` default) | +7% decode, −1 point at the tournament (44 vs 45/51) |
| NVFP4 MTP draft experts (`MODE=hybrid-mtp`) | option | +22% KV pool, −3.9 GiB weights, decode unchanged here, tournament neutral (44/51) |
| fp8 KV cache (`KV_DTYPE=fp8_e4m3`) | option | ×1.9 KV pool, 1M context. −4% decode, −3 to −17% prefill, tournament 88.4% (3 runs, 500k context) vs 87.8% in bf16 (3 runs); prefix-cache blocks twice as coarse |
| Exact `torch.topk` (`EXACT_TOPK=1`) | fallback | deterministic like the kernel, −20–40% long prefill |
| Persistent compile cache (`COMPILE_CACHE`) | option | −37 s of startup per boot after the first (init engine −26 s); startup only, outputs and tournament unaffected |
| `--long-prefill-token-threshold` (via `EXTRA`) | option | keeps decoding clients responsive under concurrent prefills, at a TTFT cost |

If your priority is raw throughput rather than the agent's reliability, the fast profile is
`MODE=hybrid MTP=3` (41 tok/s in the tournament against 38.5 for the default), and
`MODE=hybrid-mtp` on top if you need the KV pool more than the last few percent of quality.

## The `flash` command

`./flash` is the short path. It does not replace the scripts; it calls them with a named set of
variables, checks what is already done, and tells you what is missing.

| command | what it does |
|---|---|
| `./flash doctor [profile]` | checks arm64/GB10, the 128 GB pool and how much of it is free right now (vLLM needs `GPU_MEM` × total *free* to boot), docker + nvidia runtime, other running containers, the port, the image and its base label, the checkpoint, the prepared layouts, disk space for what is still to download, and the profile itself (YaRN vs context, context vs KV dtype) |
| `./flash setup [profile]` | build the image (`Dockerfile`; an older build under the same name is rebuilt), download the weights, prepare the hybrid layout and the MTP graft — each step skipped when already done, so re-running it is free |
| `./flash serve [profile] [KEY=VALUE…]` | loads the profile, applies your overrides, refuses early if something is missing, then `exec`s `scripts/serve.sh` |
| `./flash wait` | polls the container and the API, shows the loading stage, prints the KV pool when up |
| `./flash test` | `scripts/smoke-test.sh` against the running server |
| `./flash status` / `logs` / `stop` / `start` / `rm` | the container's state, KV pool, active patches, running requests and prefix-cache hit rate; follow the log; stop (kept, `start` reloads in ~3-4 min); remove |
| `./flash profiles` | the list below |

Profiles (`profiles/*.env`, each a handful of `serve.sh` variables; copy one to make your own):

| profile | recipe | when |
|---|---|---|
| `default` | hybrid, YaRN 500k, deterministic top-k, reduced draft vocabulary, prefix caching, MTP=2 | the recommended one: best tournament score (45/51) |
| `speed` | default + `MTP=3` | +7% decode for about one tournament point |
| `context` | `MODE=hybrid-mtp` (NVFP4 MTP draft experts) | +22% KV pool for concurrency or long contexts, decode unchanged |
| `context-1m` | hybrid + `KV_DTYPE=fp8_e4m3`, 1M context | when you need 1M tokens in one request (small speed cost) |
| `shared` | default + `--long-prefill-token-threshold 1024` | several clients at once: decoding stays responsive while others prefill, single-stream TTFT −17–36% |
| `published` | `MODE=nvfp4`, YaRN 500k | the checkpoint exactly as published, nothing to prepare; ~26 tok/s |
| `native` | hybrid, 262k, no YaRN | if you never go past the native context |

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

**Download speed.** `scripts/download-weights.sh` uses the Hugging Face Xet backend (`XET=1`,
the default). It used to be off because it stalled on some Spark setups, but the Hub now
refuses to serve files over 50 GB through the plain path at all — the NVIDIA checkpoint's PLE
table is one 50 GiB shard, and the error it prints ("install hf_xet") is misleading, hf_xet is
in the image. Xet is also much faster: `--max-workers` parallelises across *files*, so a
checkpoint that is a dozen large shards leaves most of a gigabit idle over plain HTTPS.
Measured by [@techfury90](https://github.com/techfury90) on a DGX Spark on gigabit fibre,
pulling 81 GB (we saw the same ~105 MB/s on ours):

| | rate | 81 GB takes |
|---|---|---|
| plain HTTPS, 8 workers (default) | 14.7 MB/s (117 Mbit/s) | ~92 min |
| `XET=1` | **101 MB/s (809 Mbit/s)** | **13.4 min** |

`XET=0` falls back to plain HTTPS if Xet stalls for you. One caveat seen once: a Xet run
ended in an `httpx.ReadTimeout` *after* the last file completed — every blob was intact, but
the exit code was non-zero. Re-run to confirm; it is resumable, and a finished download
re-checks in seconds. `./flash doctor` tells you if an essential file (tokenizer, configs) or a
shard named by the index is missing.

**Do not run `prepare-hybrid.sh` on an unfinished download.** The hybrid layout is a copy of the
snapshot as it is at that moment; prepared too early it lacked `tokenizer.json`, and vLLM then died
on *"Couldn't instantiate the backend tokenizer… sentencepiece"* (issue #17 — nothing is missing
from the image). Since 2026-09-14 the script refuses an incomplete snapshot and, run again on an
existing layout, repairs it by adding whatever the snapshot has gained; `serve.sh` and `flash`
check the layout too and print that fix instead of vLLM's message.

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
The example uses the default served model name. If you start the server with `SERVED_MODEL_NAME=<name>`, use that value in the request's `model` field instead.

`MODE=nvfp4 scripts/serve.sh` (the default) serves the checkpoint as published at the native
262k context; `YARN=1 CTX=500000` goes to 500k (validated with a needle-in-a-haystack at 414k
tokens); `GPU_MEM=0.80` is the long-running-service setting, see [Tuning](#tuning-env-vars-for-scriptsservesh).

## Two checkpoint modes: NVFP4 or hybrid

`scripts/serve.sh` serves one of two layouts of the same NVFP4 checkpoint (NVIDIA's by default,
RadixArk's with `MODEL=`); pick with `MODE=`. The table below was measured on RadixArk's checkpoint;
on NVIDIA's the same conversion takes decode from ~28 to ~34 tok/s and the KV pool from ~565k to ~680k.

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

### NVFP4 MTP draft experts (`MODE=hybrid-mtp`, RadixArk only)

**RadixArk only**: NVIDIA's checkpoint already ships its MTP drafter in fp8 (that is where its KV
advantage comes from), so there is nothing to graft and `MODE=hybrid-mtp` refuses it; the `context`
profile pins `MODEL=RadixArk/…` for that reason. A graft on top of the RadixArk hybrid: the MTP draft head's routed experts (BF16 fused, ~4.7 GiB)
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

**Scope.** "Deterministic" here means: the same request, repeated one at a time, gives
byte-identical greedy output. It does **not** mean batch invariance: the same request served
concurrently with others lands in batches of different shapes, the kernels reduce in a
different order, and greedy output can diverge. That is a vLLM property for GDN-hybrid
models, not something this recipe causes or can fix — vLLM's `VLLM_BATCH_INVARIANT=1` does
not support GDN attention yet ([vllm#42960](https://github.com/vllm-project/vllm/issues/42960),
[vllm#48613](https://github.com/vllm-project/vllm/issues/48613)). Measured and documented by
[@aipiJuancho](https://github.com/aipiJuancho) in
[issue #32](https://github.com/blazux/qwen3.8-Flash-DGX/issues/32).

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

Once vllm#55122 is in a vLLM release this image is built from, patch 8 becomes
redundant. (Masking the never-written logits columns before the stock kernel does **not**
restore determinism, so it is the kernel itself.)

## Optional: persistent compile cache (`COMPILE_CACHE`)

`scripts/serve.sh` recreates the container on every run (`docker rm -f`, then `docker run`), so
what vLLM and its kernels compile inside the container is discarded and rebuilt on every boot. If
you keep one container and cycle it with `./flash stop` / `./flash start` this costs nothing, which
is why it went unnoticed. It costs you when something else recreates the container for you: a proxy
that loads and evicts models on demand (llama-swap and friends), CI, a tournament run that calls
`serve.sh` between configurations, or a systemd unit around `serve.sh`.

`COMPILE_CACHE=<name>` mounts four docker volumes over the four cache directories: `<name>-vllm`
(`/root/.cache/vllm`), `<name>-flashinfer` (`/root/.cache/flashinfer`), `<name>-triton` (Triton's
`/root/.triton`) and `<name>-nv` (the CUDA driver's JIT cache, `/root/.nv`).
`COMPILE_CACHE=/some/path` binds `/some/path/{vllm,flashinfer,triton,nv}` instead, to put them on a
chosen disk. Unset — the default — is exactly the behaviour above.

Where the time goes without it (one py-spy-profiled boot):

- **Triton, ~20 s of init engine.** About 140 kernels (QSA, PLE, fused MoE, GDN, sampler) are built
  from scratch: in the memory profile, in the graph capture that sizes the KV cache (its "10 secs"
  are mostly compiles; 2 s with a warm cache), in kernel warmup and in v0.30's `warmup_kernels`.
- **CUDA driver JIT, ~5 s of init engine.** The vision tower's cuBLAS GEMMs and flash-attn hdim-96
  kernel ship as sm80 PTX, which the driver compiles for GB10 into `/root/.nv`.
- **vLLM's `modelinfos`, ~11 s of API-server startup.** Without it the model classes are inspected
  in a subprocess. v0.30 writes no torch.compile cache for this recipe, so this 16 KB is all the
  `-vllm` volume holds. The FlashInfer volume made no measurable difference.

Measured on a GX10, hybrid + YaRN 500k + MTP=2, weights over NFS, same recipe each time, one row per
boot:

| boot | init engine | graph captures | start → ready | Triton kernels / driver JIT files written |
|---|---|---|---|---|
| unset (default) | 53.4 s | 10 + 2 s | 121.1 s | 139 / 13 |
| vllm + flashinfer volumes only | 53.5 s | 10 + 2 s | 110.3 s | 139 / 13 |
| four volumes, populating | 53.9 s | 10 + 2 s | 111.1 s | 139 / 13 |
| four volumes, reused (4 boots) | 27.5–28.1 s | 2 + 2 s | 83.4–85.0 s | 0 / 0 |
| reused, only the `-nv` volume emptied | 33.3 s | 2 + 2 s | 89.4 s | 0 / 13 |

Reused against unset is **−26 s of init engine and −37 s of start → ready** per boot. Populating
costs nothing. Disk: 51 MB Triton, 124 MB driver JIT, under 1 MB for the other two. Startup only:
greedy probe (5 prompts) byte-identical in text and first-token logprobs against a boot without the
Triton and driver volumes. Triton keys each kernel by its source, constants and backend, and the
driver by PTX and driver version, so a changed kernel or a new driver compiles again instead of
reusing a stale entry.

On the preview base, where this option started (PR #21), it was vLLM's torch.compile cache that
mattered (−80 s of init engine) and Triton's made no difference; those measurements are in
[docs/HISTORY.md](docs/HISTORY.md#persistent-compile-cache-on-the-preview-base-pr-21).

## Faster weight loading (patches 14–18, default on)

Most of a boot used to be "Loading weights", and the cause was not the disk. vLLM copies each of
the ~149k routed-expert tensors to the GPU on its own, straight from the memory-mapped checkpoint.
On GB10 that copy costs ~1.7 ms per 800 KiB tensor when the source is a file-backed page, and
~0.23 ms from ordinary memory. Patch 14 clones each tensor before the copy, which produces the
same bytes and uses one transient tensor of scratch memory. With the compile cache reused as
above, it takes startup from ~11 min to 4 min 32 s. Patches 15–18 remove most of the rest: small
tensors are read with `pread` instead of mmap (never the PLE table), expert names are matched by
index, `embed_tokens` / `lm_head` are copied in 64 MiB pieces, and the MTP drafter skips the
tensors it doesn't need before reading them. Startup: 2 min 8 s.

| (DGX Spark, hybrid, NVIDIA checkpoint) | before | patch 14 | patches 14–18 |
|---|---|---|---|
| Loading weights, main model | 450–541 s | 150 s | 35.5 s |
| Loading weights, MTP drafter | 46 s | 32 s | 1.2 s |
| startup | ~11 min | 4 min 32 s | 2 min 8 s |

Each can be switched off to compare against the stock loader:
`VLLM_LOAD_CLONE=0`, `VLLM_LOAD_PREAD=0`, `VLLM_MOE_NAME_INDEX=0`, `VLLM_LOAD_EMBED_CHUNK=0`,
`VLLM_MTP_NAME_PREFILTER=0`, each added as `-e` to the `docker run` line in `scripts/serve.sh`. Profiles,
benchmarks and validation are in [HOW-IT-WORKS](docs/HOW-IT-WORKS.md#weight-loading-the-per-expert-h2d-copy-patch-14)
and [the section after it](docs/HOW-IT-WORKS.md#the-rest-of-weight-loading-patches-1518).

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

## Checkpoints: NVIDIA's NVFP4 (default) and RadixArk's

Two NVFP4 quantizations of Qwen3.8-Flash-Next fit this recipe, and the recipe does not care which
one it serves: `MODEL=<org/name>` on `download-weights.sh`, `prepare-hybrid.sh` and `serve.sh` (or
`./flash setup default MODEL=…` then `./flash serve`). Both have NVFP4 routed experts, the fp8 PLE
table this whole repo is about, and bf16 side layers that `prepare-hybrid.sh` converts to blockwise fp8
the same way (the 300 side tensors are the same weights, down to the conversion error); the
deterministic top-k, the reduced draft vocabulary, prefix caching and YaRN apply identically.

- **[nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)** — **the
  default since 2026-09-14.** NVIDIA's own ModelOpt export (issue #17, @PathosEthosLogos). 124 GiB in
  24 files, one of them a 50 GiB PLE shard that only downloads through Xet (`download-weights.sh` uses
  Xet by default). Its MTP drafter keeps its experts in blockwise fp8 under a ModelOpt *mixed-precision*
  config; vLLM 0.29 could not load that (no method for those experts, and the layer index of the
  drafter not remapped). vLLM v0.30 loads it as is (vllm#55513); before that this repo carried the
  fix as patch 11 (a backport by @techfury90 and a stopgap shim, see [docs/HISTORY.md](docs/HISTORY.md)).
- **[RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)** — the
  default until 2026-09-14, the checkpoint every number in the sections above was measured on. 122 GiB
  in 418 files, plain ModelOpt NVFP4 config, bf16 MTP drafter (hence the `hybrid-mtp` graft, which only
  exists for it). Fully supported, one variable away:

```bash
MODEL=RadixArk/Qwen3.8-Flash-Next-NVFP4 scripts/download-weights.sh
MODEL=RadixArk/Qwen3.8-Flash-Next-NVFP4 scripts/prepare-hybrid.sh
MODEL=RadixArk/Qwen3.8-Flash-Next-NVFP4 MODE=hybrid YARN=1 CTX=500000 scripts/serve.sh
#   or: ./flash setup default MODEL=RadixArk/Qwen3.8-Flash-Next-NVFP4 && ./flash serve default MODEL=RadixArk/Qwen3.8-Flash-Next-NVFP4
```

### Why the default changed: the head-to-head (2026-09-13/14)

The rule of this repo is quality first; a default only moves on measured data, never on a smoke test.
So the two checkpoints were run through the same protocol, back to back, on the same box:

- **same image** (`qwen38-flash-dgx`, v0.29 base, the tree of `main`), **same recipe** (hybrid,
  deterministic top-k, draft vocab 65k, MTP=2, YaRN 500k, prefix caching), **only `MODEL` differs**;
- one boot per checkpoint; on that boot: speed, memory, long-context and determinism probes, then
  **5 full passes of the agentic tournament** — 55 scenarios (tool loops with injected faults,
  long ledgers, logic grids, calendar and org-chart deduction, hidden-test coding), temperature 0.2,
  32k reasoning tokens per turn, partial credit on multi-item scenarios — then a dedicated block on
  the four coding tasks where this model's reasoning most often runs away (10 samples each);
- 16 h of GPU time, 275 scored runs per side, every trajectory kept.

| | RadixArk hybrid + MTP=2 | **NVIDIA hybrid + MTP=2 (default)** |
|---|---|---|
| tournament, 5 passes × 55 scenarios | 86.1% ± 1.9 (85.7 / 84.2 / 88.1 / 88.1 / 84.5) | **88.8% ± 1.0** (89.0 / 89.0 / 87.2 / 89.0 / 89.9) |
| items lost on the 46 scenarios that never run away (230 runs) | 4.1 | **0.7** |
| reasoning runaways (32k-token cap), all coding tasks | 49/65 | 46/65 (p = 0.55: equal) |
| scenarios where one side is behind | none for RadixArk beyond one-run noise on g9 | none for NVIDIA |
| needle 100k / 185k / 323k / 413k | 4/4 found | 4/4 found |
| deterministic at temperature 0 | yes | yes |
| decode, single stream, greedy prose | **37.1 tok/s** | 34.5 tok/s (−7%) |
| decode per stream, 4 concurrent agents | 21.6 tok/s | 22.1 tok/s |
| prefill 8k / 32k (cold) | 5.1 s / 10.6 s | 5.0 s / 11.3 s |
| prefill at 185k / 323k / 413k | 1,919 / 2,497 / 2,856 tok/s | 1,857 / 2,465 / 2,782 tok/s |
| KV pool @500k, `GPU_MEM=0.80` | 589k tokens | **679k tokens (+15%)**; +22% on another boot |
| weights loaded | 77.3 GiB | **74.9 GiB** |
| download | 122 GiB, 418 files, HTTPS or Xet | 124 GiB, 24 files, **Xet required** (50 GiB shard) |

How to read it, honestly:

- The 2.7-point gap is above the noise floor of a 5-pass design (bootstrap CI [+0.4, +5.3], permutation
  p = 0.048) but about half of it is luck on the runaway-prone coding tasks, where both checkpoints
  derail at the same rate. The half that is not luck is a handful of slips only RadixArk made: a
  counting error on a paginated ledger (5 of 5 passes there, 8 of 18 across three images; NVIDIA 14 of
  15), one fabricated tool trajectory, two mis-executed multi-step tool tasks. NVIDIA's only systematic
  weakness in the data is a reporting convention (it under-counts a `total_moved` line while executing
  the 30 operations perfectly).
- The claim this data supports is **"parity or better on quality, +15–22% KV, −8% single-stream
  decode, vendor checkpoint whose loading fix is upstream"** — not "NVIDIA is 2.7 points smarter".
- Under that reading the default rule is unambiguous: at quality parity or better, the checkpoint that
  leaves more room for context and concurrency, and that the next vLLM release will load with no patch
  at all, is the one to ship. If your use is a single chat stream where 37 vs 34 tok/s is what you
  feel, RadixArk keeps a small edge and is one variable away.

Derivatives of either checkpoint in the same ModelOpt layout (abliterated variants such as
`Jiunsong/SuperQwen3.8-Flash-Next-abliterated-NVFP4-DGX-Spark` or
`drowzeys/keys-Qwen3.8-Flash-Next-NVFP4-dual-ablit-house-qsa-L3-47`) run with the same commands; we do
not ship or endorse them, we only note that the recipe does not care. **Compressed-tensors derivatives**
(`orcarouter/Qwen3.8-Flash-Next-Uncensored-NVFP4` and `lychee888/…-FP8PLE`, issue #23) are a different
format: their quantization config lives in `config.json` (no `hf_quant_config.json`, which is fine, vLLM
reads it from there) and their dense side layers are **already fp8**, so the hybrid step has nothing to
convert — `prepare-hybrid.sh` and `flash doctor` say so and point to the `published` profile
(`MODE=nvfp4`). The PLE table of the lychee888 build uses RadixArk's exact file and key layout, so the
mmap patch should apply; we have not booted one ourselves.

## Tuning (env vars for `scripts/serve.sh`)

| Var | Default | Notes |
|---|---|---|
| `MODE` | `nvfp4` | `hybrid` = fp8 side layers (see above; needs `scripts/prepare-hybrid.sh`). |
| `PREFIX_CACHE` | `1` | `--enable-prefix-caching`. Correct with this image (block_size fix). |
| `DET_TOPK` | `1` | Deterministic QSA top-k **kernel** (vllm#55122): identical outputs at T=0 at full kernel speed. `0` = stock kernel (non-deterministic, may drop attention candidates, issue #3). |
| `EXACT_TOPK` | `0` | `1` = exact `torch.topk` fallback (deterministic; −8% prefill at 8k, −20–40% at 32k+). Wins over `DET_TOPK` when set. |
| `DRAFT_VOCAB` | `1` | MTP drafter scores only the 65,536 most frequent tokens (+20% decode, same tournament score, outputs unchanged). `0` = full vocabulary; a path = your own ids (`tools/build_draft_vocab.py`). |
| `MADVISE` | `random` | `madvise` on the mmapped PLE table: `random` (no readahead: cold prefill −4–8%, cleaner page cache) or `normal`. |
| `FAST_ROWS` | `0` | PLE gathers of up to this many unique rows run inline on one thread; larger ones are split across the `WORKERS` pool. `0` sends every gather to the pool, so page faults on rows the page cache dropped overlap instead of queueing: +8% decode at 1 stream, +17% aggregate at 4 streams, same rows (see the 2026-09-14 update). `512` = the old inline fast path, faster only when every row is already cached. |
| `EFFORT_ALIAS` | `1` | Accept every `reasoning_effort` a client can send. The checkpoints' template takes only `xhigh` (default), `medium` and `low` and 400s the rest — including Claude Code's default `high`. `1` serves a copy of the checkpoint's own template whose effort-resolving line maps `high`/`max` → `xhigh` and `minimal` → `low` (other values render byte-identically; the copy goes to the first writable of `$HF_CACHE/qwen38-flash-dgx/chat-templates/`, `~/.cache/qwen38-flash-dgx/chat-templates/` and `.cache/chat-templates/` in the checkout, and is bind-mounted into the container, so a root-owned HF cache does not disable it). Applied only when the template has that check and that exact line. `0` = the template as shipped. |
| `SERVED_MODEL_NAME` | `qwen3.8-flash-next` | Model name exposed by the OpenAI-compatible API (`--served-model-name`). Set this to a stable client-facing name without changing the underlying Hugging Face `MODEL`. |
| `PORT` | `18300` | API port |
| `CTX` | `262144` | Max context. Native is 262144; with `YARN=1` up to `500000` is validated. |
| `YARN` | `0` | `1` = YaRN rope scaling (factor 4, Qwen's recipe) for `CTX` > 262144. |
| `SEQS` | `8` | Max concurrent sequences. **Do not benchmark with 1–2**: excess requests queue silently and aggregate tok/s flatlines (see below). |
| `GPU_MEM` | `0.80` | Fraction of the 128 GB pool for weights+KV. `0.85` buys ~2 GiB more KV, but after a day at `0.85` the box drifted into swap, and `0.875` got OOM-killed on a 300k-token prefill with MTP. The lower you set it, the more RAM the page cache has for the 48 GiB table — which is what your prefill speed depends on (below). Right after stopping another big container the first boot can fail with "13.5 GiB KV cache is needed, larger than available" — memory not yet released; the `unless-stopped` retry succeeds. |
| `MTP` | `2` | Speculative tokens from the model's MTP head (`0` = off). `3` is +7% decode but cost a point at the tournament (44 vs 45/51), so it stays an option. |
| `KV_DTYPE` | `auto` | `auto` = bf16 (recommended). `fp8_e4m3` = ~1.9× KV pool, 1M context on one box, at −10% decode / −30% prefill and a measurable quality cost — see [fp8 KV cache](docs/HOW-IT-WORKS.md#fp8-kv-cache-on-the-qsa-path-opt-in) before using it. |
| `PREWARM` | `0` | `1` streams the 48 GiB table once at boot to warm the page cache — steadier first-request latency, ~10 s extra startup. |
| `WORKERS` | `32` | Threads used for the mmap gather: every gather at the default `FAST_ROWS=0`; with `FAST_ROWS=512`, only gathers above 512 unique rows (decode-sized gathers then run inline). |
| `COMPILE_CACHE` | | Keep vLLM's, FlashInfer's, Triton's and the CUDA driver's caches across boots — this script recreates the container every run, so by default they are rebuilt each time. `<name>` = four docker volumes, `/abs/path` = four bind mounts. **−37 s of startup** (init engine −26 s) per boot after the first, ~175 MB of disk; only worth setting if something recreates the container for you (a model-swapping proxy, CI, tournament runs, a systemd unit). See [above](#optional-persistent-compile-cache-compile_cache). |
| `LOG_REQUESTS` | `0` | `1` logs every prompt and output (`--enable-log-requests --enable-log-outputs`, with DEBUG on vLLM's request logger only) so `tools/vllm_watch.py` can show sessions live. Debugging only: it puts user content in the Docker log, unbounded. |
| `PROM_MULTIPROC` | `0` | `1` runs prometheus_client in multiprocess mode so engine-side metrics (`vllm:ple_mmap_*`) reach `/metrics`. Opt-in, because it stops vLLM exporting its `*_created` samples and the `process_*` / `python_*` metrics (`process_start_time_seconds` included; `vllm:ple_mmap_engine_start_time_seconds` stands in as a restart marker); see *Watching the mmapped table* below. |
| `KV_CACHE_MEM` | | Passed through as `--kv-cache-memory-bytes`. `GPU_MEM` is a fraction of *total* device memory, so it leaves whatever was already resident on the table; vLLM prints the exact figure it would accept at startup ("Replace gpu_memory_utilization config with `--kv-cache-memory=...`"). On a Spark that headroom is also what the page cache uses for the PLE table, so taking it is a trade, not free memory — watch `vllm:ple_mmap_gather_seconds_total` when you do. |
| `EXTRA` | | Extra vLLM flags, passed verbatim — e.g. `--long-prefill-token-threshold 1024` for multi-client responsiveness (see [the concurrency section](#decoding-clients-stall-while-other-clients-prefill-the-long-prefill-token-threshold-slider)), `--api-key <secret>`. |

### Watching the mmapped table (`vllm:ple_mmap_*`)

The PLE table is the one component whose cost depends on runtime state rather than
configuration: how much of its 47.7 GiB the page cache is holding decides your prefill
speed, and that moves as the KV pool, the request mix and the OS all pull on the same
unified memory. The module exports five counters so this is visible on a dashboard
rather than only in a windowed log line that a container restart destroys:

```
vllm:ple_mmap_lookup_ops_total         lookups (hash + gather + H2D)
vllm:ple_mmap_op_seconds_total         cumulative seconds in the lookup op, GPU wait included
vllm:ple_mmap_gpu_wait_seconds_total   of which: waiting for GPU work queued ahead of the lookup (not PLE cost)
vllm:ple_mmap_dedup_seconds_total      of which: copying the row ids to the host and deduplicating them
vllm:ple_mmap_gather_seconds_total     of which: the row gather (disk reads)
vllm:ple_mmap_stage_seconds_total      of which: staging the rows for the GPU (pinned copy, H2D launch)
vllm:ple_mmap_rows_total               rows gathered
vllm:ple_mmap_bytes_total              bytes read from the table
```

The lookup starts with a blocking copy of the row ids to the host, and that copy waits for every GPU
kernel queued ahead of it: the n-gram hashing and the layers before the PLE layer. `op_seconds` therefore
mixes their compute with the lookup's own cost (one prefill window read 165 ms per lookup, of which 8 ms
was the gather). `gpu_wait_seconds` is that wait on its own, so `op − gpu_wait` is what the lookup itself
costs the step. The periodic `PLE mmap stats` log line shows the same split at its end:
`gpu-wait X ms/op, host Y ms/op (dedup, gather, stage)`.

They are registered in the EngineCore process, so they only reach `/metrics` when
prometheus_client runs in multiprocess mode. vLLM turns that on only for
`api_server_count > 1`; `scripts/serve.sh` does it for the single-server setup used here when you opt in with
`PROM_MULTIPROC=1`, by pointing `PROMETHEUS_MULTIPROC_DIR` at a fresh tmpfs.

Switching to multiprocess mode was checked against a live server by diffing the
complete `/metrics` before and after: vLLM's other 71 metric families are exported with
identical label sets and no per-process `pid` label. Two things are lost: the 35
`*_created` families, which prometheus_client does not export in multiprocess mode, and the
default `process_*` / `python_*` collectors (`process_start_time_seconds`,
`process_resident_memory_bytes`, `process_cpu_seconds_total`, `python_gc_*`, `python_info`),
because vLLM then serves a fresh registry that holds only the multiprocess collector (reported
by [@PhilX-rgb](https://github.com/PhilX-rgb) in [#36](https://github.com/blazux/qwen3.8-Flash-DGX/issues/36)).
For restart detection, `vllm:ple_mmap_engine_start_time_seconds` carries the EngineCore's start
time and changes on every restart. That is why exporting the counters is opt-in, so nothing
changes for existing dashboards unless you ask for it.

The views worth graphing:

```promql
# host seconds per lookup: what the PLE lookup itself costs each step
(rate(vllm:ple_mmap_op_seconds_total[5m]) - rate(vllm:ple_mmap_gpu_wait_seconds_total[5m]))
  / rate(vllm:ple_mmap_lookup_ops_total[5m])
# page-cache health: the share of the lookup's own time spent on disk
rate(vllm:ple_mmap_gather_seconds_total[5m])
  / (rate(vllm:ple_mmap_op_seconds_total[5m]) - rate(vllm:ple_mmap_gpu_wait_seconds_total[5m]))
# NVMe read bandwidth from the table
rate(vllm:ple_mmap_bytes_total[5m])
```

The disk share is the page-cache health signal. It climbs as the cache is squeezed and falls
as the hot region settles in. Divide by `op_seconds` alone and it also moves with GPU load,
which says nothing about the cache. Pair it with `Cached` from a node exporter, since nothing
in vLLM's own metrics exposes the quantity that actually governs it. `VLLM_PLE_MMAP_PROMETHEUS=0` turns the counters off.

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
  for decode-sized batches (`VLLM_PLE_MMAP_FAST_ROWS`, module default 512; larger gathers are split
  into `VLLM_PLE_MMAP_CHUNK`=2048-row tasks across `WORKERS` threads). `serve.sh` now sets it to 0
  (`FAST_ROWS`): on a Spark enough rows miss the page cache that overlapping their faults on the pool
  wins, by 8% at 1 stream and 17% at 4. Also: bf16/f16 tables,
  `VLLM_PLE_MMAP_DIR` to serve the table from another directory, and a periodic
  `PLE mmap stats` log line (`VLLM_PLE_MMAP_STATS_SEC`, default 30).
- **Mamba state-copy guard** — on the preview image, [vllm#50729](https://github.com/vllm-project/vllm/pull/50729)
  (the overlapping-copy race fix by @AndreasKaratzas) plus a bounds check. v0.30 ships the race fix
  itself, so the drop-in went away with the preview image ([docs/HISTORY.md](docs/HISTORY.md)).
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
profiles/*.env                    named recipes for it (default, speed, context, context-1m, shared, published, native)
Dockerfile                        official vLLM v0.31.0 image + the patches below (qwen38-flash-dgx:v0.31).
                                     Gaps in the numbering are upstream now: 3, 9, 11 (docs/HISTORY.md)
src/vllm_ple_mmap.py              1. mmap PLE table (opaque op, breaks the CUDA-graph capture)  VLLM_PLE_MMAP=1
src/patch_mamba_block_size.py     4. prefix-caching block_size fix
src/patch_qsa_exact_topk.py       5. exact, deterministic QSA top-k                  VLLM_QSA_EXACT_TOPK=1
(Dockerfile patch 8)              8. deterministic persistent_topk kernel, built at docker build  VLLM_QSA_DET_TOPK=1
src/patch_qsadet.py                  wires that kernel into the QSA indexer
                                     from @jschmied's repo (pinned commit) — vllm#55122
src/patch_mtp_draft_vocab.py     10. reduced draft vocabulary for the MTP drafter          VLLM_MTP_DRAFT_VOCAB=<ids.npy>
src/draft_vocab_65536.npy            the default 65,536-id set (tools/build_draft_vocab.py rebuilds it)
src/vllm_fp8_hybrid_modelopt.py   6. NVFP4 experts + fp8 side layers dispatch        VLLM_FP8_HYBRID=1
                                     (patches the NVFP4 and the mixed-precision ModelOpt config classes)
src/patch_qsa_fp8_kv.py           7. fp8_e4m3 KV cache on the QSA path (by @Nanetnounou) --kv-cache-dtype fp8_e4m3
src/patch_moe_load_clone.py      14. clone mmap-backed expert weights before the H2D copy          VLLM_LOAD_CLONE=0 disables
                                     (main weight load 541 -> 150 s on a Spark; docs/HOW-IT-WORKS.md)
src/patches/qwen-tool-*.patch    12, 13. Qwen tool-marker fixes in the parser engine (+ their test modules)
src/patch_load_pread.py          15. pread checkpoint tensors <= 64 MiB instead of mmap views     VLLM_LOAD_PREAD=0 disables
                                     (never the PLE table; needs 14)
src/patch_moe_name_index.py      16. indexed FusedMoE expert-name matching (~30 s of Python)        VLLM_MOE_NAME_INDEX=0 disables
src/patch_embed_chunked_copy.py  17. embed_tokens / lm_head copied to the GPU in 64 MiB pieces     VLLM_LOAD_EMBED_CHUNK=0 disables
                                     15-17 together: weight load 150 + 32 s -> 35 + 12 s (docs/HOW-IT-WORKS.md)
src/patch_mtp_name_prefilter.py  18. MTP drafter skips non-MTP tensors before reading (12 -> 1.2 s) VLLM_MTP_NAME_PREFILTER=0 disables
src/test_ple_mmap_cpu.py          CPU unit test for the gather (no GPU needed)
src/test_qsa_exact_topk_cpu.py    CPU unit test for the exact top-k (no GPU needed)
src/test_moe_name_index_cpu.py    CPU unit test: patch 16 visits exactly the entries of the original loop
src/test_load_patches_cpu.py      CPU check of patches 15 and 17 against real checkpoint files
src/test_mtp_prefilter_cpu.py     CPU check of patch 18: the drafter's exact tensor set, on a real snapshot
tests/test_fp8_kv_read.py         GPU check of patch 7: the fp8 read path matches bf16 bit for bit
tools/fp8_convert.py              side-layer bf16 -> blockwise fp8 (by @Saren-Arterius)
tools/bench_moe_load.py           per-expert H2D copy micro-benchmark behind patch 14 (needs a free GPU)
tools/profile_boot.sh             py-spy every process of a boot in 10 s slices (needs SYS_PTRACE, see header)
tools/pyspy_slices.py             summarize those slices by process / thread / frame
scripts/download-weights.sh       MODEL (default nvidia/Qwen3.8-Flash-Next-NVFP4), EXCLUDE, MAX_WORKERS, XET
scripts/prepare-hybrid.sh         one-time: build the -fp8hybrid snapshot
scripts/prepare-mtp-graft.sh      one-time: graft the NVFP4 MTP draft experts onto it (MODE=hybrid-mtp, RadixArk only)
tools/vllm_watch.py               live per-session view of prompts / reasoning / outputs / stats (needs LOG_REQUESTS=1; @0x3dlux)
scripts/serve.sh                  MODE=nvfp4|hybrid|hybrid-mtp, SERVED_MODEL_NAME, PREFIX_CACHE, DET_TOPK, DRAFT_VOCAB, MADVISE, EXACT_TOPK, KV_DTYPE, YARN, ...
scripts/smoke-test.sh             health, coherence, prefix-cache hit, determinism, tok/s
scripts/greedy-probe.sh           greedy probe set; diff two arms to gate a draft/checkpoint swap
docs/HOW-IT-WORKS.md              how each patch works, with the measurements
docs/HISTORY.md                   the earlier updates and the preview / v0.29 bases
```

Run the unit tests (no GPU):

```bash
docker run --rm -v "$PWD/src:/t" -w /t --entrypoint python3 qwen38-flash-dgx:v0.31 test_ple_mmap_cpu.py
docker run --rm -v "$PWD/src:/t" -w /t --entrypoint python3 qwen38-flash-dgx:v0.31 test_qsa_exact_topk_cpu.py
# test_moe_name_index_cpu.py exercises the retired local patch, not v0.31's native index.
```

## Limitations & notes

- **One big model at a time.** At `GPU_MEM=0.80` this uses most of the 128 GB pool;
  don't co-locate another large model (an 8B embedding model next to it already
  starves the KV cache — we moved ours to another machine).
- **No `torch.compile` for this model.** vLLM v0.30 captures it with *breakable* piecewise CUDA
  graphs instead; the PLE lookup, which cannot live inside a capture, ends a graph segment by itself
  and writes into a static buffer (docs/HOW-IT-WORKS.md).
- **1M context** needs the fp8 KV cache (`KV_DTYPE=fp8_e4m3`, profile `context-1m`), which costs some speed; in bf16 a single 1M request needs ~26 GiB of KV and 500k with
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

- Download knobs (Xet, `EXCLUDE`, `MAX_WORKERS`), the `vllm:ple_mmap_*` Prometheus counters and `KV_CACHE_MEM`:
  **[@techfury90](https://github.com/techfury90)** (PR #19); also spotted the upstream fix for NVIDIA's fp8 MTP experts (vllm#55513).
- `COMPILE_CACHE`, the persistent compile cache: **[@AronRubin](https://github.com/AronRubin)** (PR #21).
- The pointer to NVIDIA's own NVFP4 checkpoint: **[@PathosEthosLogos](https://github.com/PathosEthosLogos)** (issue #17).
- `tools/vllm_watch.py`, the live session viewer: **[@0x3dlux](https://github.com/0x3dlux)** (issue #12).
- The nudge to port the recipe to the vLLM releases (v0.29, then v0.30): **[@ChengYen-Tang](https://github.com/ChengYen-Tang)** (issue #14).

- The two ideas behind the reduced draft vocabulary and the `MADV_RANDOM` table advice come
  from **[MiaAI-Lab](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark)**'s
  recipe for their own checkpoint; both are reimplemented here from scratch (their code is
  AGPL-3.0) and measured on this checkpoint.

- Model: **Qwen team, Alibaba** — Qwen3.8-Flash-Next.
- NVFP4 checkpoints: **[nvidia/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4)** (the default since 2026-09-14) and **[RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)** (the default before, still supported).
- NVFP4 MTP draft experts (the `hybrid-mtp` graft donor): **[Inferact/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/Inferact/Qwen3.8-Flash-Next-NVFP4)**; the graft recipe follows
  [thavoc's graft write-up](https://gist.github.com/thavoc/d7083457f6f2d981f879670c34df34ab)
  and [Peuqui/mtp-quant-transplant](https://github.com/Peuqui/mtp-quant-transplant).
- Serving engine and base image: **vLLM** (`vllm/vllm-openai:v0.31.0`; the model's support started as
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
  finding and its padding drop-in (patch 9, until vllm#52775), the independent
  reproduction on a DGX Spark, the native-offload fixes and the concurrency measurements:
  **[@jschmied](https://github.com/jschmied)**
  ([issue #1](https://github.com/blazux/qwen3.8-Flash-DGX/issues/1),
  [qwen38-flash-next-gb10](https://github.com/jschmied/qwen38-flash-next-gb10)).
- The mmap-PLE patch, the hybrid dispatch for ModelOpt-NVFP4, the prefix-caching root
  cause and fix, the exact top-k path and the GB10 serving recipe in this repo: see
  [LICENSE](LICENSE) (Apache-2.0).
