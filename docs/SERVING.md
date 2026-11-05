# Serving reference

Start with the [README](../README.md) for installation and profile selection.

## Profiles

| profile | recipe | when |
| --- | --- | --- |
| `default` / `mtp-fast` | NVIDIA hybrid, MTP3 NVFP4 proposal head, 16k batch, 13 GiB FP8 KV, 500k context | default on vLLM 0.33. [Performance and memory](../README.md#vllm-v033-upgrade) |
| `mtp-standard` | NVIDIA hybrid, YaRN 500k, MTP2, reduced draft vocabulary | previous default recipe |
| `dflash` | `mtp-standard` + the DFlash2 draft model (incoai, 0.6B) instead of the MTP head (`DRAFT_MODEL`, `MTP=0`, `DRAFT_TOKENS=4`) | Higher GPU memory use. Performance varies with the prompt and sampling settings. [v0.33 measurements](VLLM-033.md#what-changed-in-serving). |
| `radixark-graft` | RadixArk NVFP4 with the NVFP4 MTP draft-experts graft | Use with a prepared RadixArk checkpoint. `context` is an alias. |
| `dflash-1m` | DFlash2 K4, 1M context, BF16 recurrent state, 21 GiB fixed FP8 KV | faster single-stream decoding. [Compare 1M profiles](../README.md#choosing-a-1m-context-profile) |
| `mtp4-1m` | MTP4, full draft vocabulary, same 1M context and cache budget | matched reference with more KV capacity at the same memory budget |
| `shared` | `mtp-standard` with a 1,024-token long-prefill threshold | Shorter pauses in ongoing streams at the cost of a slower concurrent prefill. |
| `published` | NVIDIA NVFP4 checkpoint as published, YaRN 500k | No hybrid conversion. Automatic KV sizing can be tight at 500k. |
| `native` | hybrid, 262k, no YaRN | if you never go past the native context |

The earlier MTP recipes remain available for existing commands.

| Compatibility profile | Equivalent invocation | When to use it |
| --- | --- | --- |
| `speed` | `./flash serve mtp-standard MTP=3` | Similar throughput to MTP2 on the tested sampled workloads. |
| `concurrent` | `./flash serve mtp-standard SEQS=16 MAMBA_SSM_DTYPE=bfloat16` | Similar throughput to `mtp-fast` at eight concurrent requests in the 0.33 check. |
| `context-1m` | `./flash serve mtp-standard CTX=1000000 KV_DTYPE=fp8_e4m3` | MTP2 with automatic KV sizing. See the two measured 1M profiles for fixed-budget comparisons. |

On the GX10 with vLLM 0.33, `shared` reduced the longest median pause
during an ongoing decode from 2.63 s to 0.48 s versus `mtp-standard` in
a three-repeat test with a concurrent 52k prompt. The long prompt's median
time to first token increased from 17.2 s to 29.1 s.

The `published` profile passed the smoke check after two automatic restarts.
On the first two attempts vLLM estimated capacity at roughly 490k tokens.
If startup reports a KV shortage, lower `CTX` below the ceiling in the log.

## Tuning (env vars for `scripts/serve.sh`)

The defaults in this table apply to the script when called directly. Named
profiles set their own values as shown above.

| Var | Default | Notes |
| --- | --- | --- |
| `MODE` | `nvfp4` | `hybrid` = fp8 side layers (see above; needs `scripts/prepare-hybrid.sh`). |
| `PREFIX_CACHE` | `1` | `--enable-prefix-caching`. Correct with this image (block_size fix). |
| `DET_TOPK` | `1` | Deterministic QSA top-k **kernel** (vllm#55122): identical outputs at T=0 at full kernel speed. `0` = stock kernel (non-deterministic, may drop attention candidates, issue #3). |
| `EXACT_TOPK` | `0` | `1` = exact `torch.topk` fallback (deterministic; −8% prefill at 8k, −20–40% at 32k+). Wins over `DET_TOPK` when set. |
| `DRAFT_VOCAB` | `1` | MTP drafter scores only the 65,536 most frequent tokens (+20% decode, same tournament score, outputs unchanged). `0` = full vocabulary; a path = your own ids (`tools/build_draft_vocab.py`). MTP-head only — ignored for a `DRAFT_MODEL` (dflash) draft, which requires the full verifier vocabulary. |
| `MADVISE` | `random` | `madvise` on the mmapped PLE table: `random` (no readahead: cold prefill −4–8%, cleaner page cache) or `normal`. |
| `FAST_ROWS` | `0` | PLE gathers of up to this many unique rows run inline on one thread; larger ones are split across the `WORKERS` pool. `0` sends every gather to the pool, so page faults on rows the page cache dropped overlap instead of queueing: +8% decode at 1 stream, +17% aggregate at 4 streams, same rows (see the 2026-09-14 update). `512` = the old inline fast path, faster only when every row is already cached. |
| `EFFORT_ALIAS` | `1` | Accept every `reasoning_effort` a client can send. The checkpoints' template takes only `xhigh` (default), `medium` and `low` and 400s the rest — including Claude Code's default `high`. `1` serves a copy of the checkpoint's own template whose effort-resolving line maps `high`/`max` → `xhigh` and `minimal` → `low` (other values render byte-identically; the copy goes to the first writable of `$HF_CACHE/qwen38-flash-dgx/chat-templates/`, `~/.cache/qwen38-flash-dgx/chat-templates/` and `.cache/chat-templates/` in the checkout, and is bind-mounted into the container, so a root-owned HF cache does not disable it). Applied only when the template has that check and that exact line. `0` = the template as shipped. |
| `SERVED_MODEL_NAME` | `qwen3.8-flash-next` | Model name exposed by the OpenAI-compatible API (`--served-model-name`). Set this to a stable client-facing name without changing the underlying Hugging Face `MODEL`. |
| `PORT` | `18300` | API port |
| `CTX` | `262144` | Max context. Native is 262144; with `YARN=1` up to `500000` is validated. |
| `YARN` | `0` | `1` = YaRN rope scaling (factor 4, Qwen's recipe) for `CTX` > 262144. |
| `SEQS` | `8` | Max concurrent sequences. **Do not benchmark with 1–2**: excess requests queue silently and aggregate tok/s flatlines (see below). |
| `CPUSET` | `5-9,15-19` | docker `--cpuset-cpus`: pin the container to the GB10's 10 Cortex-X5 performance cores. `CPUSET=` (empty) lets a newly created container use all 20 cores. The [measured concurrent profile](HISTORY.md) uses this pinning; its independent speed contribution has not been established. |
| `MAMBA_SSM_DTYPE` | | `--mamba-ssm-cache-dtype` for GDN recurrent state. Unset follows the checkpoint configuration: the tested NVIDIA checkpoint specifies `float32`. Explicit `bfloat16` halves bytes per recurrent-state element and changes its precision. Enabled by `mtp-fast`, the 1M A/B profiles and `concurrent`. |
| `GPU_MEM` | `0.80` | Fraction of the 128 GB pool for weights+KV. `0.85` buys ~2 GiB more KV, but after a day at `0.85` the box drifted into swap, and `0.875` got OOM-killed on a 300k-token prefill with MTP. The lower you set it, the more RAM the page cache has for the 48 GiB table — which is what your prefill speed depends on (below). Right after stopping another big container the first boot can fail with "13.5 GiB KV cache is needed, larger than available" — memory not yet released; the `unless-stopped` retry succeeds. |
| `MTP` | `2` | Speculative tokens from the model's MTP head (`0` = off). The `mtp-fast` profile sets `3`. |
| `MTP_DRAFT_QUANT` | | Quantize the MTP proposal head with `fp8` or `nvfp4`. Requires `DRAFT_VOCAB=0`. The `mtp-fast` profile sets `nvfp4`. |
| `BATCH_TOKENS` | `8192` | Maximum scheduled tokens per engine step. The `mtp-fast` profile sets `16384`. |
| `KV_DTYPE` | `auto` | Use `fp8_e4m3` for the validated 1M recipes. See [FP8 KV cache](HOW-IT-WORKS.md#fp8-kv-cache-on-the-qsa-path-opt-in) for the implementation and earlier trade-offs. |
| `PREWARM` | `0` | `1` streams the 48 GiB table once at boot to warm the page cache — steadier first-request latency, ~10 s extra startup. |
| `WORKERS` | `32` | Threads used for the mmap gather: every gather at the default `FAST_ROWS=0`; with `FAST_ROWS=512`, only gathers above 512 unique rows (decode-sized gathers then run inline). |
| `COMPILE_CACHE` | | Keep vLLM's, FlashInfer's, Triton's and the CUDA driver's caches across boots — this script recreates the container every run, so by default they are rebuilt each time. `<name>` = four docker volumes, `/abs/path` = four bind mounts. **−37 s of startup** (init engine −26 s) per boot after the first, ~175 MB of disk; only worth setting if something recreates the container for you (a model-swapping proxy, CI, tournament runs, a systemd unit). See [above](#optional-persistent-compile-cache-compile_cache). |
| `LOG_REQUESTS` | `0` | `1` logs every prompt and output (`--enable-log-requests --enable-log-outputs`, with DEBUG on vLLM's request logger only) so `tools/vllm_watch.py` can show sessions live. Debugging only: it puts user content in the Docker log, unbounded. |
| `PROM_MULTIPROC` | `0` | `1` runs prometheus_client in multiprocess mode so engine-side metrics (`vllm:ple_mmap_*`) reach `/metrics`. Opt-in, because it stops vLLM exporting its `*_created` samples and the `process_*` / `python_*` metrics (`process_start_time_seconds` included; `vllm:ple_mmap_engine_start_time_seconds` stands in as a restart marker); see *Watching the mmapped table* below. |
| `KV_CACHE_MEM` | | Passed through as `--kv-cache-memory-bytes`. `GPU_MEM` is a fraction of *total* device memory, so it leaves whatever was already resident on the table; vLLM prints the exact figure it would accept at startup ("Replace gpu_memory_utilization config with `--kv-cache-memory=...`"). On a Spark that headroom is also what the page cache uses for the PLE table, so taking it is a trade, not free memory — watch `vllm:ple_mmap_gather_seconds_total` when you do. |
| `EXTRA` | | Extra vLLM flags, passed verbatim — e.g. `--long-prefill-token-threshold 1024` for multi-client responsiveness (see [the concurrency section](#decoding-clients-stall-while-other-clients-prefill-the-long-prefill-token-threshold-slider)), `--api-key <secret>`. |
| `DOCKER_EXTRA` | | Extra `docker run` options, e.g. `-v /srv/certs:/certs:ro -e TRANSFORMERS_OFFLINE=1`. Whitespace separates arguments; embedded quotes, shell substitutions and wildcards are not evaluated. Paths and values must not contain whitespace. Shown in `DRY=1 ./flash serve` output. |

### API connection and TLS

`./flash wait`, `./flash status` and `./flash test` read the running container's
API arguments and published port. They select HTTPS when `--ssl-certfile` is set,
even if the current shell has a different `PORT`. Select another container with
`NAME=...`; an explicit `FLASH_API_BASE=https://hostname:port` overrides discovery
(for example, host networking or a hostname required by the certificate).

TLS certificates are verified by default. For a private/self-signed CA, use
`FLASH_CA=/path/to/ca.pem` on the **host**. `CURL_CA_BUNDLE`, `CURL_CAINFO` and
`SSL_CERT_FILE` are also supported, in that order after `FLASH_CA`. These settings
apply to both curl and Python requests. For a local test with an untrusted
certificate, `FLASH_INSECURE=1` explicitly disables certificate verification.

```bash
FLASH_CA=/srv/certs/ca.pem FLASH_API_BASE=https://spark.example:8000 ./flash wait
FLASH_CA=/srv/certs/ca.pem FLASH_API_BASE=https://spark.example:8000 ./flash test
# Explicit opt-out for a local self-signed endpoint:
FLASH_INSECURE=1 ./flash status
# Direct smoke tests accept a full URL; host:port still defaults to HTTP:
FLASH_CA=/srv/certs/ca.pem scripts/smoke-test.sh https://spark.example:8000
```

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
| --- | --- | --- | --- | --- |
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
[docs/HISTORY.md](HISTORY.md#persistent-compile-cache-on-the-preview-base-pr-21).

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
| --- | --- | --- |
| Weights on card | 77.83 GiB | **74.75 GiB** (−3.1) |
| KV cache @0.80 | 625,669 tok | **734,292 tok (+17%)** |
| Max concurrency @262k | 2.39x | **2.80x** |
| Decode, 5-prompt greedy probe | 26.6–33.8 tok/s | **34.7–42.5 tok/s (+20–27%)** |
| Mean acceptance length | ~2.73 | ~2.58 (same band) |
| Tournament quality | 45/51 | same target model, byte-identical greedy outputs (see below) |

Those are the author's numbers. Ours, measured the way defaults are decided here (same
box, same day as the table in [reduced draft vocabulary](HOW-IT-WORKS.md#the-mtp-drafters-vocabulary-and-madv_random-on-the-table),
full-vocabulary draft on both arms):

| | `MODE=hybrid` | `MODE=hybrid-mtp` |
| --- | --- | --- |
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
| --- | --- | --- | --- | --- | --- |
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

## Benchmarking

```bash
python3 bench/perf.py
python3 bench/longctx.py
./scripts/ab-bench.sh
python3 bench/ab_report.py results/ab/*.jsonl --baseline mtp4-1m
```

The A/B harness temporarily stops the serving container and measures the arms
in alternating order. It restarts the original container and waits for
readiness at exit. Run it during a maintenance window. `--keep` leaves the last
arm running after a successful run. See the
[1M comparison](../README.md#choosing-a-1m-context-profile).

Use `GB10_BASE_URL`, `GB10_API_KEY` and `GB10_MODEL` to configure the benchmark
client. Report cold prefill separately from cached-prefix latency. Record the
checkpoint, serving flags, cache budget, sampling parameters and concurrency
for each arm.

## Credits

- DFlash profile integration: **[@roozich](https://github.com/roozich)** ([PR #62](https://github.com/blazux/qwen3.8-Flash-DGX/pull/62)).
- The 1M profiles and A/B harness: **[@aphex3k](https://github.com/aphex3k)** ([PR #66](https://github.com/blazux/qwen3.8-Flash-DGX/pull/66)).

- Download knobs (Xet, `EXCLUDE`, `MAX_WORKERS`), the `vllm:ple_mmap_*` Prometheus counters and `KV_CACHE_MEM`:
  **[@techfury90](https://github.com/techfury90)** (PR #19); also spotted the upstream fix for NVIDIA's fp8 MTP experts (vllm#55513).
- `COMPILE_CACHE`, the persistent compile cache: **[@AronRubin](https://github.com/AronRubin)** (PR #21).
- The pointer to NVIDIA's own NVFP4 checkpoint: **[@PathosEthosLogos](https://github.com/PathosEthosLogos)** (issue #17).
- `tools/vllm_watch.py`, the live session viewer: **[@0x3dlux](https://github.com/0x3dlux)** (issue #12).
- Qwen token-identity and Responses output-integrity fixes (patch 19), and schema-validated tool-argument repair (patch 21), with source bundles and regression fixtures: **[@Chaos2522](https://github.com/Chaos2522)** ([original fixes](https://github.com/blazux/qwen3.8-Flash-DGX/issues/31#issuecomment-6191209070), [schema repair](https://github.com/blazux/qwen3.8-Flash-DGX/issues/31#issuecomment-6291191080)); adapted here for vLLM v0.31. Thank you for the detailed reproductions and follow-up work.
- MTP/routed-expert draft-config diagnosis and patch (patch 20): **[@rp-tn](https://github.com/rp-tn)** ([issue #55](https://github.com/blazux/qwen3.8-Flash-DGX/issues/55)) — thank you for sharing the fix.
- Chat tool-name compatibility regression report: **[@FinSnow](https://github.com/FinSnow)** ([issue #56](https://github.com/blazux/qwen3.8-Flash-DGX/issues/56)).
- Long-context MTP crash report: **[@FinSnow](https://github.com/FinSnow)** ([issue #60](https://github.com/blazux/qwen3.8-Flash-DGX/issues/60)); M-RoPE served-window fix: **[@wtogami](https://github.com/wtogami)** ([vllm#62409](https://github.com/vllm-project/vllm/pull/62409)), adapted as patch 22 to preserve the existing YaRN frequency base and cache rows.
- CPU pinning, BF16 recurrent-state control, the `concurrent` profile and benchmark suite: **[@aphex3k](https://github.com/aphex3k)** ([PR #54](https://github.com/blazux/qwen3.8-Flash-DGX/pull/54)).
- The vLLM v0.31 port and contributed GB10 measurements: **[@mercant1982](https://github.com/mercant1982)** ([PR #49](https://github.com/blazux/qwen3.8-Flash-DGX/pull/49)).
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
- Serving engine and base image: **vLLM** (`vllm/vllm-openai:v0.33.0`; the model's support started as
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
  [LICENSE](../LICENSE) (Apache-2.0).
