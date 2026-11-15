# NVIDIA prefix-cache workaround: 500k/YaRN observations

Across two separately reported protocols, the workaround restored exact solo cold/hit agreement in
**7/7 matched observations** through 450000 prompt tokens, versus **0/7** for
the baseline: four initial observations plus three on fresh engines. This is a
descriptive count, not seven independent semantic tasks. Concurrent results remain
mixed despite the completed isolated retake, including fresh controls.
The benefit supports the measured solo use cases; it does not establish batch invariance or a general quality gain.

The published `prefix-cache-safe` profile remains **262144 context / YaRN off**.
The 500000-context / YaRN configuration below is a separate qualification, not a
silent change to that profile. See the [setup and probe guide](../PREFIX-CACHE-WORKAROUND.md).

## Hardware and recipe

- One ASUS GX10 / NVIDIA GB10; driver 580.178.04.
- `nvidia/Qwen3.8-Flash-Next-NVFP4`, revision
  `fc694b54fb0174e0913e6adf86691ef85a4ead47`, locally converted hybrid checkpoint.
- Context 500000; YaRN factor 4, original context 262144; fixed 13 GiB FP8 E4M3
  KV cache; BF16 Mamba recurrent state; MTP3 with NVFP4 proposal head.
- Maximum 16384 batched tokens / 16 sequences; CPU affinity `5-9,15-19`.
  Other recipe settings were held fixed between arms; batch invariance was not enabled.
- Baseline: existing v0.33 image. Candidate: a small guarded overlay of that same
  image, applying the proposed patch. A full rebuild of the repository Dockerfile
  was **not** the GPU-tested artifact.

Test-container versions, read from package metadata:

| Package | Version |
| --- | --- |
| vLLM | 0.33.0 |
| PyTorch | 2.14.0+cu130 |
| flashinfer-python / flashinfer-cubin | 0.7.2.post1 |
| Triton | 3.8.0 |

Runtime SHA256 of `vllm/model_executor/layers/mamba/abstract.py`:

| Arm | `SPLIT_PREFILL` | Source SHA256 |
| --- | --- | --- |
| Baseline | 0 | `08506d9dc3e1a58d42a72cb80c81e32a51d4bda7f7be2551d355db19a0d02f9d` |
| Candidate | 1 | `bfd0a08d627ac0dc6604425f89474328abb71b43e79c67047673cdff858e0bae` |

## What was compared

The [probe](../../scripts/probe-prefix-cache.py) uses the direct vLLM API with
temperature 0, seed 42, `logprobs=5`, real cache accounting and independent client
salts. Its SHA256 is
`d871d8f8c62462f195e84dc1a93e6523fc228980185443f10e43bdb9591ad9f1`.
Each case runs
**cold → hit → repeat → independent fresh**. Both fresh waves must report zero
cached tokens; both hit waves must report positive reuse. Strict equality means
identical text, returned token entries, all reported generated/top-5 logprobs and
finish reason, with no tolerance. A probability mismatch remains a strict failure
even when the generated text agrees.

The original synthetic reproduction prompt has 10581 tokens and appears in every
group. Additional prompt targets were 32768, 131072, 270000 and 450000 tokens.
Short/medium/long groups cap outputs at 64/16/8 tokens respectively; EOS is honored.
These bounded synthetic continuations are not distinct semantic tasks; the configured context was 500000, but no 500000-token prompt was tested.

## Initial campaign: one engine boot per arm

Arms ran sequentially without dropping the OS cache. The candidate stopped before
completing its 4×131072 case; its long-solo group did not run. Do not count missing
observations as successes or pool them with the fresh-engine continuation below.

Strict matches / completed client-case comparisons:

| Scope | Arm | Cold vs hit | Hit vs repeat | Cold vs fresh |
| --- | --- | --- | --- | --- |
| Solo through 131072, four matched observations | Baseline | 0/4 | 4/4 | 3/4 |
| Same solo observations | Candidate | 4/4 | 4/4 | 4/4 |
| Concurrency 4, short: 10581 + 32768 | Baseline | 0/8 | 6/8 | 0/8 |
| Same concurrency 4 observations | Candidate | 3/8 | 5/8 | 2/8 |
| Concurrency 8, short: 10581 + 32768 | Baseline | 0/16 | 7/16 | 5/16 |
| Same concurrency 8 observations | Candidate | 3/16 | 7/16 | 3/16 |
| Concurrency 4, medium: 10581 + 131072 | Baseline | 0/8 | 6/8 | 0/8 |
| Concurrency 4, medium: **10581 only** | Candidate | 3/4 | 4/4 | 1/4 |
| Solo long: 10581 + 270000 + 450000 | Baseline only | 0/3 | 3/3 | 1/3 |

Text equality and full generated-token equality had the same counts in these
rows. The next table separates their count **T** from first-token equality **F**;
each cell is `T; F`, with the denominator shown in the scope. Neither is a strict pass.

| Scope (denominator) | Arm | Cold vs hit T; F | Hit vs repeat T; F | Cold vs fresh T; F |
| --- | --- | --- | --- | --- |
| Solo through 131072 (4) | Baseline | 2; 2 | 4; 4 | 4; 4 |
| Same solo (4) | Candidate | 4; 4 | 4; 4 | 4; 4 |
| Concurrency 4 short (8) | Baseline | 4; 4 | 6; 6 | 6; 6 |
| Same concurrency 4 (8) | Candidate | 7; 7 | 7; 7 | 8; 8 |
| Concurrency 8 short (16) | Baseline | 10; 10 | 15; 15 | 13; 13 |
| Same concurrency 8 (16) | Candidate | 15; 15 | 15; 15 | 16; 16 |
| Concurrency 4 medium, both cases (8) | Baseline | 4; 4 | 8; 8 | 6; 7 |
| Concurrency 4 medium, original only (4) | Candidate | 3; 3 | 4; 4 | 4; 4 |
| Solo long (3) | Baseline only | 2; 2 | 3; 3 | 3; 3 |

The original solo prompt is a useful baseline discriminator: its cold and fresh
outputs/probabilities agree, while its cold and hit first tokens differ. Under
concurrency, fresh controls also vary, so not every difference is cache-specific.
Probability diagnostics compare generated logprobs only over the identical output
prefix, and common returned top-5 entries only at the first generated position;
they do not compare the full vocabulary distribution.

### Guard stop and interpretation

During candidate 4×131072 after earlier workloads, available memory reached
3.28 GiB and remained below 4 GiB for 60 seconds, triggering the memory guard.
At the recorded incident the API returned HTTP 200, swap I/O was zero, and PSI
full avg10 was about 0.01%. This does not establish the source of memory pressure
or prove a leak. The fresh-engine continuation retained the same guards and recipe.

Reported KV capacity was 909509 baseline / 920807 candidate tokens at the same
13 GiB budget. That is an effective capacity estimate, not evidence of a larger
KV allocation: removing a reserved Mamba checkpoint changes the estimate's cost
per request. See [capacity calculation](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/core/kv_cache_utils.py#L1061)
and [fixed KV budget behavior](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/worker/gpu_worker.py#L641).

## Fresh-engine continuation: separate results

Heavy pressure tiers restarted the engine for each arm/tier, retaining the same
flags, output budgets and guards. All four planned NVIDIA groups completed;
coverage completion is not a strict correctness pass. Results remain separate
from the initial campaign. Jobs 03/04 cover solo-long; jobs 09/10 cover concurrency 4 medium:

| Arm / prompt set | Strict cold→hit / hit→repeat / cold→fresh | Text/token matches | First-token matches |
| --- | --- | --- | --- |
| Baseline; 10581, 270000, 450000 | 0/3 · 3/3 · 1/3 | 2/3 · 3/3 · 3/3 | 2/3 · 3/3 · 3/3 |
| Candidate; same set | 3/3 · 3/3 · 3/3 | 3/3 · 3/3 · 3/3 | 3/3 · 3/3 · 3/3 |
| Baseline; concurrency 4 medium | 0/8 · 5/8 · 0/8 | 5/8 · 8/8 · 6/8 | 5/8 · 8/8 · 6/8 |
| Candidate; concurrency 4 medium | 3/8 · 2/8 · 1/8 | 7/8 · 7/8 · 8/8 | 7/8 · 7/8 · 8/8 |

In the candidate 4×131072 case, strict cold/hit agreement was 0/4 despite 4/4
matching token sequences. The workaround therefore does not universally restore
concurrent exact agreement, and fresh-control variability remains measurable.

Paired long-request E2E latencies in seconds; every request below produced 8 tokens:

| Prompt tokens | Arm | Cold | Hit | Repeat | Fresh |
| --- | --- | --- | --- | --- | --- |
| 270000 | Baseline | 86.308 | 0.643 | 0.616 | 85.684 |
| 270000 | Candidate | 86.157 | 0.615 | 0.623 | 85.727 |
| 450000 | Baseline | 154.076 | 0.863 | 0.870 | 152.920 |
| 450000 | Candidate | 152.834 | 0.881 | 0.876 | 153.020 |

Minimum `MemAvailable` in GiB across approximately 5-second job samples, rounded to two decimals:

| Fresh-engine tier | Baseline | Candidate |
| --- | --- | --- |
| Solo long | 8.48 | 11.75 |
| Concurrency 4 medium | 7.68 | 8.51 |

These samples include startup, and are not CUDA allocations, continuous minima,
or proof of general memory savings. Completion with identical guards on isolated
fresh engines does not validate the earlier mixed-workload sequence or resolve
its guard stop. No memory-pressure cause or leak was established.

All timings include queueing, prefill and generation. They are nonstream E2E,
not TTFT or steady-state decode rates; differing output lengths prevent a clean
speedup comparison. These observations establish neither general answer quality,
all 500k workloads nor batch invariance. Upstream documents
[reproducibility limits](https://docs.vllm.ai/en/v0.33.0/usage/reproducibility/) and
[speculative-decoding guarantees](https://docs.vllm.ai/en/v0.33.0/features/speculative_decoding/#lossless-guarantees-of-speculative-decoding).

The [compact NVIDIA aggregates](prefix-cache-gx10-20261115.json) retain both
protocols separately, including the initial campaign's incomplete coverage.
They contain no raw requests, responses, salts or machine snapshots.
The recommendation is limited to the observations above. The workaround's exact numerical cause
remains unproved; [the official capability query](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/model_executor/layers/mamba/abstract.py#L98)
and its scheduler/layer callers explain which checkpoint path it bypasses.
