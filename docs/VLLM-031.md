# Evaluating vLLM 0.31 on GB10

The current recipe uses the official v0.31.0 image, with the existing model/GB10
patches. The model layouts and profiles are unchanged; no checkpoint conversion
is required when upgrading from v0.30.

## GX10 validation (2026-10-07 to 2026-10-08)

The maintainer compared v0.30 and v0.31 on the same ASUS GX10, using the NVIDIA
checkpoint revision `fc694b54fb0174e0913e6adf86691ef85a4ead47` in hybrid mode.
Both versions used MTP=2, deterministic top-k, the reduced draft vocabulary,
prefix caching, YaRN with a 500,000-token context, eight sequence slots and
`fp8_e4m3` KV caching with an explicit 8 GiB pool. No PyTorch allocator override
was used. Other serving profiles and checkpoints were outside this comparison.

### Agentic quality

Each version completed three passes of a 55-scenario agentic tournament, with
four concurrent scenarios, temperature 0.2 and a 32,000-token generation cap.

| Result | v0.30 | v0.31 |
| --- | ---: | ---: |
| Mean score | 87.53% | 90.84% |
| Scores per pass | 89.0%, 87.3%, 86.3% | 92.7%, 90.8%, 89.0% |
| Generation-cap hits | 19 / 165 | 15 / 165 |
| Request errors | 0 | 0 |

The observed difference was +3.31 percentage points. The paired scenario
bootstrap's 95% interval included zero (approximately 0 to +6.97 points), and
the run-level permutation test gave p=0.20. The results show no quality drop
in this sample; they do not establish a statistically significant improvement.
Long reasoning sometimes reached the generation cap on both releases.

### Serving and long context

- Both versions retrieved the planted information in two synthetic prompts
  with **499,000 input tokens**, counted with the server's chat tokenizer.
- Prefix-cache hits, sequential greedy determinism and the Anthropic-compatible
  `/v1/messages` endpoint with high reasoning effort passed.
- Single-stream decode was close: approximately 37–39 output tok/s on v0.30
  versus 37–38 on v0.31. Four-client aggregate throughput was 107.7 versus
  104.4 output tok/s, about 3% lower on v0.31 in that measurement.
- A v0.30/v0.31/v0.30 prefill recheck at approximately 8k, 32k and 128k tokens
  did not reproduce the initially suspected large prefill slowdown.
- All 173 Qwen parser cases in the validation suite, the PLE mmap and Exact-Top-k
  CPU checks, and four upstream native FP8-QSA GPU correctness cases passed.
- Repeated v0.31 starts succeeded with the standard allocator settings.

### Automatic KV sizing investigation (#51)

The FP8 comparison above used `KV_CACHE_MEM=8589934592`, which skips automatic
memory-budget profiling. It validates serving with that fixed pool; it does
not measure the automatically selected KV capacity. The lower automatic
capacity reported in [issue #51](https://github.com/blazux/qwen3.8-Flash-DGX/issues/51)
remains under investigation.

On an integrated GPU, vLLM's `MemorySnapshot` uses Linux `MemAvailable` for its
free-memory reading. Its `total_consumed` is the difference between startup
and post-profile available memory. The startup label "weights + non-torch"
therefore includes changes in shared host memory; it does not identify which
component allocated the difference. Both v0.30 and v0.31 use this accounting.
See vLLM's [`MemorySnapshot` and `memory_profiling` implementation](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/utils/mem_utils.py).
The native QSA FP8 kernel dequantizes tiles in the kernel and its Python split-K
scratch buffers are PyTorch allocations. A persistent non-torch dequantization
workspace has not been established as the cause; see the
[native QSA implementation](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/models/qwen4_exp/nvidia/ops/qsa.py).

The CUDA-graph estimate is a separate deduction from the automatic budget.
Turning it off can increase the KV pool, but also removes its memory allowance;
it does not demonstrate that the underlying consumption delta has disappeared.
`KV_CACHE_MEM` likewise sets a manual pool rather than reclaiming memory.

#### Optional diagnostic images

Build a logging-only derivative of each **existing, locally built** recipe image:

```bash
docker build -f tools/Dockerfile.memory-diagnostics \
  --build-arg BASE=qwen38-flash-dgx:v0.30 \
  -t qwen38-flash-dgx:v0.30-memory-diag .
docker build -f tools/Dockerfile.memory-diagnostics \
  --build-arg BASE=qwen38-flash-dgx:v0.31 \
  -t qwen38-flash-dgx:v0.31-memory-diag .
```

These derivatives add startup logging without changing allocation or KV-budget
logic. They do not rebuild vLLM or download model weights. They use the same
diagnostic hook on both releases and fail the build if its source anchors differ.

On a test host, repeat the original launch command with only `IMAGE` changed to
the corresponding diagnostic tag. Leave `KV_CACHE_MEM` unset to exercise
automatic sizing. Keep the checkpoint, launch arguments, allocator settings,
prewarm setting, graph estimator and other services consistent across boots.
Collect the startup logs before sending inference requests:

```bash
# Substitute the container name used in the original launch command.
docker logs qwen38-flash > startup-memory.log 2>&1
grep 'qwen38-memory' startup-memory.log > startup-memory-records.log
```

Each `qwen38-memory` record contains JSON with raw CUDA free memory, the free
memory used by vLLM, PyTorch allocated/reserved/peak bytes, Linux memory fields,
the current process's RSS/PSS, call sites and package versions. The final
`profile_result` record includes the inputs to the non-KV budget calculation.
These diagnostic records contain no prompts or model outputs.

For #51, provide records from both versions, their image IDs and driver version,
and the startup summary lines for consumed memory, activation peak, graph
estimate/actual usage and available KV. These measurements distinguish changes
in host memory, persistent PyTorch allocations and transient profiling overhead
before selecting an allocator or kernel change.

## Build and run

```bash
./flash setup
./flash serve
./flash wait
./flash test
```

For a direct build:

```bash
docker build -t qwen38-flash-dgx:v0.31 .
IMAGE=qwen38-flash-dgx:v0.31 MODE=hybrid YARN=1 CTX=500000 scripts/serve.sh
```

Switching servers interrupts active requests. Preserve the old image and its
launch configuration locally before upgrading. The current wrapper accepts
only the current base; an old image alone does not preserve launch settings.
For a preserved, stopped container named `qwen38-flash-rollback-v030`:

```bash
docker stop qwen38-flash
docker rename qwen38-flash qwen38-flash-saved-v031
docker rename qwen38-flash-rollback-v030 qwen38-flash
docker start qwen38-flash
./flash wait
```

Names are examples; verify your saved container first and do not overwrite an
existing name. This returns to the original container's command and environment.

## Patch compatibility

- Patch 7 is omitted on v0.31: native QSA FP8 main-KV support landed upstream
  ([vllm#55557](https://github.com/vllm-project/vllm/pull/55557)).
- Patch 16 is omitted on v0.31: indexed expert mapping is upstream
  ([vllm#58720](https://github.com/vllm-project/vllm/pull/58720)).
- Patch 14 locates the final ordinary copy in `_load_w13` and `_load_w2` by
  Python syntax. This preserves v0.31's additional chunked-copy branch for
  noncontiguous tensor-parallel weights.
- The remaining patches are retained. Patch anchors still fail the build when
  upstream changes unexpectedly; no patch is silently skipped.
- Parser patches 12/13 use context shared by both releases; v0.31 added array
  state initialization immediately after the old patch anchor. Their behavior
  is unchanged. The two upstream Qwen parser test modules, with this repo's
  test patches, passed all 122 cases on the candidate image.

## Contributor measurements (2026-10-06)

These Lenovo GB10 measurements used BF16 KV caching and a different driver.
They document the initial upgrade evaluation; the GX10 comparison above uses
the FP8 configuration described there.

Host: Lenovo GB10, ARM64, 121.6 GiB unified host memory, NVIDIA driver 595.84,
CUDA driver 13.2, NVMe storage. Same NVIDIA checkpoint revision
`fc694b54fb0174e0913e6adf86691ef85a4ead47`, hybrid layout, context 500,000,
YARN=1, MTP=2, draft vocabulary 65,536, deterministic top-k, BF16 KV,
GPU memory utilization 0.80, maximum eight sequences, prefix caching enabled.
The v0.31 run used `PYTORCH_ALLOC_CONF=expandable_segments:True`; v0.30 did not.

Ten-minute client measurements, medians:

| Workload | v0.30 TTFT (s) | v0.31 TTFT (s) | v0.30 decode (tok/s) | v0.31 decode (tok/s) |
| --- | ---: | ---: | ---: | ---: |
| ~8k fresh | 2.961 | 2.673 | 28.83 | 38.51 |
| ~8k repeated | 1.186 | 1.134 | 33.41 | 39.13 |
| ~31k fresh | 11.045 | 10.201 | 26.89 | 37.40 |
| ~31k repeated | 1.252 | 1.055 | 28.67 | 37.81 |

Four-client aggregate throughput: 104.54 versus 126.21 output tok/s.
The v0.30 run recorded 42 sequential requests and 10 concurrent batches over
604.4 seconds; v0.31 recorded 51 sequential requests and 12 concurrent batches
over 606.8 seconds. Fresh/repeated complete answers matched in 7/21 pairs on
v0.30 and 25/25 on v0.31. This is a sample observation, not a proof of general
determinism or model quality. Different answer lengths/wording and speculative
acceptance can affect throughput.

Important limitation: the first v0.30 run overlapped the v0.31 image download
and extraction. It is not an isolated engine-version A/B experiment. Additional
120-second runs with seed 9100, without downloads or builds, are stored beside
the full runs. Other local services remained running; requests to the unsupported
embedding endpoint were visible during measurements. No private/user prompts
were used. Raw synthetic requests' answers and measurements are in
[`../results/`](../results/).

Matching only the nine sequential request keys present in both 120-second
control runs (seed 9100, same cycle/size/phase), the medians were:

| Workload | Samples per version | v0.30 decode (tok/s) | v0.31 decode (tok/s) |
| --- | ---: | ---: | ---: |
| ~8k fresh | 3 | 28.90 | 38.81 |
| ~8k repeated | 2 | 28.39 | 40.09 |
| ~31k fresh | 2 | 29.48 | 36.91 |
| ~31k repeated | 2 | 26.91 | 37.57 |

This small, less-confounded check supports the observed decode improvement
(approximately 25–41%), but is not a statistically powered quality/performance
study. The two versions can produce different output text even on the same
input; only the prompt/configuration is matched, not the decode token sequence.

The serving smoke test passed on both versions: coherent response,
`/v1/messages` with high reasoning effort, prefix-cache hit and identical
first-token logprobs. Its single 400-token decode sample improved from 34.2 to
37.1 end-to-end tok/s. The first v0.31 smoke prefill was slower (11.42 versus
3.76 seconds); repeated benchmark prefills above did not reproduce that result.
The subsequent GX10 validation above adds the agentic tournament,
499k-token retrieval probes and native FP8-QSA GPU correctness checks.

CPU checks on the final image also passed: synthetic PLE mmap gathers, the
placeholder/zero/out-of-range/prewarm paths, and Exact-Top-k against `torch.topk`.

## Reproducible serving benchmark

Run against an already healthy server:

```bash
python3 tools/bench_serving.py --seconds 600 --label v0.30.0 --output results/gb10-v030.jsonl
# Switch the server, wait for readiness, and use the same seed and settings:
python3 tools/bench_serving.py --seconds 600 --label v0.31.0 --output results/gb10-v031.jsonl
```

The standard-library client streams chat completions. Each cycle tests a seeded
synthetic prompt of approximately 8k and 31k tokens, then immediately repeats it
to exercise prefix caching. It records client-observed time to first text,
decode throughput from the first to the last nonempty chunk, token usage, and
whether the complete text matches. The four-client test uses the same short
prompt concurrently and reports aggregate end-to-end output throughput.
Generation stops at EOS or 256 output tokens; `ignore_eos` is never used.

The duration bounds when new work starts; an in-flight request may finish after
the deadline. There is no request-rate control or server-cache reset. Fresh
prefixes do not imply a cold operating-system page cache. Stream chunks may
contain multiple MTP tokens, so decode rate is a client estimate.

Use an idle host for a controlled comparison. Image downloads, builds, other
clients, swap and page-cache pressure can affect results. This is a performance
and serving check, not the repository's agentic quality tournament or a
500k-context correctness test.

## Initial GB10 startup observation

On a Lenovo GB10 host with driver 595.84, the first v0.31 attempt loaded the
hybrid model and captured CUDA graphs, then remained inside
`vllm/v1/worker/utils.py::allocate_kv_cache` at its large `torch.zeros` call.
The API never became ready. A nonblocking Python stack dump located the stall;
the process was stopped and GPU queries recovered. An isolated allocation of
16,600 MiB in the unmodified official v0.31 image subsequently completed in
0.505 seconds. This does not establish a general vLLM or driver bug.

An opt-in second trial uses PyTorch's documented expandable-segment allocator:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True ./flash serve
```

The second trial became ready after 277 seconds with 590,909 KV-cache tokens
(the v0.30 run reported 615,151). It passed the serving smoke test. This is an
observed successful workaround, not proof of the first stall's cause.

The cleaned, v0.31-only Dockerfile was rebuilt and started again with the same
allocator setting. It became ready after 236 seconds with 525,757 KV-cache
tokens. Cache capacity varied across starts on this shared unified-memory host;
do not assume a fixed 590k/615k-token capacity or multiple full-500k requests.
The running container's image ID matched the final build.
The smoke test passed again: first fresh 10,665-token prefill 8.32 seconds,
same-prefix repeat 0.93 seconds, identical first-token logprobs, and 400-token
answer at 37.1 end-to-end tok/s. The slower first-prefill observations are a
real caveat; the measured advantage is chiefly steady-state decode, not every
possible cold-start/prefill workload.

`serve.sh` forwards `PYTORCH_ALLOC_CONF` only when set. No allocator override is
imposed by default. See [PyTorch memory management](https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-alloc-conf).
