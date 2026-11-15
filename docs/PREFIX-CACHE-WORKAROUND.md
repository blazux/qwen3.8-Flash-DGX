# Prefix-cache checkpoint workaround

This opt-in workaround targets the pinned **vLLM 0.33.0** image. It avoids the
internal recurrent-state prefill checkpoint path while retaining prefix reuse,
MTP and the configured weight and KV formats.

[Issue #73](https://github.com/blazux/qwen3.8-Flash-DGX/issues/73) reports different
greedy output and token probabilities between fresh requests and cache hits.
The implementation follows the guarded workaround contributed by
[@jtbrown6](https://github.com/jtbrown6/qwen3.8-Flash-DGX).
The exact numerical cause remains under investigation.

## Build and enable

```bash
./flash setup prefix-cache-safe
./flash serve prefix-cache-safe
./flash wait
./flash test
```

`serve` replaces the named container and reloads the model. The profile uses
the NVIDIA hybrid checkpoint, **262144 context tokens, YaRN off**, MTP3, BF16
recurrent state, 13 GiB FP8 KV, prefix caching and a loopback API binding.
Its image tag is `qwen38-flash-dgx:v0.33-prefix-safe`.
If a gateway connects through another interface, select a `PORT` binding it can
reach; see the [serving reference](SERVING.md).

The reported NVIDIA tests used revision
`fc694b54fb0174e0913e6adf86691ef85a4ead47` with hybrid preparation. Setup can
resolve a newer checkpoint; check the selected snapshot when reproducing those
results and repeat the probes for a different revision.

`SPLIT_PREFILL=1` enables the workaround. The launcher requires the image label
`dgx.prefix-boundaries=opt-in-v1`, then passes `VLLM_DGX_SPLIT_PREFILL=1` into
the container. The image label records a successful guarded build; setting an
environment variable alone on an older image does not install the patch.

The contributor reported native-context validation on one DGX Spark in
[issue #73](https://github.com/blazux/qwen3.8-Flash-DGX/issues/73#issuecomment-6884205051).
Our separate [GX10 500k/YaRN observations](validation/PREFIX-CACHE-GX10-20261115.md)
passed exact solo comparisons through **450000 prompt tokens**. Concurrent waves
still differed, including independent fresh controls. A mixed 500k workload also
reached the test's available-memory guard; the 13 GiB KV budget does not cap total
RAM use. To select the 500k override:

```bash
./flash serve prefix-cache-safe CTX=500000 YARN=1
```

Check the printed configuration before launch. Environment variables override
profile values, and `KEY=VALUE` arguments take precedence over both.

To return to the standard `mtp-fast` recipe, recreate it with the workaround off:

```bash
./flash serve mtp-fast SPLIT_PREFILL=0
```

If you previously used custom overrides, supply those again. Selecting another
profile recreates the named container; it does not recover earlier overrides.

## Check fresh requests against real cache hits

Use the **direct vLLM API**, including when applications normally use LiteLLM.
The probe needs `/tokenize`, `/v1/completions`, `cache_salt` and detailed cache
usage; gateways may hide or transform these fields. The default URL is
`http://127.0.0.1:18300`; change it with `--base-url` when needed.

```bash
python3 scripts/probe-prefix-cache.py --output results/cache-first-token
python3 scripts/probe-prefix-cache.py --output results/cache-continuations \
  --tokens 10000 20000 --max-tokens 64
python3 scripts/probe-prefix-cache.py --output results/cache-concurrent \
  --tokens 32768 --max-tokens 64 --concurrency 4 --timeout 600
```

Use a new output directory for each run. The probe always includes the original
synthetic reproduction prompt. Each case compares **cold → hit → repeat → fresh
independent control**, with salts separating the fresh controls from reuse.
Passing requires:

- Zero cached tokens for both fresh controls and positive reuse for both hits.
- Exact matching text, output tokens, finish reason and every reported generated
  token/top-token log probability.
- Complete, valid responses and usable cache accounting.

Responses, salts, timings and the summary are saved. A mismatch, malformed or
truncated response, missing probabilities or missing reuse produces a nonzero
exit status. Generation may end before the requested output budget at EOS.

Run without other inference traffic. `--concurrency` accepts 1, 2, 4 or 8;
each client has independent salts, and all clients finish a wave before the
next begins. Select prompt lengths that fit the shared KV pool: several large
requests can exceed capacity even when each fits the per-request context limit.
Comparisons retain text/token and probability differences separately, including
cold versus independent-fresh controls. Diverging fresh controls also indicate
instability beyond cache reuse.

Timings and output-token rates are nonstream **end-to-end** observations,
including queueing, prefill and generation. They are not TTFT or steady-state
decode rates. Compare latency and throughput as well as correctness: splitting
at cache boundaries can change prefill work.

## CPU checks and runtime upgrades

The tests use source fixtures and fake Docker commands; no GPU is required.

```bash
python3 tests/test_prefix_cache_patch.py
python3 tests/test_prefix_cache_probe.py
python3 tests/test_flash_launch.py
```

The patcher accepts only the exact original or already-patched source hashes
in `src/prefix_cache_boundaries/manifest.json`. It refuses unexpected source
before changing the file. Runtime upgrades require reviewing the capability
query together with its scheduler, cache-manager and layer callers, adapting
the guard, then repeating CPU and GPU probes on the intended configuration.

CPU checks alone do not establish GPU cache correctness. Native-context results
also do not establish 500k/YaRN behavior, concurrency or general answer quality.
