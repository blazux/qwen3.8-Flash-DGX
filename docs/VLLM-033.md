# vLLM 0.33 on DGX Spark

The image uses the official CUDA 13 vLLM 0.33 build. The local patches retain
the NVMe-backed PLE table, FP8 hybrid projections, DFlash target integration,
long-context M-RoPE bounds and Qwen tool parsing. The parser port keeps native
incremental Chat tool arguments and the existing Responses event handling.
Existing prepared hybrid checkpoints can be reused. [Upstream release notes](https://github.com/vllm-project/vllm/releases/tag/v0.33.0rc3)

```bash
./flash setup
./flash serve
```

Use your existing profile name and overrides to recreate the container. Keep
the previous container and image available if you need to restore 0.32.

## What changed in serving

The comparison used one GB10, the same NVIDIA NVFP4 hybrid checkpoint and
template, 500k context, a 13 GiB FP8 KV cache, BF16 recurrent state and 16
sequences. Cold prefill used distinct prefixes and one output token, with three
repetitions per size. Concurrent decode is aggregate output throughput. The
warm-prefix probe is a separate measurement.

| Workload | 0.32 MTP3 | 0.33 MTP3 | 0.33 `mtp-fast` |
| --- | ---: | ---: | ---: |
| Cold prefill, 8k (tok/s) | 3,153 | 3,467 | 3,516 |
| Cold prefill, 32k (tok/s) | 3,182 | 3,333 | 3,459 |
| Cold prefill, 100k (tok/s) | 3,120 | 3,219 | 3,392 |
| Sampled decode, solo (tok/s) | 35.1 | 36.9 | 42.6 |
| Sampled decode, 8 concurrent (tok/s) | 111.6 | 118.0 | 128.3 |

DFlash K4 reached the first token in 0.20 s with a reused 7.7k-token prefix,
compared with 2.35 s on 0.32 in the same probe. That benefit depends on prefix
reuse. Sampled prose throughput varied with the prompt and sampling parameters.
Both DFlash builds passed the tool and Responses probes, vision smoke tests
and retrieval checks at 263k and 499k tokens.

The 0.33 recipe also supports quantizing the MTP proposal head without
quantizing the verifier head. This option requires Model Runner V2 and the full
draft vocabulary. [Upstream proposal-head quantization](https://github.com/vllm-project/vllm/pull/59973)

## MTP speed profile

`mtp-fast` is also selected by `./flash serve` and `./flash serve default`.
Use `./flash serve mtp-standard` for the previous MTP2 recipe.

```bash
./flash setup mtp-fast
./flash serve mtp-fast
```

This profile uses three speculative tokens, the NVFP4 proposal head, and up
to 16,384 scheduled tokens per step. It uses the NVIDIA hybrid checkpoint with
500k context, BF16 recurrent state and a 13 GiB FP8 KV cache. The larger prefill
batch increases peak memory use. On a more constrained host, use
`BATCH_TOKENS=8192` to reduce the batch budget. The engine quantizes a copy of
the proposal head at startup, so existing checkpoints are reusable.

The profile passed the Chat tools, Responses and vision smoke checks, plus
retrieval at 263k and 499k tokens.
