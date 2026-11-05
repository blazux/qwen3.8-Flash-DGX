# Qwen3.8-Flash-Next on DGX Spark

Run Qwen3.8-Flash-Next on one NVIDIA DGX Spark or ASUS GX10 with **vLLM 0.33**.
The default profile, **`mtp-fast`**, serves the NVIDIA hybrid checkpoint at up
to **500k tokens**. Optional profiles support **1M tokens**. The image reads the
large PLE embedding table from NVMe via `mmap` to leave memory for the KV cache.

## Quickstart

You need a GB10 system with 128 GB of unified memory, Docker with the NVIDIA
runtime, and roughly 140 GB of free disk for the weights plus 13 GB for the
hybrid layout. Use NVMe for the weight cache.

```bash
git clone https://github.com/blazux/qwen3.8-Flash-DGX.git
cd qwen3.8-Flash-DGX
./flash doctor
./flash setup
./flash serve
./flash wait
./flash test
```

`setup` builds the image, downloads the weights and prepares the hybrid layout.
The download is resumable. Startup loads roughly 75 GiB of weights and typically
takes a few minutes.

The OpenAI-compatible API defaults to `http://localhost:18300/v1` with the
model name `qwen3.8-flash-next`.

```bash
curl http://localhost:18300/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-flash-next","messages":[{"role":"user","content":"Explain prefix caching in two sentences."}],"max_tokens":512}'
```

The server listens on all interfaces without authentication by default. On a
shared network, use an authenticated gateway or set vLLM's `--api-key`. See
[API connection and TLS](docs/SERVING.md#api-connection-and-tls).

## Choose a profile

| What you need | Profile |
| --- | --- |
| General serving up to 500k tokens | `default` / `mtp-fast` |
| Faster greedy and sampled solo decode near 1M | `dflash-1m` |
| More shared KV capacity at 1M | `mtp4-1m` |
| Shorter interruptions to active streams during long prefills | `shared` |
| Previous MTP2 recipe, with reduced draft vocabulary | `mtp-standard` |

```bash
./flash profiles
./flash setup dflash-1m
./flash serve dflash-1m
```

Profiles share port `18300` by default. `./flash serve` recreates the named
container. Use `NAME` and `PORT` to select a separate instance if the host has
enough memory. Environment variables override the profile, and `KEY=VALUE`
arguments override both.

## vLLM v0.33 upgrade

**`mtp-fast` is the default profile on vLLM 0.33.** Running `./flash serve`,
`./flash serve default`, or `./flash serve mtp-fast` selects the same recipe.
It uses an NVFP4 copy of the MTP proposal head, three speculative tokens, a 16k
prefill batch, BF16 recurrent state, a 13 GiB FP8 KV cache, and a 500k context limit.

| GX10 workload | vLLM 0.32 MTP3 | vLLM 0.33 `mtp-fast` | Change |
| --- | ---: | ---: | ---: |
| Cold prefill at 8k | 3,153 tok/s | 3,516 tok/s | +11.5% |
| Cold prefill at 32k | 3,182 tok/s | 3,459 tok/s | +8.7% |
| Cold prefill at 100k | 3,120 tok/s | 3,392 tok/s | +8.7% |
| Sampled decode, one request | 35.1 tok/s | 42.6 tok/s | +21.3% |
| Sampled decode, eight concurrent requests | 111.6 tok/s | 128.3 tok/s | +15.0% |

These measurements used the same NVIDIA hybrid checkpoint on one GB10.
The profile passed tool calling, Responses, vision and retrieval checks at
263k and 499k tokens. The larger prefill batch uses more memory.
Use `./flash serve BATCH_TOKENS=8192` to reduce the batch budget, or
`./flash serve mtp-standard` for the previous MTP2 recipe on vLLM 0.33.
See the [port notes and workload measurements](docs/VLLM-033.md).

After updating the checkout, run `./flash setup`, then `./flash serve` to
recreate the container. Existing prepared hybrid checkpoints can be reused.

## Choosing a 1M-context profile

Choose `dflash-1m` for faster single-stream generation, especially greedy
decoding. Choose `mtp4-1m` when KV capacity matters more, for example when
other requests share the server with a long prompt. For workloads within
500k context, start with `mtp-fast`, the default profile.

| GX10 workload | `mtp4-1m` | `dflash-1m` | DFlash change |
| --- | ---: | ---: | ---: |
| KV capacity | 1.52M tokens | 1.11M tokens | −27% |
| Cold prefill, 8k | 3,132 tok/s | 2,751 tok/s | −12% |
| Cold prefill, 512k | 2,722 tok/s | 2,779 tok/s | +2% |
| Cold prefill, 900k | 2,385 tok/s | 2,438 tok/s | +2% |
| Greedy decode, one request | 36.6 tok/s | 46.4 tok/s | +27% |
| Sampled decode, one request, T=0.7 | 37.0 tok/s | 40.2 tok/s | +9% |
| Sampled throughput, eight requests, T=0.7 | 109.8 tok/s | 112.2 tok/s | +2% |

Both profiles passed retrieval at 990k tokens and a follow-up generation.
The matched run used vLLM 0.33, the NVIDIA hybrid checkpoint and 21 GiB of
fixed FP8 KV on one GX10. Prefill rates are medians of two cold runs per
size, so the small differences at 512k and 900k are indicative. Capacity
refers to the shared KV pool. The per-request limit is 1M tokens.
[PR #66](https://github.com/blazux/qwen3.8-Flash-DGX/pull/66)

## The `flash` command

| Command | Purpose |
| --- | --- |
| `./flash doctor [profile]` | Check the host, image, checkpoint and profile prerequisites |
| `./flash setup [profile]` | Build the image and prepare the selected checkpoint and drafter |
| `./flash serve [profile] [KEY=VALUE…]` | Recreate the container using the selected recipe |
| `./flash wait` and `./flash test` | Wait for the API and run a smoke test |
| `./flash status` and `./flash logs` | Inspect the server and follow its logs |
| `./flash stop` and `./flash start` | Stop or restart the existing container |

For low-memory starts, first check `./flash doctor` and the engine log.
`./flash serve BATCH_TOKENS=8192` lowers the default batch budget.
With `published`, lower `CTX` if the engine reports a smaller KV ceiling.

## Recent changes

| Date | Change |
| --- | --- |
| 2026-11-04 | **vLLM 0.33 and `mtp-fast` by default.** [Upgrade and measurements](#vllm-v033-upgrade). |
| 2026-11-04 | **1M-context profiles.** `dflash-1m` and `mtp4-1m` passed retrieval at 990k. [Results and profile choice](#choosing-a-1m-context-profile). |

## Documentation

- [Serving reference](docs/SERVING.md) for the full profile list, tuning, TLS, diagnostics and benchmarks.
- [vLLM 0.33](docs/VLLM-033.md) for upgrade details and measured workloads.
- [How it works](docs/HOW-IT-WORKS.md) for PLE offload, hybrid weights and GB10 patches.
- [Tool calling and Responses](docs/SAFE-OUTPUT.md) for parser behavior and validation.
- [Project history](docs/HISTORY.md) for previous releases and benchmarks.
- [Archived README](docs/README-ARCHIVE.md) for the full documentation before the rewrite.
- [Credits](docs/SERVING.md#credits) and [Apache-2.0 license](LICENSE). The model weights have their own license.
