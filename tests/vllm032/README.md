# vLLM compatibility validation

The migration keeps the existing checkpoint and serving-profile settings.
[VLLM-032.md](../../docs/VLLM-032.md) records the port decisions, GPU probes,
historical performance comparison and validation limits. The independent
[issue #59 audit](https://github.com/blazux/qwen3.8-Flash-DGX/issues/59) checks
patch application; the constructor and dispatch tests also check runtime
interfaces that patch application alone does not exercise.

`PORT-MANIFEST.json` records the nine installed parser/Responses file hashes.
The schema runner accepts `--manifest` so the v0.31 provenance stays intact.
The other suites check the mmap constructor, blockwise-FP8 QSA dispatch,
MTP auxiliary outputs, native Responses event names and final item IDs.
Patch 22 adds ten installed-code M-RoPE constructor/factory tests for served-window
coverage and bit-exact preservation of existing rows. Its provenance and source
hashes are recorded separately in `MROPE-MANIFEST.json`.

Build the recipe and run the installed-code CPU checks on the supported ARM64
host. Mount only `tests/`, not the repository's `src/`, so the compatibility
tests import the modules installed in the image. Provide local compatible Qwen
tokenizer metadata; no weights, GPU, network or running inference server are used.

```bash
docker build -t qwen38-flash-dgx:v0.33 .
docker run --rm --network none --runtime runc --entrypoint python3 \
  -e VLLM_TARGET_DEVICE=cpu -e CUDA_VISIBLE_DEVICES= \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e QWEN_TOKENIZER=/tokenizer \
  -v "$PWD/tests:/tests:ro" \
  -v /path/to/local/tokenizer:/tokenizer:ro \
  qwen38-flash-dgx:v0.33 /tests/vllm032/run_cpu_qa.py
```

The schema author's dotted-namespace case remains a known failure: **22/23**.
Its runner checks that the failure is exactly the documented one and rejects
additional failures, errors or skips. All other suites in this runner must pass.
The separate historical upstream parser-engine comparison is **5,913/5,914**
on both baseline and candidate; its malformed-opener failure remains recorded
in `validation/`. This runner does not rerun that full upstream suite.

The migration-only rebuild and seven-suite rerun on 2026-10-22 are recorded in
[`validation/review-20261022/summary.json`](validation/review-20261022/summary.json),
with per-suite diagnostics and build logs beside it. The earlier GPU results
remain in `validation/VALIDATION.json`; this CPU rerun did not generate tokens.

The live probes require a separately started candidate. They generate tokens
but do not execute proposed tools. A synthetic long-context retrieval and
protocol checks do not establish agentic tournament quality.

For issue #60, use an isolated candidate: an unpatched engine can crash during
this probe. The script calibrates actual input tokens through `/tokenize`, runs
cold and replay requests across the native boundary, checks the retrieved code,
requires prefix reuse on replay, and follows each long request with health and a
short generation. JSON evidence is saved after each step, including failures.

```bash
python3 tests/vllm032/probe_mrope_boundary.py \
  --base-url http://127.0.0.1:18360 \
  --sizes 260000,263000,336000,499000 --repeats 2 \
  --output /tmp/mrope-boundary.json
```

The CPU regression is `/tests/vllm032/test_mrope_window.py` with the same image
and tests mount as the runner above; no tokenizer is needed for that test alone.

The patch-22 validation is recorded in
[`validation/issue60-20261022/VALIDATION.json`](validation/issue60-20261022/VALIDATION.json).
It includes the eight-suite CPU rerun and four cold/replay GPU retrievals per
arm. The unpatched reference did not crash on those requests; its draft cache
was nevertheless observed to have only 262,144 rows for a 500k served window.
The corrected cache has 500,000 rows, with the original YaRN base preserved.
