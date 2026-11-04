# Issue #31: parser and Responses regression tests

The historical v0.32 recipe and installed-source manifest are documented in
[VLLM-032.md](../../docs/VLLM-032.md). The v0.31 results below remain historical.

Port of [Chaos2522's review2 bundle](https://github.com/blazux/qwen3.8-Flash-DGX/issues/31#issuecomment-6191209070).
Archive provenance is recorded in `SOURCE.json`; patch-19 target hashes and the
v0.31 upstream commit are in `PORT-MANIFEST.json`. The subsequent patch 21 changes
five of those files and adds the schema helper; current installed hashes and the
additional runner are in the [schema-argument suite](../issue31_schema_arguments/README.md).

The patch preserves original token IDs during Qwen3 final parsing, reports an
empty STOP as a Responses failure, and emits the correct incomplete terminal
event under LENGTH. It does not prevent premature model EOS. Chat Completions
retains incremental argument streaming; Responses buffers converted XML
arguments until the function call closes. ID-aware replay is scoped to Qwen3
so unrelated parser configurations retain their existing path.
Responses rejects unoffered names explicitly; Chat Completions retains its
earlier passthrough contract so the client can handle an unavailable tool.
Known namespace aliases are still resolved first. Clients must validate names
before executing any tool call.

## CPU regression suite

Build the candidate recipe from this branch, then provide compatible local
Qwen tokenizer metadata. No model weights or GPU are needed for these tests.

```bash
docker build -t qwen38-flash-dgx:v0.33 .
docker run --rm --network none --runtime runc --entrypoint python3 \
  -v "$PWD/tests/issue31_safe_output:/qa:ro" \
  -v /path/to/local/tokenizer:/tokenizer:ro \
  -e QWEN_TOKENIZER=/tokenizer \
  qwen38-flash-dgx:v0.33 /qa/run_qa.py
```

The runner loads installed vLLM modules without runtime source overlays and
runs 67 regression methods plus 21 literal-byte cases. The only changes to the
contributor's fixtures initialize `enable_per_request_metrics=False`, a new
v0.31 service attribute; their assertions are retained.

The separate official v0.31 parser-engine suite, including existing recipe
patch-12/13 tests, contains 4,782 cases. On the baseline and candidate alike,
4,781 pass. The existing failure is
`TestTruncatedToolOpenerStreamParity::test_content_around_unpromoted_tool_block_stays_ordered`:
upstream expects malformed marker text to disappear, while the recipe's
literal-marker guard preserves it. The port must not introduce additional
failures.

## Client protocol reproduction

`export_sse.py` runs the real installed streaming producer and final aggregator
on synthetic engine frames. It exports visible LENGTH, hidden-reasoning LENGTH,
normal STOP and empty STOP streams. Run it with the same `/qa` and `/tokenizer`
mounts, an output directory mounted at `/out`, and `QA_LABEL=candidate`.

Replay the resulting JSON to the installed Codex CLI:

```bash
python3 tests/issue31_safe_output/codex_client_probe.py \
  /path/to/candidate-sse-fixtures.json --output /tmp/candidate-client.json \
  --require-correct
```

The probe uses a temporary local HTTP endpoint, ignores user configuration,
runs an ephemeral read-only session, and makes no model inference request to
an external provider. It checks one request and zero tool executions per case.
The candidate should report LENGTH and empty STOP as failed turns, while normal
STOP remains successful. The baseline replay records incorrect silent successes
without preventing the before/after comparison.

These synthetic checks complement real-inference Responses probes, throughput
measurements and the unchanged 55-scenario agentic tournament. Their results
must not be presented as proof of general model quality or as an EOS fix.

## Chat Completions compatibility (issue #56)

Using the same image, mounts and tokenizer variable as the CPU regression suite,
run `/qa/chat_compat_tests.py`. It exercises the installed streaming and full
Chat producers with synthetic engine frames: unknown-name passthrough, offered
namespace resolution, contiguous call indices, reasoning examples, concurrent
independent streams, preserved strict Responses behavior and rejection logs.
No model tokens are generated and no proposed tool is executed. Open WebUI and
natural long-context generations are outside these tests.

GX10 integration results and the porting differences are recorded in
[scope and validation](../../docs/SAFE-OUTPUT.md).
