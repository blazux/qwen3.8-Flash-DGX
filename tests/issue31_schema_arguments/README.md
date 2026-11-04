# Issue #31: schema-authoritative tool arguments (patch 21)

Originally a minimal vLLM v0.31 port of
[Chaos2522's schema2 repair](https://github.com/blazux/qwen3.8-Flash-DGX/issues/31#issuecomment-6291191080).
For the historical v0.32 port, use its
[release manifest and validation](../../docs/VLLM-032.md).
It applies after patch 19 including the issue #56 Chat compatibility follow-up.
`SOURCE.json` records the original archive, credit and adaptations;
`PORT-MANIFEST.json` records the historical v0.31 patch inputs and output hashes.
The original argument-validation helper is unchanged. The ten-file bundle is not
copied wholesale into the recipe.

The parser repairs a sole accidental `arguments` wrapper only if the original
fails the uniquely offered tool schema and the once-unwrapped object passes it.
An already valid original is preserved, including a legitimate `arguments`
property. Missing or ambiguous schemas do not authorize inferred repair; no
recursive decoding, type coercion or remote schema retrieval is performed.

Responses fails schema-invalid calls before exposing an executable function-call
item. Unclosed calls remain incomplete on LENGTH. Chat retains its existing
unknown-name passthrough and invalid-argument client recovery; normal arguments
still stream incrementally, while a possible wrapper waits for closure.

## Run the installed-code CPU tests

Build the recipe, then mount the complete `tests/` directory and compatible local
Qwen tokenizer metadata. No weights or GPU are required. The runner verifies all
nine installed parser/Responses source hashes before importing the test modules;
it does not overlay runtime source.

```bash
docker build -t qwen38-flash-dgx:v0.33 .
docker run --rm --network none --runtime runc --entrypoint python3 \
  -e VLLM_TARGET_DEVICE=cpu -e CUDA_VISIBLE_DEVICES= \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e QWEN_TOKENIZER=/tokenizer \
  -v "$PWD/tests:/tests:ro" \
  -v /path/to/local/tokenizer:/tokenizer:ro \
  qwen38-flash-dgx:v0.33 /tests/issue31_schema_arguments/run_qa.py \
  --manifest /tests/vllm032/PORT-MANIFEST.json
```

For a JSON report, mount a writable directory at `/out` and add
`--report /out/schema-qa.json` after the runner path.
Run the previous 67-method suite and 21 byte-fidelity cases separately with the
same mounts and image, replacing the runner path with
`/tests/issue31_safe_output/run_qa.py`.

`baseline_probe.py` reproduces the wrapper on the old image and the corrected
payload on the candidate through full and streaming Chat/Responses producers.
`namespace_probe.py` compares dotted namespace aliases with the canonical
`namespace__function` form. They use synthetic engine frames and execute no tools.

## Results and known limitations

GX10 CPU validation on 2026-10-20:

- Author schema suite: **22/23 pass**, with assertions retained and only the
  fixture API adapted to v0.31. The remaining namespace test expects dotted aliases
  that already fail on the reference. Canonical `namespace__function` works in
  both images and its wrapper is repaired by the candidate.
- Producer integration: **11/11 pass**, covering Chat and Responses, chunk sizes,
  final JSON, implicit reasoning transitions, schema changes between requests,
  ordinary incremental Chat arguments and invalid-call errors.
- Existing Chat compatibility: **10/10 pass**.
- Previous regression suite: **67/67 methods and 21/21 byte cases pass**.
- Official parser-engine suite plus existing patch-12/13 tests: **4,781/4,782 pass**,
  with the same pre-existing literal-marker-preservation failure as the reference.

The runner prints the author's raw namespace failure and records it explicitly;
it exits successfully only when that is the sole expected failure and all other
groups pass without skips. It does not report all 23 tests as passed.

`VALIDATION.json` records this scope. These are installed producer/parser tests
with synthetic engine frames, not GPU inference, live Codex execution, a throughput
benchmark or the author's full 63-test matrix. No model-quality or EOS fix is
claimed. See [behavior and validation](../../docs/SAFE-OUTPUT.md#schema-authoritative-tool-arguments-patch-21).
