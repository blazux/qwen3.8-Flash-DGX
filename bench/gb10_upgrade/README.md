# Frozen Lenovo GB10 upgrade suite

These are the harness sources used for the November 10, 2026 comparison, not
an alternative serving implementation. See
[the protocol](../../docs/GB10-031-033-PROTOCOL.md) for configuration, order,
limits and interpretation. Run one suite against one already-ready deployment:

```bash
python3 bench/gb10_upgrade/run_arm.py my-deployment
```

The runner records matrix, tool/JSON, six synthetic repairs, nine retrieval
probes and a ten-minute soak under `bench/gb10_upgrade/results/my-deployment/`.
It never starts/stops an inference server or changes its configuration. Use a
new label; outputs are not overwritten. The repair stage defaults to the local
`qwen-eval-harness:20260827-r8` image. To use another Python 3.12 sandbox image,
run `harness/bench_swe_micro.py` directly with `--image IMAGE` and the settings
in the protocol. Repair execution uses no host mounts, a read-only root,
network isolation and explicit memory/CPU/PID limits.

Retrieval runs sequentially, one request per declared document position, with
a 64-token output cap in `needles()`. Shared CLI metadata also prints the generic
`output_tokens` and `concurrencies` argument defaults in needle mode; those
fields do not control retrieval requests. Interpret them together with `mode`
and the frozen request generator, rather than as concurrent full-context load.

## HumanEval+

HumanEval+ is a separate stage. The measurements reuse a local Python 3.12
evaluation image (`sha256:f66a09509ae1db5395b2d57e29b3d90b597a9b93f52708211bc0f0cc75964648`)
containing EvalPlus commit `26d6d00bb1fd0fa37f39c99d5290da67891d1c5e`
(`0.3.1.dev44`). The dataset is HumanEvalPlus v0.1.10, with full dataset/oracle
hash `fe585eb4df8c88d844eeb463ea4d0302`. This local image is not published;
re-evaluation elsewhere requires an equivalent environment with that pinned
EvalPlus source and dataset. Do not substitute a newer evaluator and describe
it as the identical harness.

Mount `harness/bench_humaneval.py` read-only in that image and run:

```bash
python3 /bench.py --base-url http://127.0.0.1:18300 \
  --label my-deployment --output /out --concurrency 4 \
  --max-tokens 4096 --effort medium --timeout 420
```

Generation needs access to the inference endpoint and a writable output mount;
it does not execute answers. Mount the existing EvalPlus cache read-only at
`/cache/evalplus`, with `XDG_CACHE_HOME=/cache` and `HF_HUB_OFFLINE=1`.
Evaluate the resulting `samples.jsonl` separately:

```bash
docker run --rm --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges --pids-limit 256 --memory 4g --cpus 4 \
  --user "$(id -u):$(id -g)" \
  --tmpfs /tmp:rw,nosuid,nodev,noexec,size=1g,mode=1777 \
  -e XDG_CACHE_HOME=/cache -e HF_HUB_OFFLINE=1 \
  -v "$EVALPLUS_CACHE:/cache/evalplus:ro" \
  -v "$GENERATION_DIR:/samples:ro" -v "$EVALUATION_DIR:/out" \
  --entrypoint python3 qwen-eval-harness:20260827-r8 \
  -m evalplus.evaluate --dataset humaneval --samples /samples/samples.jsonl \
  --parallel 4 --version v0.1.10 --output_file /out/evaluation.json
```

Set the three mount variables to existing absolute directories. Only the
evaluation output directory is writable. The evaluation stage executes model
answers; keep the isolation and resource limits in place.

## Audit

With the three declared labels present, inspect actual coverage and prompt
pairing rather than relying on subprocess exit codes:

```bash
python3 bench/gb10_upgrade/audit.py --root /path/to/results --output summary.json
```

The JSON includes per-cell throughput, TTFT, actual cached-token counts,
quality counts and missing coverage. It does not infer an engine-only speedup,
general quality equivalence or indefinite stability from this bounded run.
