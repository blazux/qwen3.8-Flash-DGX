# Lenovo GB10: 0.31 to 0.33 deployment comparison protocol

Declared on 2026-11-10 before completion of the first measurement arm.
The host has a single GB10 and 128 GB unified memory, runs other services,
and retains its OS page cache and swap. No claim of isolated engine causality
or statistically established improvement follows from this sequential run.

## Arms, in order

1. Existing vLLM 0.31 production container: NVIDIA hybrid checkpoint revision
   `fc694b54fb0174e0913e6adf86691ef85a4ead47`, MTP2, reduced draft vocabulary,
   BF16 KV, 500,000 maximum context, eight active sequences, 8,192 batch tokens,
   no CPU pinning, `PYTORCH_ALLOC_CONF=expandable_segments:True`.
2. Current blazux vLLM 0.33 image and patches, preserving those explicit serving
   choices as far as supported. New upstream fixes, runner behavior and dynamic
   KV capacity can still differ. This is a recipe comparison, not an engine-only
   experiment.
3. Upstream 0.33 `mtp-fast` default: MTP3, full draft vocabulary, NVFP4 draft
   proposal head, 13 GiB FP8 KV, BF16 recurrent state, 16 sequences, 16,384 batch
   tokens, and upstream CPU pinning. Retain checkpoint and 500k maximum context.

Build/download work is scheduled after the baseline suite, not alongside
measured requests. Each arm warms the endpoint before the matrix. The existing
container and image are preserved for rollback; no checkpoint reconversion is
planned. Use the existing public API port so downstream clients retain their
endpoint. Recreate serving only after readiness and basic compatibility checks.

## Identical suite per arm

Use the same frozen harness files and seed `20261110`:

- Streaming matrix: approximate 1,024 / 8,192 / 32,768 input tokens;
  1 / 4 / 8 simultaneous clients; two repetitions of fresh and immediate-repeat
  waves; greedy, thinking disabled, 128 maximum output tokens, EOS honored.
  Expected: 36 waves and 156 requests per arm. Record TTFT, aggregate output
  throughput including prefill/queueing, client-estimated streaming decode,
  actual prompt/output/cached-token counts, errors and output hashes.
- Three forced tool calls and three strict-schema JSON responses. Check exact
  function name and values. This is parser/serialization smoke coverage.
- Six synthetic Python repository repair tasks, one attempt each, two clients,
  medium reasoning, 4,096 output tokens per step, 12 steps maximum. Execute
  generated code only in the existing network-disabled, read-only, resource-
  limited Docker harness. This is not SWE-bench or HumanEval.
- Full HumanEvalPlus v0.1.10: all 164 task IDs, one greedy answer per task,
  medium reasoning, four generation clients, 4,096 output-token budget. Generate
  without executing code, then evaluate in a separate network-disabled,
  read-only Docker sandbox (4 GiB memory, four CPUs, 256 PIDs). Reuse the pinned
  dataset and oracle cache. Missing final answers and truncations remain scored
  failures. Generation and evaluation have 2,700/600-second stage limits.
- Nine exact-marker retrieval probes: 32k / 128k / 256k approximate input targets,
  at 10% / 50% / 90% document positions, 64 maximum output tokens.
- Ten-minute mixed traffic soak over 1k/8k and 1/4/8 clients, 128 output tokens,
  with prefix repeats, health probes and separately recorded deliberate client
  disconnects.

Timeouts: 420 seconds per matrix/retrieval request; matrix admission budget
1,800 seconds, whole-stage 2,250 seconds; tool/JSON 360 seconds; repair 900
seconds; retrieval 1,200 seconds; soak 1,100 seconds. Any timeout, missing cell,
failed answer or stage failure remains in the record. A normal process exit is
not proof of complete coverage. Audit artifacts before publishing conclusions.

## Interpretation and promotion

Compare matching prompt hashes and cells. Report fresh and repeat separately;
an application-fresh prompt does not guarantee zero cached tokens. Preserve
version/profiling differences and report quality gates before promoting the
fast profile. Six repair tasks and nine needles cannot establish general
quality equivalence. Streaming decode is a client estimate, especially with
speculative token bursts. A ten-minute soak establishes only that interval.
Driver warnings, restarts and host-memory telemetry are separate from HTTP
success. If the default profile regresses a gate, investigate and retain a
verified 0.33 conservative profile or restore the preserved original.

The overall execution window is approximately four hours, including build,
verification and preparing a reviewable upstream contribution. Incomplete
stages must be disclosed rather than silently reduced or scored as successful.
