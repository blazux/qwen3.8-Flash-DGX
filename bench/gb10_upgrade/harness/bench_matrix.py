#!/usr/bin/env python3
"""Streaming matrix, retrieval probes and bounded sustained load (stdlib only).

No forced EOS, no cache flush, no process restart, and no model-generated code
execution. Client concurrency can exceed the server's admission/KV capacity.
Every completed request is flushed to JSONL, including errors and truncations.
"""
import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import math
import random
import re
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

WORDS = "analysis system data model process service request response cache memory token network compute invoice contract customer product value result method detail context performance stable benchmark measure compare sequence document field".split()


def utc():
    return datetime.now(timezone.utc).isoformat()


def fetch(base, path, body=None, timeout=30):
    req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def stream(base, body, timeout):
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    first = last = None
    chunks = []
    content = []
    usage = {}
    finish = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            for raw in response:
                if not raw.startswith(b"data: ") or raw.strip() == b"data: [DONE]":
                    continue
                data = json.loads(raw[6:])
                if data.get("error"):
                    raise RuntimeError(str(data["error"]))
                for choice in data.get("choices", []):
                    delta = choice.get("delta", {})
                    text = (delta.get("reasoning_content") or delta.get("reasoning") or "") + (delta.get("content") or "")
                    if text:
                        now = time.perf_counter()
                        first = now if first is None else first
                        last = now
                        chunks.append(now - start)
                        content.append(text)
                    if choice.get("finish_reason") is not None:
                        finish = choice["finish_reason"]
                if data.get("usage"):
                    usage = data["usage"]
        tokens = usage.get("completion_tokens", 0)
        if not usage or finish is None:
            raise RuntimeError("incomplete stream: final usage or finish_reason missing")
        text = "".join(content)
        return dict(ok=True, elapsed_s=time.perf_counter() - start,
                    ttft_s=None if first is None else first - start,
                    decode_tokens_per_s=(tokens - 1) / (last - first)
                    if tokens > 1 and last > first else None,
                    chunk_gap_p95_s=quantile([b - a for a, b in zip(chunks, chunks[1:])], .95),
                    text_sha256=hashlib.sha256(text.encode()).hexdigest(),
                    text=text, usage=usage, finish_reason=finish)
    except Exception as exc:
        detail = exc.read().decode(errors="replace")[:500] if isinstance(exc, urllib.error.HTTPError) else str(exc)
        return dict(ok=False, elapsed_s=time.perf_counter() - start, error=detail)


def quantile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * fraction) - 1)]


class Writer:
    def __init__(self, path, label):
        path.parent.mkdir(parents=True, exist_ok=True)
        # Never silently replace an earlier experiment.
        self.file = path.open("x")
        self.label = label
        self.lock = threading.Lock()

    def emit(self, row):
        row.update(label=self.label, timestamp_utc=utc())
        with self.lock:
            self.file.write(json.dumps(row) + "\n")
            self.file.flush()
        if row.get("kind") not in ("telemetry", "request", "repair"):
            print(json.dumps(row), flush=True)

    def close(self):
        self.file.close()


def runtime(container):
    try:
        obj = json.loads(subprocess.check_output(["docker", "inspect", container], timeout=15))[0]
        allowed = {"PYTORCH_ALLOC_CONF", "VLLM_PLE_MMAP_WORKERS", "VLLM_PLE_MMAP_PREWARM", "VLLM_PLE_MMAP_FAST_ROWS",
                   "VLLM_PLE_MMAP_MADVISE", "VLLM_MTP_DRAFT_VOCAB", "VLLM_QSA_DET_TOPK", "VLLM_QSA_EXACT_TOPK"}
        env = [v for v in obj["Config"]["Env"] if v.split("=", 1)[0] in allowed]
        cmd = obj["Config"]["Cmd"][:]
        cmd[0] = Path(cmd[0]).name  # no local host paths in the public record
        return dict(image_id=obj["Image"], image_tag=obj["Config"]["Image"], command=cmd, environment=env,
                    restart_count=obj["RestartCount"], oom_killed=obj["State"]["OOMKilled"])
    except Exception as exc:
        return {"error": str(exc)}


def metrics(base):
    values = {}
    with urllib.request.urlopen(base + "/metrics", timeout=5) as response:
        for line in response.read().decode().splitlines():
            if line.startswith("#") or not line:
                continue
            key, _, val = line.rpartition(" ")
            name = key.split("{", 1)[0]
            if any(x in name for x in ("num_requests_running", "num_requests_waiting", "cache_usage", "preemption",
                                       "spec_decode", "prompt_tokens_total", "generation_tokens_total", "request_success_total")):
                with contextlib.suppress(ValueError):
                    values[name] = values.get(name, 0.0) + float(val)
    return values


class Sampler:
    def __init__(self, base, writer, container):
        self.base, self.writer, self.container = base, writer, container
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        tick = 0
        while not self.stop.is_set():
            row = {"kind": "telemetry"}
            try:
                row["metrics"] = metrics(self.base)
                row["memory_kib"] = {k: int(v.split()[0]) for k, v in
                                     (line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
                                     if k in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree", "Cached", "Dirty")}
                vm = dict(line.split() for line in Path("/proc/vmstat").read_text().splitlines())
                row["vmstat"] = {k: int(vm[k]) for k in ("pswpin", "pswpout", "pgmajfault")}
                if tick % 6 == 0:
                    row["runtime"] = runtime(self.container)
                    try:
                        row["gpu"] = subprocess.check_output(
                            ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,power.draw", "--format=csv,noheader,nounits"],
                            timeout=5, stderr=subprocess.DEVNULL).decode().strip()
                    except Exception as exc:
                        row["gpu_error"] = str(exc)
            except Exception as exc:
                row["error"] = str(exc)
            self.writer.emit(row)
            tick += 1
            self.stop.wait(5)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=20)


def make_prompt(target, seed, nonce):
    rng = random.Random(f"{seed}:{nonce}")
    prefix = f"Synthetic benchmark request {nonce} {rng.getrandbits(64):016x}.\n"
    filler = " ".join(rng.choice(WORDS) for _ in range(max(1, target - 100)))
    suffix = "\nSummarize this synthetic document in a detailed technical paragraph. Do not mention the request identifier. /no_think"
    return prefix + filler + suffix


def body_for(model, prompt, output):
    return dict(model=model, messages=[{"role": "user", "content": prompt}], temperature=0,
                max_tokens=output, chat_template_kwargs={"enable_thinking": False}, stream=True,
                stream_options={"include_usage": True})


def wave(args, writer, model, target, concurrency, repetition, phase, prompts, output):
    started = time.perf_counter()
    barrier = threading.Barrier(concurrency)

    def run(index):
        barrier.wait(timeout=30)
        result = stream(args.base_url, body_for(model, prompts[index], output), args.request_timeout)
        result.update(kind="request", target_input_tokens=target, concurrency=concurrency,
                      repetition=repetition, phase=phase, client=index,
                      prompt_sha256=hashlib.sha256(prompts[index].encode()).hexdigest())
        # Raw benchmark text is not needed for throughput; retain hashes and integrity signals.
        result.pop("text", None)
        writer.emit(result)
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        rows = list(pool.map(run, range(concurrency)))
    wall = time.perf_counter() - started
    good = [r for r in rows if r["ok"]]
    writer.emit(dict(kind="wave", target_input_tokens=target, concurrency=concurrency,
                     repetition=repetition, phase=phase, wall_s=wall, requests=concurrency,
                     successful=len(good), failed=concurrency - len(good),
                     output_tokens=sum(r["usage"]["completion_tokens"] for r in good),
                     output_tokens_per_s=sum(r["usage"]["completion_tokens"] for r in good) / wall,
                     prompt_tokens=sum(r["usage"]["prompt_tokens"] for r in good),
                     ttft_p50_s=statistics.median(r["ttft_s"] for r in good if r["ttft_s"] is not None) if any(r.get("ttft_s") is not None for r in good) else None,
                     ttft_p95_s=quantile([r["ttft_s"] for r in good if r["ttft_s"] is not None], .95),
                     latency_p95_s=quantile([r["elapsed_s"] for r in good], .95)))
    return rows


def matrix(args, writer, model):
    started = time.monotonic()
    stream(args.base_url, body_for(model, "Explain prefix caching briefly. /no_think", 32), args.request_timeout)
    for target in args.contexts:
        for concurrency in args.concurrencies:
            repeats = args.repetitions if target <= 32768 else args.long_repetitions
            for rep in range(repeats):
                if time.monotonic() - started > args.seconds:
                    writer.emit(dict(kind="budget_exhausted", target_input_tokens=target, concurrency=concurrency, repetition=rep))
                    return
                prompts = [make_prompt(target, args.seed, f"matrix-{target}-{concurrency}-{rep}-{i}") for i in range(concurrency)]
                # Actual server usage is authoritative; targets are approximate token sizes.
                phases = getattr(args, "phases", ["fresh", "repeat"])
                fresh = (wave(args, writer, model, target, concurrency, rep, "fresh", prompts, args.output_tokens)
                         if "fresh" in phases else None)
                if "repeat" in phases and (fresh is None or all(r["ok"] for r in fresh)):
                    cached = wave(args, writer, model, target, concurrency, rep, "repeat", prompts, args.output_tokens)
                    if fresh is not None:
                        writer.emit(dict(kind="repeat_integrity", target_input_tokens=target, concurrency=concurrency, repetition=rep,
                                         identical=sum(a["text_sha256"] == b.get("text_sha256") for a, b in zip(fresh, cached)), total=concurrency))
                    else:
                        writer.emit(dict(kind="repeat_without_immediate_fresh", target_input_tokens=target,
                                         concurrency=concurrency, repetition=rep,
                                         limitation="supplementary repeat; cache warmth is not guaranteed"))
                if fresh is not None and not all(r["ok"] for r in fresh):
                    writer.emit(dict(kind="cell_failure", target_input_tokens=target, concurrency=concurrency, repetition=rep))
                    # Do not keep sending requests to a failed or unhealthy engine.
                    try:
                        fetch(args.base_url, "/v1/models", timeout=10)
                    except Exception:
                        writer.emit(dict(kind="server_unhealthy"))
                        return
                if args.cooldown:
                    time.sleep(args.cooldown)


def needles(args, writer, model):
    for target in args.contexts:
        for position in (.1, .5, .9):
            nonce = f"needle-{target}-{position}"
            rng = random.Random(f"{args.seed}:{nonce}")
            secret = f"N-{rng.getrandbits(64):016x}"
            filler = [rng.choice(WORDS) for _ in range(max(1, target - 150))]
            filler.insert(int(len(filler) * position), f"\nThe hidden project access marker is {secret}.\n")
            prompt = f"Document {nonce}.\n" + " ".join(filler) + "\nReturn only the hidden project access marker from this document. /no_think"
            result = stream(args.base_url, body_for(model, prompt, 64), args.request_timeout)
            writer.emit(dict(kind="needle", target_input_tokens=target, position=position, expected=secret,
                             passed=result["ok"] and result["text"].strip() == secret, **result))


def soak(args, writer, model):
    started = time.monotonic()
    rng = random.Random(args.seed)
    cycle = 0
    failures = 0
    while time.monotonic() - started < args.seconds:
        target = rng.choice(args.contexts)
        concurrency = rng.choice(args.concurrencies)
        prompts = [make_prompt(target, args.seed, f"soak-{cycle}-{i}") for i in range(concurrency)]
        rows = wave(args, writer, model, target, concurrency, cycle, "soak", prompts, args.output_tokens)
        failures += sum(not r["ok"] for r in rows)
        if failures >= 3:
            writer.emit(dict(kind="soak_stopped", reason="three request failures", failures=failures))
            return
        if cycle % 3 == 0:
            wave(args, writer, model, target, concurrency, cycle, "soak_repeat", prompts, args.output_tokens)
        # A deliberate client disconnect is not scored as a failed request.
        if cycle % 20 == 0:
            body = body_for(model, "Explain database transaction isolation and recovery in detail. /no_think", 4096)
            req = urllib.request.Request(args.base_url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            abort_start = time.perf_counter()
            with urllib.request.urlopen(req, timeout=60) as response:
                for raw in response:
                    if raw.startswith(b"data: ") and raw.strip() != b"data: [DONE]":
                        chunk = json.loads(raw[6:])
                        if any(c.get("delta", {}).get("content") for c in chunk.get("choices", [])):
                            break
            writer.emit(dict(kind="client_disconnect", after_first_text=True, elapsed_s=time.perf_counter() - abort_start))
        if cycle % 10 == 0:
            writer.emit(dict(kind="health_probe", result=fetch(args.base_url, "/v1/models", timeout=10)))
        cycle += 1
    writer.emit(dict(kind="soak_complete", elapsed_s=time.monotonic() - started, cycles=cycle, failures=failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("matrix", "needles", "soak"))
    parser.add_argument("--base-url", default="http://localhost:18300")
    parser.add_argument("--container", default="qwen38-flash")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", default="1024,4096,8192,16384,32768,65536,131072,262144")
    parser.add_argument("--concurrencies", default="1,2,4,8,12")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--long-repetitions", type=int, default=1)
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--request-timeout", type=int, default=1200)
    parser.add_argument("--seconds", type=int, default=10800)
    parser.add_argument("--cooldown", type=float, default=0)
    parser.add_argument("--phases", default="fresh,repeat",
                        help="Matrix phases to measure; repeat alone is supplementary, not necessarily warm")
    args = parser.parse_args()
    args.base_url = args.base_url.rstrip("/")
    args.contexts = [int(v) for v in args.contexts.split(",")]
    args.concurrencies = [int(v) for v in args.concurrencies.split(",")]
    args.phases = args.phases.split(",")
    if args.phases not in (["fresh"], ["repeat"], ["fresh", "repeat"]):
        parser.error("phases must be fresh, repeat, or fresh,repeat")
    if args.mode != "matrix" and args.phases != ["fresh", "repeat"]:
        parser.error("phase selection applies only to matrix mode")
    if any(v <= 0 for v in args.contexts + args.concurrencies) or args.seconds <= 0:
        parser.error("sizes, concurrencies and seconds must be positive")
    writer = Writer(args.output, args.label)
    model = fetch(args.base_url, "/v1/models")["data"][0]["id"]
    writer.emit(dict(kind="metadata", mode=args.mode, model=model, seed=args.seed,
                     contexts=args.contexts, concurrencies=args.concurrencies, output_tokens=args.output_tokens,
                     repetitions=args.repetitions, long_repetitions=args.long_repetitions, phases=args.phases,
                     runtime=runtime(args.container),
                     method=("simultaneous chat streams; EOS honored; "
                             + ("fresh and immediate repeat" if args.phases == ["fresh", "repeat"]
                                else "supplementary selected phase; no guaranteed warm prefix")
                             + "; OS page cache not flushed")))
    try:
        with Sampler(args.base_url, writer, args.container):
            globals()[args.mode](args, writer, model)
        writer.emit(dict(kind="finished"))
    finally:
        writer.close()


if __name__ == "__main__":
    main()
