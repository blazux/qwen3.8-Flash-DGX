#!/usr/bin/env python3
"""Bounded, repeatable streaming benchmark against an existing server.

Measures client-observed TTFT, inter-token decode rate, and cached tokens.
Unique seeded prompts avoid accidental prefix reuse; each is repeated once.
Does not clear server caches or force generation past EOS.
"""
import argparse
import concurrent.futures
import json
import random
import time
import urllib.request
from pathlib import Path


def request(base, body):
    req = urllib.request.Request(
        base + "/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    first = last = None
    text = ""
    usage = {}
    with urllib.request.urlopen(req, timeout=120) as response:
        for raw in response:
            if not raw.startswith(b"data: ") or raw.strip() == b"data: [DONE]":
                continue
            chunk = json.loads(raw[6:])
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                part = (delta.get("reasoning_content") or delta.get("reasoning") or "") + (delta.get("content") or "")
                if part:
                    last = time.perf_counter()
                    if first is None:
                        first = last
                    text += part
            if chunk.get("usage"):
                usage = chunk["usage"]
    end = time.perf_counter()
    tokens = usage.get("completion_tokens", 0)
    return {
        "ttft_s": first - start if first else None,
        "elapsed_s": end - start,
        "decode_tokens_per_s": (tokens - 1) / (last - first)
        if tokens > 1 and last and last > first else None,
        "usage": usage, "text": text,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:18300")
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--seed", type=int, default=3100)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    with urllib.request.urlopen(base + "/v1/models", timeout=10) as response:
        model = json.load(response)["data"][0]["id"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    words = "ledger invoice payroll contract clause annex schedule amount date vendor total net gross tax due paid".split()
    with args.output.open("w") as out:
        def emit(row):
            row.update(label=args.label, model=model, seed=args.seed)
            out.write(json.dumps(row) + "\n")
            out.flush()
            print(json.dumps({k: v for k, v in row.items() if k != "text"}), flush=True)

        emit({"kind": "metadata", "seconds": args.seconds,
              "cache_note": "fresh application prefixes; OS page cache is warm; external traffic is not isolated"})
        cycle = 0
        while time.monotonic() - started < args.seconds:
            for size in (1600, 6400):
                if time.monotonic() - started >= args.seconds:
                    break
                rng = random.Random(args.seed + cycle * 2 + size)
                prompt = f"Benchmark dataset {rng.getrandbits(64)}\n" + " ".join(
                    rng.choice(words) + str(rng.randrange(10000)) for _ in range(size))
                prompt += "\nSummarize the dataset in a detailed paragraph. Answer:"
                body = {"model": model, "messages": [{"role": "user", "content": prompt + " /no_think"}], "max_tokens": 256,
                        "chat_template_kwargs": {"enable_thinking": False},
                        "temperature": 0, "stream": True,
                        "stream_options": {"include_usage": True}}
                pair = []
                for phase in ("fresh", "repeat"):
                    if time.monotonic() - started >= args.seconds:
                        break
                    try:
                        result = request(base, body)
                        pair.append(result["text"])
                        emit(dict(kind="sample", cycle=cycle, words=size,
                                  phase=phase, **result))
                    except Exception as exc:
                        emit(dict(kind="error", cycle=cycle, words=size,
                                  phase=phase, error=str(exc)))
                        raise
                if len(pair) == 2:
                    emit(dict(kind="determinism", cycle=cycle, words=size,
                              identical=pair[0] == pair[1]))
            if time.monotonic() - started < args.seconds:
                prompt = f"Benchmark batch {args.seed}-{cycle}. Explain how an operating system page cache works. Answer:"
                body = {"model": model, "messages": [{"role": "user", "content": prompt + " /no_think"}], "max_tokens": 256,
                        "chat_template_kwargs": {"enable_thinking": False},
                        "temperature": 0, "stream": True,
                        "stream_options": {"include_usage": True}}
                batch_start = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                    results = list(pool.map(lambda _: request(base, body), range(4)))
                elapsed = time.perf_counter() - batch_start
                emit(dict(kind="concurrent", cycle=cycle, concurrency=4,
                          elapsed_s=elapsed,
                          tokens_per_s=sum(r["usage"].get("completion_tokens", 0)
                                           for r in results) / elapsed))
            cycle += 1
        emit(dict(kind="finished", elapsed_s=time.monotonic() - started))


if __name__ == "__main__":
    main()
