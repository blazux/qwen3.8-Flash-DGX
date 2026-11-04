#!/usr/bin/env python3
"""One arm of a speculative-decoding A/B: measure a fixed workload matrix against a live server.

    python3 bench/ab.py --label mtp4-1m.block1 --out results/ab/mtp4-1m.block1.jsonl

Run it once per arm (scripts/ab-bench.sh does the boots); bench/ab_report.py compares the files.
Same --seed on both arms is what makes the comparison valid: every prompt is derived from the
seed rather than the clock, so both arms see a byte-identical prompt stream and the only
difference between the two result files is the serving recipe. Each REP gets its own seeded
prompt, because PREFIX_CACHE=1 makes a repeated prompt free — reusing one would measure the
radix cache instead of the model.

Every chat cell streams, so each one reports tok/s over the whole request (the convention the
README tables use), a decode-only tok/s with TTFT subtracted, and the prefill rate. A drafter
swap moves all three, and the gap between them is where its cost lands. Each cell is bracketed
with a /metrics snapshot, so throughput comes with the acceptance rate and mean acceptance
length that produced it (needs PROM_MULTIPROC=1 on the serving side: vLLM keeps EngineCore's
counters out of a single-process /metrics, which is why scripts/ab-bench.sh sets it for
both arms).

The gate cells are the five prompts of scripts/greedy-probe.sh, greedy at temperature 0. Their
text is hashed rather than stored: speculative decoding is lossless (the draft only decides how
many of its own proposals survive verification), so a differing hash means a mispaired drafter,
not noise. Everything else stores only a hash too, so a full-suite run is a few tens of KB.
"""
import argparse
import hashlib
import json
import os
import random
import re
import statistics
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (BASE_URL, CHAT_URL, CODE_PROMPT, HEADERS, MODEL,  # noqa: E402
                    chat)

TIMEOUT = 3600
WORDS = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike "
         "november oscar papa quebec romeo sierra tango uniform victor whiskey xray "
         "yankee zulu scheduler kernel latency throughput quantize tensor gradient cache "
         "pointer buffer register pipeline parallel entropy manifold").split()

# scripts/greedy-probe.sh's corpus — kept identical so the two gates agree.
GATE_PROMPTS = [
    "The capital of France is",
    "Write a haiku about a desktop supercomputer. /no_think",
    "Explain in about 300 words how a page cache works and why random reads from an "
    "NVMe-backed mmap get faster over time. /no_think",
    "Write a Python function that checks whether a string is a palindrome, with a "
    "docstring. /no_think",
    "Summarize the plot of Hamlet in four sentences, then name the speaker of "
    "'To be, or not to be'. /no_think",
]

SPEC_RE = re.compile(r"^vllm:spec_decode_(num_drafts|num_draft_tokens|num_accepted_tokens)"
                     r"_total(?:\{[^}]*\})?\s+(\S+)", re.M)
# Per accepted-position bucket: shows where in the K-token block acceptance dies.
SPEC_POS_RE = re.compile(r"^vllm:spec_decode_num_accepted_tokens_per_pos_total"
                         r"\{[^}]*le=\"(-?[0-9.]+)\"[^}]*\}\s+(\S+)", re.M)
METRICS_URL = (BASE_URL[:-len("/v1")] if BASE_URL.endswith("/v1") else BASE_URL) + "/metrics"


def post(url, body, timeout=None):
    req = urllib.request.Request(url, json.dumps(body).encode(), HEADERS)
    return json.loads(urllib.request.urlopen(req, timeout=timeout or TIMEOUT).read())


def sha(text):
    return hashlib.sha256((text or "").encode()).hexdigest()[:16]


def seeded_prompt(seed, approx_tokens, task):
    """A prompt of roughly approx_tokens tokens, unique to `seed`.

    1.3 words per token is bench/longctx.py's calibration on this checkpoint. Only the head has
    to differ for the run to be cache-free — the radix cache matches prefixes, not substrings —
    so the Record line carries the uniqueness and the body carries the size. The row records
    cached_tokens to prove it: a nonzero value means this cell measured the cache, not the model.
    """
    rng = random.Random(seed)
    body = " ".join(rng.choice(WORDS) for _ in range(max(1, int(approx_tokens / 1.3))))
    return f"Record {seed}. Below is an unindexed log excerpt.\n\n{body}\n\n{task}"


def spec_snapshot():
    """vLLM's cumulative spec-decode counters, or {} when /metrics has none."""
    try:
        req = urllib.request.Request(METRICS_URL, headers=HEADERS)
        txt = urllib.request.urlopen(req, timeout=10).read().decode()
    except Exception:
        return {}
    out = {}
    for name, val in SPEC_RE.findall(txt):
        out[name] = out.get(name, 0.0) + float(val)
    for bucket, val in SPEC_POS_RE.findall(txt):
        out.setdefault("pos", {})[bucket] = out.get("pos", {}).get(bucket, 0.0) + float(val)
    return out


def spec_delta(before, after):
    """Acceptance over just this cell — the counters are cumulative since boot."""
    keys = ("num_drafts", "num_draft_tokens", "num_accepted_tokens")
    if not before or not all(k in after for k in keys):
        return None
    d = {k: after[k] - before.get(k, 0.0) for k in keys}
    drafts, dtok, acc = d["num_drafts"], d["num_draft_tokens"], d["num_accepted_tokens"]
    if drafts <= 0:
        return None
    pos = {b: after.get("pos", {}).get(b, 0.0) - before.get("pos", {}).get(b, 0.0)
           for b in after.get("pos", {})}
    return {
        "drafts": int(drafts),
        "acceptance_rate": acc / dtok if dtok else None,
        # Tokens committed per step, including the one the target always emits.
        "mean_acceptance_length": 1 + acc / drafts,
        "accepted_per_pos": [int(v) for _, v in sorted(pos.items(), key=lambda kv: float(kv[0]))]
                            if pos else None,
    }


def timed(row, r):
    """Add the three rates a request implies to a row."""
    n, e2e, ttft = r["completion_tokens"], r["e2e"], r["ttft"]
    row["e2e"] = e2e
    row["tok_s"] = n / e2e if e2e else None
    row["decode_tok_s"] = ((n - 1) / (e2e - ttft)) if ttft and e2e > ttft and n > 1 else None
    row["prefill_tok_s"] = (r["prompt_tokens"] / ttft) if ttft and r["prompt_tokens"] else None


def run_cell(cell, out, reps_scale, block):
    for rep in range(max(1, round(cell["reps"] * reps_scale))):
        seed = cell["seed"] + rep
        before = spec_snapshot()
        t0 = time.time()
        row = {"cell": cell["id"], "rep": rep}
        try:
            if cell["kind"] == "gate":
                d = post(CHAT_URL.replace("/chat/completions", "/completions"),
                         {"model": MODEL, "prompt": cell["prompt"], "max_tokens": 400,
                          "temperature": 0, "logprobs": 3})
                c, usage = d["choices"][0], d["usage"]
                lp = ((c.get("logprobs") or {}).get("top_logprobs") or [None])[0]
                row.update(prompt_tokens=usage["prompt_tokens"],
                           completion_tokens=usage["completion_tokens"],
                           cached=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                           ttft=None, e2e=time.time() - t0,
                           tok_s=usage["completion_tokens"] / (time.time() - t0),
                           decode_tok_s=None, prefill_tok_s=None,
                           text_sha=sha(c["text"]),
                           first_logprobs={k: round(v, 4) for k, v in lp.items()} if lp else None)
            elif cell["kind"] == "fanout":
                n = cell["streams"]
                bodies = [seeded_prompt(seed * 97 + i, cell["tokens"], cell["task"])
                          for i in range(n)]
                res = [None] * n

                def stream_one(i):
                    try:
                        res[i] = chat(bodies[i], cell["max_tokens"], thinking=cell["thinking"],
                                      stream=True, timeout=TIMEOUT)
                    except Exception as exc:        # keep the reason, not just the count
                        res[i] = exc

                w0 = time.time()
                threads = [threading.Thread(target=stream_one, args=(i,)) for i in range(n)]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join()
                wall = time.time() - w0
                done = [x for x in res if isinstance(x, dict)]
                if len(done) != n:
                    bad = next((x for x in res if isinstance(x, Exception)), None)
                    raise RuntimeError(f"{n - len(done)} of {n} streams failed"
                                       + (f": {bad}" if bad else ""))
                ttfts = sorted(x["ttft"] for x in done if x["ttft"])
                per = [x["completion_tokens"] / x["e2e"] for x in done]
                row.update(prompt_tokens=sum(x["prompt_tokens"] for x in done),
                           cached=sum(x.get("cached_tokens") or 0 for x in done),
                           completion_tokens=sum(x["completion_tokens"] for x in done),
                           streams=n, e2e=wall,
                           aggregate_tok_s=sum(x["completion_tokens"] for x in done) / wall,
                           per_stream_tok_s=statistics.median(per),
                           ttft=ttfts[len(ttfts) // 2] if ttfts else None,
                           ttft_max=ttfts[-1] if ttfts else None,
                           text_sha=sha("|".join(x["text"] for x in done)))
            else:
                prompt = (seeded_prompt(seed, cell["tokens"], cell["task"]) if cell["tokens"]
                          else cell["prompt"])
                r = chat(prompt, cell["max_tokens"], thinking=cell["thinking"],
                         stream=True, timeout=TIMEOUT)
                row.update(prompt_tokens=r["prompt_tokens"],
                           completion_tokens=r["completion_tokens"], ttft=r["ttft"],
                           cached=r.get("cached_tokens"),
                           finish_reason=r["finish_reason"], text_sha=sha(r["text"]))
                timed(row, r)
        except Exception as exc:                    # one lost request must not cost the whole run
            row["error"] = f"{type(exc).__name__}: {exc}"
        row.update(kind="cell", block=block, spec=spec_delta(before, spec_snapshot()))
        if row.get("cached"):
            row["cache_hit"] = True
            print(f"   {cell['id']} rep {rep}: !! {row['cached']} prompt tokens served from "
                  "cache — this rep measures the radix cache, not the model", flush=True)
        out.write(json.dumps(row) + "\n")
        out.flush()
        if "error" in row:
            print(f"   {cell['id']:<18} rep {rep}  !! {row['error']}", flush=True)
        else:
            s = row["spec"] or {}
            print(f"   {cell['id']:<18} rep {rep}  {row.get('aggregate_tok_s') or row['tok_s']:>6.1f} t/s"
                  f"  decode {row.get('decode_tok_s') or float('nan'):>6.1f}"
                  f"  ttft {row.get('ttft') or 0:>6.2f}s"
                  f"  acc {s.get('acceptance_rate') if s.get('acceptance_rate') is not None else float('nan'):>6.1%}"
                  f"  len {s.get('mean_acceptance_length') or float('nan'):>5.2f}", flush=True)
        time.sleep(cell["gap"])


def build_cells(suite, seed):
    """The matrix. `fast` is roughly 15-20 min per arm; `full` adds the deep-context cells."""
    cells = []

    def add(kind, cid, seed_base, *, tokens=0, task="Reply with only: OK", prompt=None,
            max_tokens=128, thinking=False, reps=3, streams=1, gap=2):
        cells.append(dict(kind=kind, id=cid, seed=seed_base, tokens=tokens, task=task,
                          prompt=prompt, max_tokens=max_tokens, thinking=thinking,
                          reps=reps, streams=streams, gap=gap))

    # Single-stream decode — the headline numbers, on the content types where a drafter's
    # acceptance differs most: code is highly predictable, free prose much less so.
    add("chat", "decode/code", seed + 1000, prompt=CODE_PROMPT, max_tokens=700, reps=5)
    add("chat", "decode/code-think", seed + 1100, prompt=CODE_PROMPT, max_tokens=700,
        thinking=True, reps=3)
    add("chat", "decode/prose", seed + 1200, tokens=2000,
        task="Continue this log in prose, about 400 words. Answer:", max_tokens=512, reps=5)
    add("chat", "decode/reasoning", seed + 1300,
        prompt="A 40 kg rope of negligible stiffness hangs over a frictionless pulley at the "
               "top of a 12 m ladder leaning against a wall. Work out whether the system is in "
               "static equilibrium and what breaks first, showing the force balance.",
        max_tokens=1200, thinking=True, reps=3)
    # Prefill against context length: the point of a 1M-context profile, and where the
    # drafter's extra memory and verification cost land first.
    for tgt in (1000, 8000, 32000, 128000):
        add("chat", f"prefill/{tgt // 1000}k", seed + 2000 + tgt, tokens=tgt, reps=2)
    # Concurrency: aggregate throughput is where a drafter usually loses, because verification
    # joins a bigger batch (the step costs more) while each accepted token matters less.
    for n in (1, 4, 8):
        add("fanout", f"conc/{n}", seed + 3000 + n, tokens=1500,
            task="Summarize this record in one paragraph. Answer:", max_tokens=300,
            streams=n, reps=3, gap=3)
    if suite == "full":
        add("chat", "decode/code-long", seed + 4000, prompt=CODE_PROMPT, max_tokens=3000, reps=3)
        for tgt in (512000, 900000):
            add("chat", f"prefill/{tgt // 1000}k", seed + 2000 + tgt, tokens=tgt, reps=2, gap=5)
        for n in (16, 24):
            add("fanout", f"conc/{n}", seed + 3000 + n, tokens=1500,
                task="Summarize this record in one paragraph. Answer:", max_tokens=300,
                streams=n, reps=2, gap=4)
    for i, p in enumerate(GATE_PROMPTS):
        add("gate", f"gate/{i}", seed + 5000 + i, prompt=p, reps=1)
    return cells


def main():
    global TIMEOUT
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="-", help="JSONL to write ('-' = stdout)")
    ap.add_argument("--label", required=True, help="arm name, echoed into every row")
    ap.add_argument("--block", type=int, default=1,
                    help="which ABBA block this is; bench/ab_report.py pools blocks per arm")
    ap.add_argument("--suite", choices=("fast", "full"), default="fast")
    ap.add_argument("--seed", type=int, default=20261103,
                    help="must match the other arm exactly, or nothing is matched")
    ap.add_argument("--reps-scale", type=float, default=1.0)
    ap.add_argument("--only", default="", help="run only cells whose id contains this")
    ap.add_argument("--meta-file", default="",
                    help="JSON from scripts/ab-bench.sh: server args, KV pool, host state")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()
    TIMEOUT = args.timeout

    meta = {}
    if args.meta_file and os.path.exists(args.meta_file):
        with open(args.meta_file) as f:
            meta = json.load(f)

    out = sys.stdout if args.out == "-" else open(args.out, "w")
    t_start = time.time()
    out.write(json.dumps(dict(
        kind="meta", label=args.label, block=args.block, suite=args.suite, seed=args.seed,
        base_url=BASE_URL, model=MODEL, server=meta,
        host=dict(sysname=os.uname().sysname, release=os.uname().release,
                  machine=os.uname().machine, cpus=os.cpu_count(),
                  load=list(os.getloadavg())),
        started=time.strftime("%Y-%m-%dT%H:%M:%S%z"))) + "\n")
    out.flush()

    print(f">> {args.label}: warmup", flush=True)
    chat("hi", 8, stream=True)                       # kernels and CUDA graphs hot before anything counts
    counters = spec_snapshot()
    if not counters:
        print("   !! no vllm:spec_decode_* counters on /metrics — the throughput numbers are "
              "still valid but unexplained. Serve both arms with PROM_MULTIPROC=1 "
              "(scripts/ab-bench.sh does).", flush=True)

    cells = build_cells(args.suite, args.seed)
    if args.only:
        cells = [c for c in cells if args.only in c["id"]]
        print(f">> --only '{args.only}': {len(cells)} cell(s)", flush=True)
    for c in cells:
        print(f">> {c['id']}", flush=True)
        run_cell(c, out, args.reps_scale, args.block)

    out.write(json.dumps(dict(kind="end", label=args.label,
                              elapsed=time.time() - t_start,
                              wall=time.time(), spec=spec_snapshot())) + "\n")
    if out is not sys.stdout:
        out.close()
    print(f">> {args.label}: {time.time() - t_start:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
