#!/usr/bin/env python3
"""Generate all HumanEval+ answers using the existing pinned EvalPlus harness.

Run inside an EvalPlus container. Never executes generated code. Evaluation must
be a separate network-disabled, resource-limited sandbox. This is coding pass@1,
not SWE-bench Verified, and its explicit reasoning budget is part of the result.
"""
import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
from evalplus.sanitize import sanitize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:18300")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--task-ids", help="Comma-separated numeric HumanEval IDs; explicit non-representative subset")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    with urllib.request.urlopen(base + "/v1/models", timeout=10) as response:
        model = json.load(response)["data"][0]["id"]
    problems = get_human_eval_plus(version="v0.1.10")
    if len(problems) != 164:
        raise RuntimeError(f"expected 164 problems, got {len(problems)}")
    if args.task_ids:
        ids = [f"HumanEval/{int(v)}" for v in args.task_ids.split(",")]
        if len(ids) != len(set(ids)) or any(v not in problems for v in ids):
            parser.error("task IDs must be unique and in the pinned dataset")
        problems = {k: problems[k] for k in ids}
    args.output.mkdir(parents=True, exist_ok=True)
    raw_path = args.output / "raw.jsonl"
    samples_path = args.output / "samples.jsonl"
    if raw_path.exists() or samples_path.exists():
        raise RuntimeError("refusing to overwrite existing generations")
    lock = threading.Lock()
    started = time.monotonic()
    instruction = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"
    with raw_path.open("x") as raw_file, samples_path.open("x") as sample_file:
        def run(item):
            task_id, problem = item
            prompt = instruction + f"\n```python\n{problem['prompt'].strip()}\n```"
            body = dict(model=model, messages=[dict(role="user", content=prompt)],
                        temperature=0, max_tokens=args.max_tokens, reasoning_effort=args.effort)
            start = time.perf_counter()
            row = dict(task_id=task_id, label=args.label,
                       prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
            try:
                request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                                 headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=args.timeout) as response:
                    data = json.load(response)
                choice = data["choices"][0]
                msg = choice["message"]
                content = msg.get("content") or ""
                reason = msg.get("reasoning_content") or msg.get("reasoning") or ""
                row.update(ok=True, content=content, reasoning=reason,
                           finish_reason=choice.get("finish_reason"), usage=data.get("usage", {}))
                solution = sanitize(content, entrypoint=problem["entry_point"]) if content.strip() else ""
                solution = solution if solution.strip() else "# MODEL_RETURNED_NO_FINAL_ANSWER\npass"
            except Exception as exc:
                row.update(ok=False, error=str(exc), content="")
                solution = "# REQUEST_FAILED\npass"
            row.update(elapsed_s=time.perf_counter() - start,
                       timestamp_utc=datetime.now(timezone.utc).isoformat())
            with lock:
                raw_file.write(json.dumps(row) + "\n")
                sample_file.write(json.dumps(dict(task_id=task_id, solution=solution)) + "\n")
                raw_file.flush()
                sample_file.flush()
            print(json.dumps({k: v for k, v in row.items() if k not in ("content", "reasoning")}), flush=True)
            return row

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            rows = list(pool.map(run, sorted(problems.items(), key=lambda item: int(item[0].split("/")[-1]))))
    metadata = dict(label=args.label, model=model, dataset="HumanEvalPlus-v0.1.10", tasks=len(rows),
                    evalplus_version=importlib.metadata.version("evalplus"), dataset_hash=get_human_eval_plus_hash(version="v0.1.10"),
                    concurrency=args.concurrency, max_tokens=args.max_tokens, reasoning_effort=args.effort,
                    temperature=0, samples_per_task=1, generation_seconds=time.monotonic() - started,
                    request_errors=sum(not r["ok"] for r in rows),
                    missing_final_answers=sum(not r.get("content", "").strip() for r in rows),
                    length_truncated=sum(r.get("finish_reason") == "length" for r in rows),
                    task_ids=sorted(problems, key=lambda k: int(k.split("/")[-1])),
                    instruction=instruction, method=("explicit subset; not a full HumanEval score" if args.task_ids else "all 164 tasks")
                    + "; official EvalPlus sanitizer; no code execution during generation")
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
