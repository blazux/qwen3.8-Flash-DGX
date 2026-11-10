#!/usr/bin/env python3
"""Small synthetic tool-using repository-repair test, NOT SWE-bench Verified.

Generated programs run only in a no-network, no-mount, resource-limited Docker
container. The agent cannot edit tests and never receives hidden tests.
"""
import argparse
import concurrent.futures
import copy
import hashlib
import json
import subprocess
import time
import uuid
from pathlib import Path

from bench_matrix import Writer, fetch, utc
from swe_micro_tasks import TASKS

SANDBOX = """
import json, os, pathlib, subprocess, sys, tempfile
data = json.load(sys.stdin)
root = pathlib.Path(tempfile.mkdtemp(prefix='repair-', dir='/tmp'))
for name, content in data['files'].items():
    if '/' in name or chr(92) in name or not name.endswith('.py'): raise ValueError('invalid filename')
    (root / name).write_text(content)
if data.get('hidden'):
    (root / 'test_hidden.py').write_text(data['hidden'])
try:
    result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-v'], cwd=root,
                            capture_output=True, text=True, timeout=20)
    print(json.dumps({'passed': result.returncode == 0, 'returncode': result.returncode,
                      'stdout': result.stdout[-8000:], 'stderr': result.stderr[-8000:]}))
except subprocess.TimeoutExpired:
    print(json.dumps({'passed': False, 'timeout': True}))
"""


def sandbox(image, files, hidden=None):
    name = "qwen-repair-" + uuid.uuid4().hex[:12]
    cmd = ["docker", "run", "--rm", "-i", "--name", name, "--network", "none", "--read-only",
           "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "64",
           "--memory", "512m", "--cpus", "1", "--user", "65534:65534",
           "--tmpfs", "/tmp:rw,nosuid,nodev,noexec,size=128m,mode=1777",
           "--entrypoint", "python3", image, "-c", SANDBOX]
    try:
        result = subprocess.run(cmd, input=json.dumps(dict(files=files, hidden=hidden)),
                                text=True, capture_output=True, timeout=40)
        if result.returncode != 0:
            return dict(passed=False, harness_error=result.stderr[-2000:])
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=15)
        return dict(passed=False, harness_timeout=True)


def schema(name, description, properties, required=()):
    return dict(type="function", function=dict(name=name, description=description,
                parameters=dict(type="object", properties=properties, required=list(required), additionalProperties=False)))


TOOLS = [schema("list_files", "List the repository files.", {}),
         schema("read_file", "Read a repository file.", {"path": {"type": "string"}}, ["path"]),
         schema("write_file", "Replace an implementation file. Tests are read-only.",
                {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
         schema("run_tests", "Run visible unittest tests in the sandbox.", {})]


def run_case(args, model, task, repetition):
    files = copy.deepcopy(task["files"])
    messages = [dict(role="system", content="You are fixing a small Python repository. Inspect files using tools, implement the reported fix and run tests. Tests cannot be edited. Use only the provided tools. Finish with a short explanation after tests pass. Hidden tests will check the specified contract."),
                dict(role="user", content=task["issue"])]
    transcript = []
    started = time.monotonic()
    calls = 0
    errors = 0
    final = ""
    try:
        for step in range(args.steps):
            data = fetch(args.base_url, "/v1/chat/completions", dict(
                model=model, messages=messages, tools=TOOLS, tool_choice="auto",
                temperature=args.temperature, seed=args.seed + repetition,
                max_tokens=args.max_tokens, reasoning_effort="medium"), timeout=600)
            choice = data["choices"][0]
            message = choice["message"]
            transcript.append(dict(kind="response", step=step, message=message,
                                   usage=data.get("usage"), finish_reason=choice.get("finish_reason")))
            assistant = {k: message[k] for k in ("role", "content", "tool_calls") if k in message}
            messages.append(assistant)
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                final = message.get("content") or ""
                break
            for tool in tool_calls:
                calls += 1
                try:
                    name = tool["function"]["name"]
                    params = json.loads(tool["function"]["arguments"])
                    if name == "list_files":
                        result = sorted(files)
                    elif name == "read_file":
                        result = files[params["path"]]
                    elif name == "write_file":
                        if params["path"] not in task["editable"]:
                            raise ValueError("only implementation files are editable")
                        if not isinstance(params["content"], str) or len(params["content"]) > 65536:
                            raise ValueError("invalid file content")
                        files[params["path"]] = params["content"]
                        result = "File updated."
                    elif name == "run_tests":
                        result = sandbox(args.image, files)
                    else:
                        raise ValueError("unknown tool")
                except Exception as exc:
                    errors += 1
                    result = {"error": str(exc)}
                transcript.append(dict(kind="tool", name=tool["function"]["name"], result=result))
                messages.append(dict(role="tool", tool_call_id=tool["id"], content=json.dumps(result)))
        score = sandbox(args.image, files, task["hidden"])
        return dict(kind="repair", task=task["name"], repetition=repetition, passed=score["passed"],
                    score=score, tool_calls=calls, tool_errors=errors, final=final,
                    elapsed_s=time.monotonic() - started, transcript=transcript, final_files=files,
                    issue_sha256=hashlib.sha256(task["issue"].encode()).hexdigest())
    except Exception as exc:
        return dict(kind="repair", task=task["name"], repetition=repetition, passed=False,
                    error=str(exc), tool_calls=calls, tool_errors=errors,
                    elapsed_s=time.monotonic() - started, transcript=transcript)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:18300")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image", default="qwen-eval-harness:20260827-r8")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=.2)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        for task in TASKS:
            result = sandbox(args.image, task["files"], task["hidden"])
            if result.get("harness_error") or result.get("harness_timeout") or result["passed"]:
                raise RuntimeError(f"invalid failing fixture {task['name']}: {result}")
            print(task["name"], "original fixture correctly fails", flush=True)
        return
    model = fetch(args.base_url, "/v1/models")["data"][0]["id"]
    writer = Writer(args.output, args.label)
    writer.emit(dict(kind="metadata", benchmark="synthetic SWE micro repair; NOT SWE-bench Verified",
                     tasks=len(TASKS), repetitions=args.repetitions, concurrency=args.concurrency,
                     max_steps=args.steps, max_tokens_per_step=args.max_tokens, temperature=args.temperature,
                     seed=args.seed, reasoning_effort="medium", image=args.image,
                     fixtures_sha256=hashlib.sha256(Path(__file__).with_name("swe_micro_tasks.py").read_bytes()).hexdigest()))
    items = [(task, rep) for rep in range(args.repetitions) for task in TASKS]
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(run_case, args, model, task, rep) for task, rep in items]
            rows = []
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                writer.emit(row)
                rows.append(row)
                print(json.dumps({k: v for k, v in row.items() if k not in ("transcript", "final_files", "final", "score")}), flush=True)
        writer.emit(dict(kind="finished", passed=sum(r["passed"] for r in rows), total=len(rows)))
    finally:
        writer.close()


if __name__ == "__main__":
    main()
