#!/usr/bin/env python3
# vllm_watch.py — live, colour-coded per-session view of what the server is doing: prompts
# (system prompt masked), reasoning, outputs and engine stats, interleaved from `docker logs`.
# Contributed by @0x3dlux (issue #12). Needs the server started with LOG_REQUESTS=1
# (scripts/serve.sh adds VLLM_LOGGING_LEVEL=DEBUG --enable-log-requests --enable-log-outputs);
# that logs every prompt and answer to the Docker json log, so keep it for debugging sessions.
#   tools/vllm_watch.py [--name qwen38-flash] [--lines 200]
import argparse
import ast
import hashlib
import re
import signal
import subprocess
import sys

IM_START = chr(60) + "|im_" + "start|" + chr(62)
IM_END = chr(60) + "|im_" + "end|" + chr(62)

PROMPT_RE = re.compile(r"Request (chatcmpl-[0-9a-f]+) details: prompt: ")
DELTA_RE = re.compile(r"Generated response (chatcmpl-[0-9a-f]+) \(streaming delta\): output: ")
COMPLETE_RE = re.compile(r"Generated response (chatcmpl-[0-9a-f]+) \(streaming complete\): output: ")
NONSTREAM_RE = re.compile(r"Generated response (chatcmpl-[0-9a-f]+): output: ")
STATS_RE = re.compile(
    r"Engine [0-9]+: Avg prompt throughput: ([\d.]+) tokens/s, "
    r"Avg generation throughput: ([\d.]+) tokens/s, "
    r"Running: (\d+) reqs, Waiting: (\d+) reqs, "
    r"GPU KV cache usage: ([\d.]+)%, Prefix cache hit rate: ([\d.]+)%"
)

COLORS = ["36", "33", "35", "32", "94", "93", "96", "95"]
RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
TAG_START_RE = re.compile(r"\[(reasoning|tool_calls?|content):")
MAX_THREADS = 4096


def fps(turns):
    return [(r, hashlib.blake2b(c.encode("utf8"), digest_size=8).digest()) for r, c in turns]


def color(tid):
    return "\033[" + COLORS[tid % len(COLORS)] + "m"


class Watcher:
    def __init__(self):
        self.threads = {}
        self.req_color = {}
        self.streamed = set()
        self.next_tid = 0

    def write(self, text, code):
        if IM_START in text or IM_END in text:
            text = text.replace(IM_START, "[im_start]").replace(IM_END, "[im_end]")
        sys.stdout.write(code + text + RESET)
        sys.stdout.flush()

    def parse_turns(self, text):
        turns = []
        for part in text.split(IM_START)[1:]:
            role, _, content = part.partition("\n")
            role = role.split(" ")[0].strip()
            tail = content.rstrip("\n")
            if tail.endswith(IM_END):
                turns.append((role, tail[: -len(IM_END)]))
        return turns

    def handle_prompt(self, rid, lit):
        try:
            raw = ast.literal_eval(lit)
        except (ValueError, SyntaxError):
            return
        if not isinstance(raw, str):
            return
        turns = self.parse_turns(raw)
        new_fps = fps(turns)
        best_tid, best_k = None, 0
        for tid, tfps in self.threads.items():
            k = 0
            n = min(len(tfps), len(new_fps))
            while k < n and tfps[k] == new_fps[k]:
                k += 1
            if k > best_k:
                best_k, best_tid = k, tid
        if best_tid is not None and best_k >= 2:
            tid, k = best_tid, best_k
            self.threads.pop(tid)
        elif rid in self.req_color and not self.threads.get(self.req_color[rid]):
            tid, k = self.req_color[rid], 0
        else:
            tid = self.next_tid
            self.next_tid += 1
            k = 0
        self.threads[tid] = new_fps
        self.req_color[rid] = tid
        while len(self.threads) > MAX_THREADS:
            del self.threads[next(iter(self.threads))]
        code = color(tid)
        for role, content in turns[k:]:
            if role == "system":
                self.write("<system prompt>", code)
            elif role == "user":
                self.write(content.rstrip("\n"), code)
            else:
                continue
            sys.stdout.write("\n")
            sys.stdout.flush()

    def handle_delta(self, rid, lit):
        try:
            body = ast.literal_eval(lit)
        except (ValueError, SyntaxError):
            return
        if not isinstance(body, str) or not body:
            return
        tid = self.req_color.get(rid)
        if tid is None:
            tid = self.next_tid
            self.next_tid += 1
            self.req_color[rid] = tid
        base = color(tid)
        self.streamed.add(rid)
        self.emit_parts(body, base)

    def handle_complete(self, rid, lit):
        if rid in self.streamed:
            return
        try:
            body = ast.literal_eval(lit)
        except (ValueError, SyntaxError):
            return
        if not isinstance(body, str) or not body:
            return
        tid = self.req_color.get(rid)
        if tid is None:
            tid = self.next_tid
            self.next_tid += 1
            self.req_color[rid] = tid
        self.emit_parts(body, color(tid))

    def emit_parts(self, body, base):
        starts = [(m.start(), m.end(), m.group(1)) for m in TAG_START_RE.finditer(body)]
        pos = 0
        for i, (s, e, tag) in enumerate(starts):
            if s > pos:
                self.write(body[pos:s], base)
            bound = starts[i + 1][0] if i + 1 < len(starts) else len(body)
            j = body.rfind("]", e, bound)
            if j == -1:
                text, pos = body[e:bound], bound
            else:
                text, pos = body[e:j], j + 1
            if text.startswith(" "):
                text = text[1:]
            code = base
            if tag == "reasoning":
                code = DIM + base
            elif tag.startswith("tool"):
                code = BOLD + base
            self.write(text, code)
        if pos < len(body):
            self.write(body[pos:], base)

    def handle_stats(self, m):
        pt, gt, run, wait, kv, hit = m.groups()
        if float(pt) == 0 and float(gt) == 0 and int(run) == 0 and int(wait) == 0:
            return
        sys.stdout.write(
            DIM
            + "[stats] prompt %s tok/s | gen %s tok/s | run %s wait %s | KV %s%% | prefix hit %s%%"
            % (pt, gt, run, wait, kv, hit)
            + RESET + "\n"
        )
        sys.stdout.flush()

    def process(self, line):
        m = PROMPT_RE.search(line)
        if m:
            tail = line[m.end():]
            b = tail.rfind(", prompt_token_ids:")
            if b > 0:
                tail = tail[:b]
            else:
                tail = re.sub(r", prompt_embeds.*$", "", tail)
            self.handle_prompt(m.group(1), tail.rstrip())
            return
        m = DELTA_RE.search(line)
        if m:
            tail = line[m.end():]
            c = tail.rfind(", finish_reason:")
            if c >= 0:
                tail = tail[:c]
            self.handle_delta(m.group(1), tail.rstrip())
            return
        m = COMPLETE_RE.search(line)
        if m:
            tail = line[m.end():]
            c = tail.rfind(", finish_reason:")
            if c >= 0:
                tail = tail[:c]
            self.handle_complete(m.group(1), tail.rstrip())
            return
        m = NONSTREAM_RE.search(line)
        if m:
            tail = line[m.end():]
            c = tail.rfind(", finish_reason:")
            if c >= 0:
                tail = tail[:c]
            self.handle_complete(m.group(1), tail.rstrip())
            return
        m = STATS_RE.search(line)
        if m:
            self.handle_stats(m)


def main():
    ap = argparse.ArgumentParser(description="distill vLLM container logs to a chat-like view")
    ap.add_argument("--name", default="qwen38-flash", help="docker container name")
    ap.add_argument("--lines", type=int, default=200, help="history lines to replay before following")
    args = ap.parse_args()
    cmd = ["docker", "logs", "-n", str(args.lines), "-f", args.name]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, errors="replace", bufsize=1,
    )
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(130))
    watcher = Watcher()
    try:
        for line in proc.stdout:
            watcher.process(line.rstrip("\n"))
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        proc.terminate()
        try:
            proc.wait(5)
        except subprocess.TimeoutExpired:
            proc.kill()
        sys.stdout.write(RESET + "\n")


if __name__ == "__main__":
    main()
