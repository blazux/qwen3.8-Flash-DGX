#!/usr/bin/env bash
# Greedy probe set — the gate for the NVFP4-MTP graft (scripts/prepare-mtp-graft.sh):
# run it against a BF16-MTP arm (MODE=hybrid) and the graft (MODE=hybrid-mtp), then diff
# the two JSON files. Greedy output must be byte-identical: every emitted token is the
# target model's argmax (the draft only changes how many tokens are accepted per step),
# so a diverging text means a mispaired drafter, not a quantization artefact.
#   scripts/greedy-probe.sh <label> [host:port]     # writes ~/q38-tmp/gate/<label>.json
set -euo pipefail
LABEL="${1:?usage: greedy-probe.sh <label> [host:port]}"
EP="${2:-localhost:18300}"
mkdir -p "$HOME/q38-tmp/gate"
python3 - "$EP" "$HOME/q38-tmp/gate/$LABEL.json" <<'PY'
import json, sys, time, urllib.request
base, out = sys.argv[1], sys.argv[2]
if not base.startswith("http"):
    base = "http://" + base
prompts = [
    "The capital of France is",
    "Write a haiku about a desktop supercomputer. /no_think",
    "Explain in about 300 words how a page cache works and why random reads from an NVMe-backed mmap get faster over time. /no_think",
    "Write a Python function that checks whether a string is a palindrome, with a docstring. /no_think",
    "Summarize the plot of Hamlet in four sentences, then name the speaker of 'To be, or not to be'. /no_think",
]
results = []
for i, p in enumerate(prompts):
    body = {"model": "qwen3.8-flash-next", "prompt": p, "max_tokens": 400,
            "temperature": 0, "logprobs": 3}
    t = time.time()
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    c = r["choices"][0]
    results.append({"prompt": p, "text": c["text"], "tokens": r["usage"]["completion_tokens"],
                    "seconds": round(time.time() - t, 2),
                    "tok_s": round(r["usage"]["completion_tokens"] / (time.time() - t), 2),
                    "first_logprobs": (c.get("logprobs") or {}).get("top_logprobs", [None])[0]})
    print(f"   [{i}] {results[-1]['tokens']} tok in {results[-1]['seconds']}s "
          f"({results[-1]['tok_s']} tok/s)")
json.dump(results, open(out, "w"), indent=1)
print(f">> wrote {out}")
PY
