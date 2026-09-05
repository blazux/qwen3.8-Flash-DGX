#!/usr/bin/env bash
# Quick check that the server is up, coherent, deterministic, that prefix caching hits,
# and measure prefill + decode.
#   scripts/smoke-test.sh [host:port]
set -euo pipefail
EP="${1:-localhost:18300}"
BASE="http://$EP"

echo ">> health"
curl -sf -m 5 "$BASE/health" >/dev/null && echo "   OK" || { echo "   not ready"; exit 1; }

echo ">> coherence"
curl -s -m 120 "$BASE/v1/completions" -H 'Content-Type: application/json' -d \
  '{"model":"qwen3.8-flash-next","prompt":"The capital of France is","max_tokens":12,"temperature":0}' \
  | python3 -c 'import json,sys;print("  ",repr(json.load(sys.stdin)["choices"][0]["text"]))'

echo ">> prefill (TTFT on a ~8k-token prompt), then determinism + prefix-cache hit on the same prompt"
python3 - "$BASE" <<'PY'
import json,sys,time,urllib.request,random
base=sys.argv[1]; random.seed(1)
words="ledger invoice payroll contract clause annex schedule amount date vendor total net gross tax due paid".split()
prompt=" ".join(random.choice(words)+(str(random.randint(1,9999)) if random.random()<0.2 else "") for _ in range(5800))
prompt+="\n\nQuestion: list three numbers that appear right after the word 'invoice', then name the most frequent word. Answer:"
def hits():
    """vLLM prefix-cache hit counter, or None when /metrics is unavailable."""
    try:
        txt=urllib.request.urlopen(base+"/metrics",timeout=10).read().decode()
    except Exception:
        return None
    tot,seen=0.0,False
    for line in txt.splitlines():
        if line.startswith("vllm:prefix_cache_hits_total"):
            try: tot+=float(line.rsplit(" ",1)[1]); seen=True
            except ValueError: pass
    return tot if seen else None
def run(max_tokens):
    t=time.time()
    req=urllib.request.Request(base+"/v1/completions",
        data=json.dumps({"model":"qwen3.8-flash-next","prompt":prompt,"max_tokens":max_tokens,"temperature":0,"logprobs":3}).encode(),
        headers={"Content-Type":"application/json"})
    r=json.load(urllib.request.urlopen(req,timeout=600)); dt=time.time()-t
    c=r["choices"][0]; return r["usage"]["prompt_tokens"], dt, c["text"], c["logprobs"]["top_logprobs"][0]
n,dt,t1,lp1=run(1)
print(f"   {n} tok in {dt:.2f}s  =>  {n/dt:.0f} tok/s prefill (cold)")
h1=hits()
n,dt2,t2,lp2=run(1)
h2=hits()
if h1 is not None and h2 is not None:
    # The wall-clock comparison cannot separate the two effects: with the PLE table on
    # NVMe, a warm page cache alone also speeds up the second pass (HOW-IT-WORKS:139).
    verdict='prefix-cache HIT' if h2>h1 else 'NO prefix-cache hit (serve with PREFIX_CACHE=1?)'
    print(f"   same prompt again: {dt2:.2f}s  =>  {verdict} (hits {h1:.0f} -> {h2:.0f})")
else:
    print(f"   same prompt again: {dt2:.2f}s  =>  timing only: {'consistent with a HIT' if dt2 < dt/2 else 'no speedup'} (no /metrics; a warm page cache alone can halve this)")
same = t1==t2 and all(abs(lp1.get(k,-99)-lp2.get(k,-99))<1e-6 for k in set(lp1)|set(lp2))
print(f"   first-token logprobs identical across runs: {'YES (deterministic)' if same else 'NO (stock top-k kernel? DET_TOPK=0 EXACT_TOPK=0)'}")
PY

echo ">> decode (real answer, greedy — never use ignore_eos with this model)"
python3 - "$BASE" <<'PY'
import json,sys,time,urllib.request
base=sys.argv[1]
msgs=[{"role":"user","content":"Explain in about 300 words how a page cache works and why random reads from an NVMe-backed mmap get faster over time. /no_think"}]
t=time.time()
req=urllib.request.Request(base+"/v1/chat/completions",
    data=json.dumps({"model":"qwen3.8-flash-next","messages":msgs,"max_tokens":400,"temperature":0}).encode(),
    headers={"Content-Type":"application/json"})
r=json.load(urllib.request.urlopen(req,timeout=600)); dt=time.time()-t
n=r["usage"]["completion_tokens"]
print(f"   {n} tok in {dt:.2f}s  =>  {n/dt:.1f} tok/s end-to-end (includes TTFT; steady-state decode is higher, see README)")
PY
